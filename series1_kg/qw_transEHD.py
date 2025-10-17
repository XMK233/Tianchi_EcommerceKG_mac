import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
import os
from tqdm import tqdm
import zipfile

# 设置随机种子
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

scheme_type = "ensemble"

# 数据路径（请根据你的实际路径修改）
BASE_DIR = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac"
TRAIN_FILE_PATH = f"{BASE_DIR}/originalData/OpenBG500/OpenBG500_train.tsv"
TEST_FILE_PATH = f"{BASE_DIR}/originalData/OpenBG500/OpenBG500_test.tsv"
DEV_FILE_PATH = f"{BASE_DIR}/originalData/OpenBG500/OpenBG500_dev.tsv"
OUTPUT_FILE_PATH = f"{BASE_DIR}/preprocessedData/OpenBG500_test__{scheme_type}.tsv"

# 模型保存路径
MODEL_DIR = f"{BASE_DIR}/trained_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# 模型路径
TRAINED_MODEL_PATHS = {
    'TransE': f"{MODEL_DIR}/trained_model__transE.pth",
    'TransH': f"{MODEL_DIR}/trained_model__transH.pth",
    'TransD': f"{MODEL_DIR}/trained_model__transD.pth"
}

# 超参数
EMBEDDING_DIM = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 1
BATCH_SIZE = 256
NEGATIVE_SAMPLES = 10
MAX_LINES = None
MAX_HEAD_ENTITIES = None
LR_DECAY_STEP = 5
LR_DECAY_FACTOR = 0.1


# ==================== 数据集 ====================
class KnowledgeGraphDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, is_test=False, max_lines=None, is_train=False):
        self.triples = []
        self.is_train = is_train
        self._load_data(file_path, is_test, max_lines)

    def _load_data(self, file_path, is_test, max_lines):
        print(f"加载数据: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if max_lines:
                lines = lines[:max_lines]
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 3:
                    h, r, t = parts
                    self.triples.append((h, r, t))
                elif is_test and len(parts) == 2:
                    h, r = parts
                    self.triples.append((h, r, "<UNK>"))
        print(f"共加载 {len(self.triples)} 个三元组")

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]


def collate_fn(batch):
    h_list, r_list, t_list = zip(*batch)
    return list(h_list), list(r_list), list(t_list)


# ==================== 映射器（带过滤支持）====================
class EntityRelationMapper:
    def __init__(self):
        self.entity_to_id = {}
        self.id_to_entity = {}
        self.relation_to_id = {}
        self.id_to_relation = {}
        self.entity_count = 0
        self.relation_count = 0
        self.all_train_triples = []  # 用于过滤评估

    def build_mappings(self, *datasets):
        entities = set()
        relations = set()
        for dataset in datasets:
            for h, r, t in dataset.triples:
                entities.add(h)
                entities.add(t)
                relations.add(r)
                if dataset.is_train:
                    self.all_train_triples.append((h, r, t))

        for e in sorted(entities):
            self.entity_to_id[e] = self.entity_count
            self.id_to_entity[self.entity_count] = e
            self.entity_count += 1
        for r in sorted(relations):
            self.relation_to_id[r] = self.relation_count
            self.id_to_relation[self.relation_count] = r
            self.relation_count += 1


# ==================== TransE ====================
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.E = nn.Embedding(num_entities, dim)
        self.R = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.E.weight)
        nn.init.xavier_uniform_(self.R.weight)

    def forward(self, h, r, t):
        return torch.norm(self.E(h) + self.R(r) - self.E(t), p=1, dim=1)

    def get_score(self, h, r):
        return -torch.norm(self.E(h) + self.R(r).unsqueeze(1) - self.E.weight.unsqueeze(0), p=1, dim=2)


# ==================== TransH ====================
class TransH(nn.Module):
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.E = nn.Embedding(num_entities, dim)
        self.R = nn.Embedding(num_relations, dim)
        self.W = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.E.weight)
        nn.init.xavier_uniform_(self.R.weight)
        nn.init.xavier_uniform_(self.W.weight)

    def project(self, emb, norm):
        norm = torch.nn.functional.normalize(norm, p=2, dim=1)
        return emb - torch.sum(emb * norm, dim=1, keepdim=True) * norm

    def forward(self, h, r, t):
        h_emb = self.project(self.E(h), self.W(r))
        t_emb = self.project(self.E(t), self.W(r))
        return torch.norm(h_emb + self.R(r) - t_emb, p=1, dim=1)

    def get_score(self, h, r):
        h_emb = self.project(self.E(h), self.W(r))
        all_e = self.project(self.E.weight, self.W(r).unsqueeze(1))
        return -torch.norm(h_emb.unsqueeze(1) + self.R(r).unsqueeze(1) - all_e, p=1, dim=2)


# ==================== TransD ====================
class TransD(nn.Module):
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.dim = dim
        self.E = nn.Embedding(num_entities, dim)
        self.R = nn.Embedding(num_relations, dim)
        self.E_proj = nn.Embedding(num_entities, dim)
        self.R_proj = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.E.weight)
        nn.init.xavier_uniform_(self.R.weight)
        nn.init.xavier_uniform_(self.E_proj.weight)
        nn.init.xavier_uniform_(self.R_proj.weight)

    def project(self, e, r_proj):
        return e + torch.sum(e * r_proj, dim=1, keepdim=True)

    def forward(self, h, r, t):
        h_emb = self.project(self.E(h), self.R_proj(r))
        t_emb = self.project(self.E(t), self.R_proj(r))
        r_vec = self.R(r)
        return torch.norm(h_emb + r_vec - t_emb, p=1, dim=1)

    def get_score(self, h, r):
        h_emb = self.project(self.E(h), self.R_proj(r))
        all_e = self.E.weight
        all_e_proj = all_e + torch.sum(all_e * self.R_proj(r).unsqueeze(1), dim=2, keepdim=True)
        r_vec = self.R(r)
        return -torch.norm(h_emb.unsqueeze(1) + r_vec.unsqueeze(1) - all_e_proj, p=1, dim=2)


# ==================== 训练模型 ====================
def train_model(model, model_name, train_dataset, mapper, device):
    if os.path.exists(TRAINED_MODEL_PATHS[model_name]):
        print(f"[{model_name}] 已存在训练好的模型，跳过训练")
        return
    print(f"[{model_name}] 开始训练...")
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_FACTOR)
    model.to(device)
    model.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0
        progress = tqdm(loader, desc=f"{model_name} Epoch {epoch+1}")
        for h_list, r_list, t_list in progress:
            h = torch.tensor([mapper.entity_to_id[h] for h in h_list], device=device)
            r = torch.tensor([mapper.relation_to_id[r] for r in r_list], device=device)
            t = torch.tensor([mapper.entity_to_id[t] for t in t_list], device=device)
            neg_t = torch.randint(0, mapper.entity_count, (len(h), NEGATIVE_SAMPLES), device=device)
            pos_score = model(h, r, t)
            neg_score = model(h.unsqueeze(1).expand(-1, NEGATIVE_SAMPLES).contiguous().view(-1),
                              r.unsqueeze(1).expand(-1, NEGATIVE_SAMPLES).contiguous().view(-1),
                              neg_t.view(-1)).view(-1, NEGATIVE_SAMPLES)
            loss = torch.mean(torch.relu(pos_score.unsqueeze(1) - neg_score + 1.0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())
        scheduler.step()
        print(f"[{model_name}] Epoch {epoch+1} Loss: {epoch_loss / len(loader):.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'entity_count': mapper.entity_count,
        'relation_count': mapper.relation_count,
        'embedding_dim': EMBEDDING_DIM,
        'entity_to_id': mapper.entity_to_id,
        'relation_to_id': mapper.relation_to_id,
    }, TRAINED_MODEL_PATHS[model_name])
    print(f"[{model_name}] 模型已保存")


# ==================== 加载模型 ====================
def load_model(model_class, model_path, mapper, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = model_class(checkpoint['entity_count'], checkpoint['relation_count'], EMBEDDING_DIM)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


# ==================== 评估函数 ====================
def evaluate_model(model, dataset, mapper, device, k_list=(1, 3, 10)):
    print("🔍 开始在开发集上评估模型性能...")
    model.eval()
    hits_at = {k: 0.0 for k in k_list}
    mrr = 0.0
    count = 0

    with torch.no_grad():
        for h, r, t in tqdm(dataset.triples, desc="Evaluating"):
            try:
                h_id = torch.tensor([mapper.entity_to_id[h]], device=device)
                r_id = torch.tensor([mapper.relation_to_id[r]], device=device)
                t_id = torch.tensor([mapper.entity_to_id[t]], device=device)
            except KeyError:
                continue  # 跳过未登录实体

            # ========== 尾实体预测 (h, r, ?) ==========
            scores = model.get_score(h_id, r_id).squeeze(0)
            filter_mask = torch.ones(mapper.entity_count, device=device)
            for tail in get_filtered_tails(mapper, h, r):
                if tail in mapper.entity_to_id:
                    filter_mask[mapper.entity_to_id[tail]] = 0
            scores = scores.masked_fill(filter_mask == 0, -float('inf'))
            _, indices = torch.sort(scores, descending=True)
            rank = (indices == t_id).nonzero(as_tuple=True)[0].item() + 1
            for k in k_list:
                if rank <= k:
                    hits_at[k] += 1
            mrr += 1.0 / rank
            count += 1

            # ========== 头实体预测 (?, r, t) ==========
            if hasattr(model, 'get_score_for_head'):
                scores_h = model.get_score_for_head(r_id, t_id).squeeze(0)
            else:
                query_h = model.E(t_id) - model.R(r_id)
                scores_h = -torch.norm(query_h - model.E.weight, p=1, dim=1)
            filter_mask_h = torch.ones(mapper.entity_count, device=device)
            for head in get_filtered_heads(mapper, r, t):
                if head in mapper.entity_to_id:
                    filter_mask_h[mapper.entity_to_id[head]] = 0
            scores_h = scores_h.masked_fill(filter_mask_h == 0, -float('inf'))
            _, indices_h = torch.sort(scores_h, descending=True)
            rank_h = (indices_h == h_id).nonzero(as_tuple=True)[0].item() + 1
            for k in k_list:
                if rank_h <= k:
                    hits_at[k] += 1
            mrr += 1.0 / rank_h
            count += 1

    for k in hits_at:
        hits_at[k] /= count
    mrr /= count

    print("✅ 评估完成！")
    print(f"📊 HITS@1:  {hits_at[1]:.4f}")
    print(f"📊 HITS@3:  {hits_at[3]:.4f}")
    print(f"📊 HITS@10: {hits_at[10]:.4f}")
    print(f"📊 MRR:     {mrr:.4f}")

    return hits_at, mrr


def get_filtered_tails(mapper, h, r):
    return [t for h_, r_, t in mapper.all_train_triples if h_ == h and r_ == r]

def get_filtered_heads(mapper, r, t):
    return [h for h, r_, t_ in mapper.all_train_triples if r_ == r and t_ == t]


# ==================== 融合预测 ====================
def predict_ensemble(models_with_weights, test_dataset, mapper, device, max_head_entities=None):
    print("🔍 开始融合预测 (加权融合: TransE + TransH + TransD) ...")
    results = []
    triples = test_dataset.triples
    if max_head_entities:
        triples = triples[:max_head_entities]

    with torch.no_grad():
        for h, r, _ in tqdm(triples, desc="Ensemble Predict"):
            try:
                h_id = torch.tensor([mapper.entity_to_id[h]], device=device)
                r_id = torch.tensor([mapper.relation_to_id[r]], device=device)
            except KeyError:
                preds = [h] * 10
                results.append('\t'.join([h, r] + preds))
                continue

            all_scores = torch.zeros(mapper.entity_count, device=device)
            for name, (model, weight) in models_with_weights.items():
                if hasattr(model, 'get_query_embedding'):
                    query = model.get_query_embedding(h_id, r_id)
                    if isinstance(query, np.ndarray):
                        query = torch.tensor(query, device=device)
                else:
                    query = model.E(h_id) + model.R(r_id)
                scores = -torch.norm(query - model.E.weight, p=1, dim=1)
                all_scores += weight * scores

            _, topk_indices = torch.topk(all_scores, k=10)
            preds = [mapper.id_to_entity[i.item()] for i in topk_indices]
            results.append('\t'.join([h, r] + preds))

    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results) + '\n')
    print(f"✅ 融合结果已保存至: {OUTPUT_FILE_PATH}")

    zip_path = OUTPUT_FILE_PATH.replace(".tsv", ".zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUTPUT_FILE_PATH, arcname=os.path.basename(OUTPUT_FILE_PATH))
    print(f"✅ 已压缩为: {zip_path}")


# ==================== 主函数 ====================
def main():
    device = torch.device('mps') if torch.backends.mps.is_available() else \
             torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"🚀 使用设备: {device}")

    # 加载数据
    train_data = KnowledgeGraphDataset(TRAIN_FILE_PATH, max_lines=MAX_LINES, is_train=True)
    dev_data = KnowledgeGraphDataset(DEV_FILE_PATH, is_test=False, is_train=False)
    test_data = KnowledgeGraphDataset(TEST_FILE_PATH, is_test=True, is_train=False)

    mapper = EntityRelationMapper()
    mapper.build_mappings(train_data, dev_data, test_data)
    print(f"实体数: {mapper.entity_count}, 关系数: {mapper.relation_count}")

    model_classes = {
        'TransE': TransE,
        'TransH': TransH,
        'TransD': TransD,
    }

    # 训练所有模型
    for name, Cls in model_classes.items():
        model = Cls(mapper.entity_count, mapper.relation_count, EMBEDDING_DIM)
        train_model(model, name, train_data, mapper, device)

    # 加载模型并评估
    loaded_models_with_weight = {}
    for name, Cls in model_classes.items():
        model = load_model(Cls, TRAINED_MODEL_PATHS[name], mapper, device)
        weight = 1.0
        loaded_models_with_weight[name] = (model, weight)
        print(f"\n📈 正在评估模型: {name}")
        evaluate_model(model, dev_data, mapper, device)

    # 融合预测
    predict_ensemble(loaded_models_with_weight, test_data, mapper, device, MAX_HEAD_ENTITIES)
    print("🎉 所有任务完成！融合预测已生成，开发集评估已完成。")


if __name__ == "__main__":
    main()