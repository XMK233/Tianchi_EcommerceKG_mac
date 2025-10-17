import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
import os
from tqdm import tqdm
import zipfile
import faiss  # 加速 Top-K 搜索

# 设置随机种子
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

scheme_type = "es_fs_e10"

# 数据路径（请根据你的实际路径修改）
BASE_DIR = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac"
TRAIN_FILE_PATH = f"{BASE_DIR}/originalData/OpenBG500/OpenBG500_train.tsv"
TEST_FILE_PATH = f"{BASE_DIR}/originalData/OpenBG500/OpenBG500_test.tsv"
DEV_FILE_PATH = f"{BASE_DIR}/originalData/OpenBG500/OpenBG500_dev.tsv"
OUTPUT_FILE_PATH = f"{BASE_DIR}/preprocessedData/OpenBG500_test.tsv" 

# 模型保存路径
MODEL_DIR = f"{BASE_DIR}/trained_models/{scheme_type}"
os.makedirs(MODEL_DIR, exist_ok=True)

# 模型路径 
TRAINED_MODEL_PATHS = {
    'TransE': f"{MODEL_DIR}/transE.pth",
    'TransH': f"{MODEL_DIR}/transH.pth",
    'TransD': f"{MODEL_DIR}/transD.pth"
}

# 超参数
EMBEDDING_DIM = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 10 ##【TODO】这里可以修改多一点。
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


# ==================== 映射器 ====================
class EntityRelationMapper:
    def __init__(self):
        self.entity_to_id = {}
        self.id_to_entity = {}
        self.relation_to_id = {}
        self.id_to_relation = {}
        self.entity_count = 0
        self.relation_count = 0
        self.all_train_triples = []

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

    def get_query_embedding(self, h, r):
        return self.E(h) + self.R(r)


# ==================== TransH （已修复）====================
class TransH(nn.Module):
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.E = nn.Embedding(num_entities, dim)
        self.R = nn.Embedding(num_relations, dim)      # 关系向量 d_r
        self.W = nn.Embedding(num_relations, dim)    # 法向量 W (用于超平面)
        nn.init.xavier_uniform_(self.E.weight)
        nn.init.xavier_uniform_(self.R.weight)
        nn.init.xavier_uniform_(self.W.weight)

    def project(self, emb, w):  # 将实体投影到关系超平面上
        # w: [B, dim], emb: [B, dim]
        norm_w = torch.nn.functional.normalize(w, p=2, dim=1)  # 单位化法向量
        scale = torch.sum(emb * norm_w, dim=1, keepdim=True)  # <e, w>
        return emb - scale * norm_w  # e - <e, w> * w

    def forward(self, h, r, t):
        h_emb = self.E(h)  # [B, dim]
        t_emb = self.E(t)
        r_vec = self.R(r)
        W = self.W(r)

        h_proj = self.project(h_emb, W)
        t_proj = self.project(t_emb, W)

        return torch.norm(h_proj + r_vec - t_proj, p=1, dim=1)

    def get_query_embedding(self, h, r):
        h_emb = self.E(h)
        r_vec = self.R(r)
        W = self.W(r)
        h_proj = self.project(h_emb, W)
        return h_proj + r_vec  # 查询向量


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

    def get_query_embedding(self, h, r):
        h_emb = self.project(self.E(h), self.R_proj(r))
        r_vec = self.R(r)
        return h_emb + r_vec


# ==================== 训练 & 加载 ====================
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
            ## 【TODO】这里要确保负样本和原来的不一样。
            neg_t = torch.randint(0, mapper.entity_count, (len(h), NEGATIVE_SAMPLES), device=device)             

            pos_score = model(h, r, t)
            neg_score = model(
                h.unsqueeze(1).expand(-1, NEGATIVE_SAMPLES).reshape(-1),
                r.unsqueeze(1).expand(-1, NEGATIVE_SAMPLES).reshape(-1),
                neg_t.reshape(-1)
            ).reshape(-1, NEGATIVE_SAMPLES)

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


def load_model(model_class, model_path, mapper, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = model_class(checkpoint['entity_count'], checkpoint['relation_count'], EMBEDDING_DIM)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, dataset, mapper, device, k_list=(1, 3, 10)):
    print("🔍 开始在开发集上评估模型性能（仅尾实体预测）...")
    model.eval()
    hits_at = {k: 0.0 for k in k_list}
    mrr = 0.0
    count = 0

    entity_emb = model.E.weight.data.cpu().numpy()
    index = faiss.IndexFlatL2(entity_emb.shape[1])
    index.add(entity_emb)

    with torch.no_grad():
        for h, r, t in tqdm(dataset.triples, desc="Evaluating"):
            try:
                h_id = torch.tensor([mapper.entity_to_id[h]], device=device)
                r_id = torch.tensor([mapper.relation_to_id[r]], device=device)
                t_id = mapper.entity_to_id[t]
            except KeyError:
                continue

            # 只做尾实体预测 (h, r, ?)
            query = model.get_query_embedding(h_id, r_id).detach().cpu().numpy()
            _, indices = index.search(query, 1000)
            pred_ids = indices[0]

            # 过滤
            filtered_tails = [tail for head, rel, tail in mapper.all_train_triples if head == h and rel == r and tail != t]
            filter_ids = [mapper.entity_to_id[tail] for tail in filtered_tails if tail in mapper.entity_to_id]
            for fid in filter_ids:
                if fid in pred_ids:
                    mask = pred_ids == fid
                    pred_ids = np.concatenate([pred_ids[~mask], pred_ids[mask]])

            rank = np.where(pred_ids == t_id)[0]
            final_rank = rank[0] + 1 if len(rank) > 0 else 10000

            for k in k_list:
                if final_rank <= k:
                    hits_at[k] += 1
            mrr += 1.0 / final_rank
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

def evaluate_ensemble(models_with_weights, dev_dataset, mapper, device, k_list=(1, 3, 10), rrf_k=60):
# def evaluate_ensemble_rrf(models_with_weights, dev_dataset, mapper, device, k_list=(1, 3, 10), rrf_k=60):
    """
    使用 Reciprocal Rank Fusion (RRF) 融合多个模型的排序结果
    特别要求：若正确答案不在 Top10 候选中，则 MRR 得分为 0
    """
    print("\n" + "="*50)
    print("🚀 开始评估融合模型 (RRF 融合) - Top10 外答案得分为 0")
    print("="*50)

    hits_at = {k: 0.0 for k in k_list}
    mrr_scores = []  # 每个 query 的 MRR 得分（可能为 0）
    count = 0

    # 获取实体嵌入（用于 FAISS 检索）
    sample_model = next(iter(models_with_weights.values()))[0]
    entity_emb = sample_model.E.weight.data.cpu().numpy()
    index = faiss.IndexFlatL2(entity_emb.shape[1])
    index.add(entity_emb)

    with torch.no_grad():
        for h, r, t in tqdm(dev_dataset.triples, desc="RRF Eval"):
            try:
                h_id = torch.tensor([mapper.entity_to_id[h]], device=device)
                r_id = torch.tensor([mapper.relation_to_id[r]], device=device)
                t_id = mapper.entity_to_id[t]
            except KeyError:
                # 跳过未登录实体
                continue

            # ========== 收集每个模型的排序得分（RRF）==========
            rrf_scores = np.zeros(mapper.entity_count)

            for name, (model, weight) in models_with_weights.items():
                q = model.get_query_embedding(h_id, r_id).detach().cpu().numpy()
                _, indices = index.search(q, 1000)  # 检索 top 1000
                candidate_ids = indices[0]

                # RRF 公式: score += weight / (k + rank)
                for rank, idx in enumerate(candidate_ids):
                    rrf_scores[idx] += weight / (rrf_k + rank + 1)

            # ========== 过滤训练集中已存在的三元组（除当前外）==========
            filtered_tails = [
                tail for head, rel, tail in mapper.all_train_triples
                if head == h and rel == r and tail != t
            ]
            for tail in filtered_tails:
                if tail in mapper.entity_to_id:
                    rrf_scores[mapper.entity_to_id[tail]] = -1e9

            # ========== 排序（降序）==========
            ranked_indices = np.argsort(rrf_scores)[::-1]

            # ========== 获取 Top10 预测结果 ==========
            top10_ids = ranked_indices[:10]
            top10_entities = [mapper.id_to_entity[i] for i in top10_ids]

            # ========== 计算指标 ==========
            # 检查真实 tail 是否在 Top10
            if t_id in top10_ids:
                rank = np.where(ranked_indices == t_id)[0][0] + 1  # 排名从 1 开始
                mrr_score = 1.0 / rank
            else:
                mrr_score = 0.0  # Top10 不包含，得分为 0

            # 更新 Hits
            for k in k_list:
                if t_id in top10_ids[:k]:
                    hits_at[k] += 1

            mrr_scores.append(mrr_score)
            count += 1

    # ========== 计算最终指标 ==========
    for k in hits_at:
        hits_at[k] /= count
    mrr = np.mean(mrr_scores) if mrr_scores else 0.0

    print("✅ RRF 融合评估完成！")
    print(f"📊 RRF HITS@1:  {hits_at[1]:.4f}")
    print(f"📊 RRF HITS@3:  {hits_at[3]:.4f}")
    print(f"📊 RRF HITS@10: {hits_at[10]:.4f}")
    print(f"📊 RRF MRR:     {mrr:.4f}")

    return hits_at, mrr


# ==================== 融合预测 ====================
def predict_ensemble(models_with_weights, test_dataset, mapper, device, max_head_entities=None, rrf_k=60): 
# def predict_ensemble_rrf(models_with_weights, test_dataset, mapper, device, max_head_entities=None, rrf_k=60):
    """
    使用 RRF 融合多个模型的预测结果（与 evaluate_ensemble_rrf 保持一致）
    输出每个 query 的 Top10 预测结果
    """
    print("🔍 开始融合预测 (RRF 融合 + FAISS 加速) ...")
    results = []

    # 获取实体嵌入
    sample_model = next(iter(models_with_weights.values()))[0]
    entity_emb = sample_model.E.weight.data.cpu().numpy()
    index = faiss.IndexFlatL2(entity_emb.shape[1])
    index.add(entity_emb)

    triples = test_dataset.triples
    if max_head_entities:
        triples = triples[:max_head_entities]

    with torch.no_grad():
        for h, r, _ in tqdm(triples, desc="RRF Predict"):
            try:
                h_id = torch.tensor([mapper.entity_to_id[h]], device=device)
                r_id = torch.tensor([mapper.relation_to_id[r]], device=device)
            except KeyError:
                # 若实体未登录，返回 10 个自身（或随机）
                preds = [h] * 10
                results.append('\t'.join([h, r] + preds))
                continue

            # ========== RRF 融合 ==========
            rrf_scores = np.zeros(mapper.entity_count)
            for name, (model, weight) in models_with_weights.items():
                q = model.get_query_embedding(h_id, r_id).detach().cpu().numpy()
                _, indices = index.search(q, 1000)
                candidate_ids = indices[0]
                for rank, idx in enumerate(candidate_ids):
                    rrf_scores[idx] += weight / (rrf_k + rank + 1)

            # ========== 排序并取 Top10 ==========
            ranked_indices = np.argsort(rrf_scores)[::-1]
            top10_ids = ranked_indices[:10]
            preds = [mapper.id_to_entity[i] for i in top10_ids]

            results.append('\t'.join([h, r] + preds))

    # ========== 保存结果 ==========
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results) + '\n')

    print(f"✅ RRF 融合结果已保存至: {OUTPUT_FILE_PATH}")

    # 压缩为 zip
    zip_path = OUTPUT_FILE_PATH.replace(".tsv", "") + f"__{scheme_type}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUTPUT_FILE_PATH, arcname=os.path.basename(OUTPUT_FILE_PATH))
    print(f"✅ 已压缩为: {zip_path}")


# ==================== 主函数 ====================
def main():
    device = torch.device('mps') if torch.backends.mps.is_available() else \
             torch.device('cpu')
    print(f"🚀 使用设备: {device}")

    train_data = KnowledgeGraphDataset(TRAIN_FILE_PATH, max_lines=MAX_LINES, is_train=True)
    dev_data = KnowledgeGraphDataset(DEV_FILE_PATH, is_test=False, is_train=False)
    test_data = KnowledgeGraphDataset(TEST_FILE_PATH, is_test=True, is_train=False)

    mapper = EntityRelationMapper()
    mapper.build_mappings(train_data, dev_data, test_data)
    print(f"实体数: {mapper.entity_count}, 关系数: {mapper.relation_count}")

    model_classes = {
        # 'TransE': TransE, ##【TODO】实际跑的时候要放开的这里。
        'TransH': TransH,
        # 'TransD': TransD,
    }

    # 训练所有模型
    for name, Cls in model_classes.items():
        model = Cls(mapper.entity_count, mapper.relation_count, EMBEDDING_DIM)
        train_model(model, name, train_data, mapper, device)

    # 加载并评估每个模型
    loaded_models_with_weight = {}
    for name, Cls in model_classes.items():
        model = load_model(Cls, TRAINED_MODEL_PATHS[name], mapper, device)
        loaded_models_with_weight[name] = (model, 1.0)
        print(f"\n📈 正在评估模型: {name}")
        ##【TODO】实际跑的时候要放开的这里。
        evaluate_model(model, dev_data, mapper, device) 
    
    # ##【TODO】实际跑的时候要放开的这里。
    # # 🔥 评估融合模型
    # evaluate_ensemble(loaded_models_with_weight, dev_data, mapper, device) 

    # ##【TODO】实际跑的时候要放开的这里。
    # # 执行融合预测
    # predict_ensemble(loaded_models_with_weight, test_data, mapper, device, MAX_HEAD_ENTITIES)
    # print("🎉 所有任务完成！融合预测及评估已完成。")

if __name__ == "__main__":
    main()