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

scheme_type = "s7_ep10_bs128"
# 数据路径（请根据你的实际路径修改）
BASE_DIR = "/mnt/d/forCoding_data/Tianchi_EcommerceKG"
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
EMBEDDING_DIM = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 10  # 【TODO】可增加
BATCH_SIZE = 256
NEGATIVE_SAMPLES = 10
MAX_LINES = None
MAX_HEAD_ENTITIES = None
LR_DECAY_STEP = 5
LR_DECAY_FACTOR = 0.1
FORCE_RETRAIN = False  # 【TODO】根据情况设置

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

# ==================== TransE （已优化）====================
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
    def normalize_entities(self):
        """归一化实体嵌入"""
        with torch.no_grad():
            self.E.weight.data.div_(torch.norm(self.E.weight.data, dim=1, keepdim=True) + 1e-9)

# ==================== TransH （已修复）====================
class TransH(nn.Module):
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.E = nn.Embedding(num_entities, dim)
        self.R = nn.Embedding(num_relations, dim)      # 关系向量 d_r
        self.W = nn.Embedding(num_relations, dim)     # 法向量 W (用于超平面)
        nn.init.xavier_uniform_(self.E.weight)
        nn.init.xavier_uniform_(self.R.weight)
        nn.init.xavier_uniform_(self.W.weight)
    
    def project(self, emb, w):  # 将实体投影到关系超平面上
        norm_w = torch.nn.functional.normalize(w, p=2, dim=1)
        scale = torch.sum(emb * norm_w, dim=1, keepdim=True)
        return emb - scale * norm_w

    def forward(self, h, r, t):
        h_emb = self.E(h)
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
        return h_proj + r_vec
    
    def normalize_entities(self):
        with torch.no_grad():
            self.E.weight.data.div_(torch.norm(self.E.weight.data, dim=1, keepdim=True) + 1e-9)

# ==================== TransD （已修复 + 优化）====================
class TransD(nn.Module):
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.dim = dim
        self.E = nn.Embedding(num_entities, dim)
        self.R = nn.Embedding(num_relations, dim)
        self.E_proj = nn.Embedding(num_entities, dim)  # 实体投影向量
        self.R_proj = nn.Embedding(num_relations, dim)  # 关系投影向量
        # 初始化
        nn.init.xavier_uniform_(self.E.weight)
        nn.init.xavier_uniform_(self.R.weight)
        nn.init.xavier_uniform_(self.E_proj.weight)
        nn.init.xavier_uniform_(self.R_proj.weight)
    
    def project(self, e, e_proj, r_proj):
        return e + torch.sum(e * e_proj, dim=1, keepdim=True) * r_proj

    def forward(self, h, r, t):
        h_emb = self.project(self.E(h), self.E_proj(h), self.R_proj(r))
        t_emb = self.project(self.E(t), self.E_proj(t), self.R_proj(r))
        r_vec = self.R(r)
        return torch.norm(h_emb + r_vec - t_emb, p=1, dim=1)
    
    def get_query_embedding(self, h, r):
        h_emb = self.project(self.E(h), self.E_proj(h), self.R_proj(r))
        r_vec = self.R(r)
        return h_emb + r_vec
    
    def normalize_entities(self):
        with torch.no_grad():
            self.E.weight.data.div_(torch.norm(self.E.weight.data, dim=1, keepdim=True) + 1e-9)

# ==================== 训练函数（支持多级初始化）====================
def train_model(model, model_name, train_dataset, mapper, device, pretrained_E=None, pretrained_R=None):
    if os.path.exists(TRAINED_MODEL_PATHS[model_name]) and not FORCE_RETRAIN:
        print(f"[{model_name}] 已存在训练好的模型，跳过训练")
        return
    print(f"[{model_name}] 开始训练...")

    # ========== 初始化：从上游模型加载 E 和 R ==========
    if pretrained_E is not None and hasattr(model, 'E'):
        print(f"✅ 使用上游模型的实体嵌入初始化 {model_name}.E")
        with torch.no_grad():
            model.E.weight.data.copy_(pretrained_E)
    
    if pretrained_R is not None and hasattr(model, 'R'):
        print(f"✅ 使用上游模型的关系嵌入初始化 {model_name}.R")
        with torch.no_grad():
            model.R.weight.data.copy_(pretrained_R)

    # ========== 正常训练流程 ==========
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

            # 负样本生成
            neg_t = torch.randint(0, mapper.entity_count, (len(h), NEGATIVE_SAMPLES), device=device)
            pos_t_expanded = t.unsqueeze(1).expand(-1, NEGATIVE_SAMPLES)
            mask = (neg_t == pos_t_expanded)
            while mask.any():
                neg_t[mask] = torch.randint(0, mapper.entity_count, (mask.sum(),), device=device)
                mask = (neg_t == pos_t_expanded)

            # 前向
            pos_score = model(h, r, t)
            neg_score = model(
                h.unsqueeze(1).expand(-1, NEGATIVE_SAMPLES).reshape(-1),
                r.unsqueeze(1).expand(-1, NEGATIVE_SAMPLES).reshape(-1),
                neg_t.reshape(-1)
            ).reshape(-1, NEGATIVE_SAMPLES)

            loss = torch.mean(torch.relu(pos_score.unsqueeze(1) - neg_score + 1.0))

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 归一化
            if hasattr(model, 'normalize_entities'):
                model.normalize_entities()

            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        scheduler.step()
        print(f"[{model_name}] Epoch {epoch+1} Loss: {epoch_loss / len(loader):.4f}")

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'entity_count': mapper.entity_count,
        'relation_count': mapper.relation_count,
        'embedding_dim': EMBEDDING_DIM,
        'entity_to_id': mapper.entity_to_id,
        'relation_to_id': mapper.relation_to_id,
    }, TRAINED_MODEL_PATHS[model_name])
    print(f"[{model_name}] 模型已保存")

# ==================== 加载与评估 ====================
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
            query = model.get_query_embedding(h_id, r_id).detach().cpu().numpy()
            _, indices = index.search(query, 1000)
            pred_ids = indices[0]
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
    print("\n" + "="*50)
    print("🚀 开始评估融合模型 (RRF 融合) - Top10 外答案得分为 0")
    print("="*50)
    hits_at = {k: 0.0 for k in k_list}
    mrr_scores = []
    count = 0
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
                continue
            rrf_scores = np.zeros(mapper.entity_count)
            for name, (model, weight) in models_with_weights.items():
                q = model.get_query_embedding(h_id, r_id).detach().cpu().numpy()
                _, indices = index.search(q, 1000)
                candidate_ids = indices[0]
                for rank, idx in enumerate(candidate_ids):
                    rrf_scores[idx] += weight / (rrf_k + rank + 1)
            filtered_tails = [
                tail for head, rel, tail in mapper.all_train_triples
                if head == h and rel == r and tail != t
            ]
            for tail in filtered_tails:
                if tail in mapper.entity_to_id:
                    rrf_scores[mapper.entity_to_id[tail]] = -1e9
            ranked_indices = np.argsort(rrf_scores)[::-1]
            top10_ids = ranked_indices[:10]
            if t_id in top10_ids:
                rank = np.where(ranked_indices == t_id)[0][0] + 1
                mrr_score = 1.0 / rank
            else:
                mrr_score = 0.0
            for k in k_list:
                if t_id in top10_ids[:k]:
                    hits_at[k] += 1
            mrr_scores.append(mrr_score)
            count += 1
    for k in hits_at:
        hits_at[k] /= count
    mrr = np.mean(mrr_scores) if mrr_scores else 0.0
    print("✅ RRF 融合评估完成！")
    print(f"📊 RRF HITS@1:  {hits_at[1]:.4f}")
    print(f"📊 RRF HITS@3:  {hits_at[3]:.4f}")
    print(f"📊 RRF HITS@10: {hits_at[10]:.4f}")
    print(f"📊 RRF MRR:     {mrr:.4f}")
    return hits_at, mrr

def predict_ensemble(models_with_weights, test_dataset, mapper, device, max_head_entities=None, rrf_k=60):
    print("🔍 开始融合预测 (RRF 融合 + FAISS 加速) ...")
    results = []
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
                preds = [h] * 10
                results.append('\t'.join([h, r] + preds))
                continue
            rrf_scores = np.zeros(mapper.entity_count)
            for name, (model, weight) in models_with_weights.items():
                q = model.get_query_embedding(h_id, r_id).detach().cpu().numpy()
                _, indices = index.search(q, 1000)
                candidate_ids = indices[0]
                for rank, idx in enumerate(candidate_ids):
                    rrf_scores[idx] += weight / (rrf_k + rank + 1)
            ranked_indices = np.argsort(rrf_scores)[::-1]
            top10_ids = ranked_indices[:10]
            preds = [mapper.id_to_entity[i] for i in top10_ids]
            results.append('\t'.join([h, r] + preds))
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results) + '\n')
    print(f"✅ RRF 融合结果已保存至: {OUTPUT_FILE_PATH}")
    zip_path = OUTPUT_FILE_PATH.replace(".tsv", "") + f"__{scheme_type}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUTPUT_FILE_PATH, arcname=os.path.basename(OUTPUT_FILE_PATH))
    print(f"✅ 已压缩为: {zip_path}")

# ==================== 主函数 ====================
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else \
             torch.device('mps') if torch.backends.mps.is_available() else \
             torch.device('cpu')
    print(f"🚀 使用设备: {device}")

    train_data = KnowledgeGraphDataset(TRAIN_FILE_PATH, max_lines=MAX_LINES, is_train=True)
    dev_data = KnowledgeGraphDataset(DEV_FILE_PATH, is_test=False, is_train=False)
    test_data = KnowledgeGraphDataset(TEST_FILE_PATH, is_test=True, is_train=False)

    mapper = EntityRelationMapper()
    mapper.build_mappings(train_data, dev_data, test_data)
    print(f"实体数: {mapper.entity_count}, 关系数: {mapper.relation_count}")

    # ==================== 第一步：训练 TransE ====================
    print("\n" + "="*50)
    print("🚀 第一步：训练 TransE")
    print("="*50)
    transE_model = TransE(mapper.entity_count, mapper.relation_count, EMBEDDING_DIM)
    train_model(transE_model, 'TransE', train_data, mapper, device)

    # 提取 E 和 R 用于 TransH
    transE_checkpoint = torch.load(TRAINED_MODEL_PATHS['TransE'], map_location=device)
    transE_E_weight = transE_checkpoint['model_state_dict']['E.weight']
    transE_R_weight = transE_checkpoint['model_state_dict']['R.weight']
    print(f"✅ TransE 训练完成，提取 E({transE_E_weight.shape}) 和 R({transE_R_weight.shape}) 用于初始化 TransH")

    # ==================== 第二步：训练 TransH（复用 TransE 的 E 和 R）====================
    print("\n" + "="*50)
    print("🚀 第二步：训练 TransH（使用 TransE 的 E 和 R 初始化）")
    print("="*50)
    transH_model = TransH(mapper.entity_count, mapper.relation_count, EMBEDDING_DIM)
    train_model(
        model=transH_model,
        model_name='TransH',
        train_dataset=train_data,
        mapper=mapper,
        device=device,
        pretrained_E=transE_E_weight,
        pretrained_R=transE_R_weight
    )

    # 提取 E 和 R 用于 TransD（来自 TransH，也可以用回 TransE，这里用最新）
    transH_checkpoint = torch.load(TRAINED_MODEL_PATHS['TransH'], map_location=device)
    transH_E_weight = transH_checkpoint['model_state_dict']['E.weight']
    transH_R_weight = transH_checkpoint['model_state_dict']['R.weight']
    print(f"✅ TransH 训练完成，提取 E({transH_E_weight.shape}) 和 R({transH_R_weight.shape}) 用于初始化 TransD")

    # ==================== 第三步：训练 TransD（复用 TransH 的 E 和 R）====================
    print("\n" + "="*50)
    print("🚀 第三步：训练 TransD（使用 TransH 的 E 和 R 初始化）")
    print("="*50)
    transD_model = TransD(mapper.entity_count, mapper.relation_count, EMBEDDING_DIM)
    train_model(
        model=transD_model,
        model_name='TransD',
        train_dataset=train_data,
        mapper=mapper,
        device=device,
        pretrained_E=transH_E_weight,
        pretrained_R=transH_R_weight
    )

    # ==================== 评估 ====================
    print("\n" + "="*50)
    print("📈 开始评估")
    print("="*50)

    transE_model_eval = load_model(TransE, TRAINED_MODEL_PATHS['TransE'], mapper, device)
    transH_model_eval = load_model(TransH, TRAINED_MODEL_PATHS['TransH'], mapper, device)
    transD_model_eval = load_model(TransD, TRAINED_MODEL_PATHS['TransD'], mapper, device)

    # print(f"\n📊 评估 TransE")
    # evaluate_model(transE_model_eval, dev_data, mapper, device)

    # print(f"\n📊 评估 TransH (warm-start)")
    # evaluate_model(transH_model_eval, dev_data, mapper, device)

    # print(f"\n📊 评估 TransD (warm-start)")
    # evaluate_model(transD_model_eval, dev_data, mapper, device)

    # ==================== 融合预测 ====================
    loaded_models_with_weight = {
        'TransE': (transE_model_eval, 1.0),
        'TransH': (transH_model_eval, 1.2),
        'TransD': (transD_model_eval, 1.5),
    }

    evaluate_ensemble(loaded_models_with_weight, dev_data, mapper, device)
    predict_ensemble(loaded_models_with_weight, test_data, mapper, device, MAX_HEAD_ENTITIES)

    print("🎉 三阶段串行训练完成！TransE → TransH → TransD 迁移成功。")

if __name__ == "__main__":
    main()

## 输出：
# /home/xiuminke/miniconda3/envs/ml12/lib/python3.11/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
#   import pynvml  # type: ignore[import]
# 🚀 使用设备: cuda
# 加载数据: /mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_train.tsv
# 共加载 1242550 个三元组
# 加载数据: /mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_dev.tsv
# 共加载 5000 个三元组
# 加载数据: /mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_test.tsv
# 共加载 5000 个三元组
# 实体数: 249747, 关系数: 500

# ==================================================
# 🚀 第一步：训练 TransE
# ==================================================
# [TransE] 开始训练...
# TransE Epoch 1: 100%|██████████████████████████████████████████████████| 4854/4854 [00:39<00:00, 122.74it/s, loss=0.125]
# [TransE] Epoch 1 Loss: 0.2690
# TransE Epoch 2: 100%|█████████████████████████████████████████████████| 4854/4854 [00:38<00:00, 126.49it/s, loss=0.0561]
# [TransE] Epoch 2 Loss: 0.0548
# TransE Epoch 3: 100%|█████████████████████████████████████████████████| 4854/4854 [00:39<00:00, 121.49it/s, loss=0.0289]
# [TransE] Epoch 3 Loss: 0.0235
# TransE Epoch 4: 100%|█████████████████████████████████████████████████| 4854/4854 [00:39<00:00, 124.16it/s, loss=0.0166]
# [TransE] Epoch 4 Loss: 0.0137
# TransE Epoch 5: 100%|█████████████████████████████████████████████████| 4854/4854 [00:39<00:00, 123.64it/s, loss=0.0116]
# [TransE] Epoch 5 Loss: 0.0090
# TransE Epoch 6: 100%|████████████████████████████████████████████████| 4854/4854 [00:39<00:00, 122.46it/s, loss=0.00416]
# [TransE] Epoch 6 Loss: 0.0054
# TransE Epoch 7: 100%|████████████████████████████████████████████████| 4854/4854 [00:40<00:00, 120.56it/s, loss=0.00978]
# [TransE] Epoch 7 Loss: 0.0042
# TransE Epoch 8: 100%|████████████████████████████████████████████████| 4854/4854 [00:39<00:00, 122.82it/s, loss=0.00382]
# [TransE] Epoch 8 Loss: 0.0035
# TransE Epoch 9: 100%|█████████████████████████████████████████████████| 4854/4854 [00:40<00:00, 121.17it/s, loss=0.0018]
# [TransE] Epoch 9 Loss: 0.0031
# TransE Epoch 10: 100%|███████████████████████████████████████████████| 4854/4854 [00:38<00:00, 126.02it/s, loss=0.00145]
# [TransE] Epoch 10 Loss: 0.0027
# [TransE] 模型已保存
# ✅ TransE 训练完成，提取 E(torch.Size([249747, 100])) 和 R(torch.Size([500, 100])) 用于初始化 TransH

# ==================================================
# 🚀 第二步：训练 TransH（使用 TransE 的 E 和 R 初始化）
# ==================================================
# [TransH] 开始训练...
# ✅ 使用上游模型的实体嵌入初始化 TransH.E
# ✅ 使用上游模型的关系嵌入初始化 TransH.R
# TransH Epoch 1: 100%|████████████████████████████████████████████████| 4854/4854 [00:45<00:00, 105.87it/s, loss=0.00436]
# [TransH] Epoch 1 Loss: 0.0039
# TransH Epoch 2: 100%|█████████████████████████████████████████████████| 4854/4854 [00:46<00:00, 104.59it/s, loss=0.0033]
# [TransH] Epoch 2 Loss: 0.0047
# TransH Epoch 3: 100%|████████████████████████████████████████████████| 4854/4854 [00:44<00:00, 108.24it/s, loss=0.00298]
# [TransH] Epoch 3 Loss: 0.0041
# TransH Epoch 4: 100%|████████████████████████████████████████████████| 4854/4854 [00:46<00:00, 104.23it/s, loss=0.00239]
# [TransH] Epoch 4 Loss: 0.0035
# TransH Epoch 5: 100%|█████████████████████████████████████████████████| 4854/4854 [00:46<00:00, 104.05it/s, loss=0.0033]
# [TransH] Epoch 5 Loss: 0.0030
# TransH Epoch 6: 100%|████████████████████████████████████████████████| 4854/4854 [00:46<00:00, 104.94it/s, loss=0.00391]
# [TransH] Epoch 6 Loss: 0.0023
# TransH Epoch 7: 100%|███████████████████████████████████████████████| 4854/4854 [00:46<00:00, 105.08it/s, loss=0.000223]
# [TransH] Epoch 7 Loss: 0.0018
# TransH Epoch 8: 100%|████████████████████████████████████████████████| 4854/4854 [00:45<00:00, 105.88it/s, loss=0.00212]
# [TransH] Epoch 8 Loss: 0.0016
# TransH Epoch 9: 100%|████████████████████████████████████████████████| 4854/4854 [00:45<00:00, 107.59it/s, loss=0.00333]
# [TransH] Epoch 9 Loss: 0.0014
# TransH Epoch 10: 100%|███████████████████████████████████████████████| 4854/4854 [00:45<00:00, 106.20it/s, loss=0.00129]
# [TransH] Epoch 10 Loss: 0.0013
# [TransH] 模型已保存
# ✅ TransH 训练完成，提取 E(torch.Size([249747, 100])) 和 R(torch.Size([500, 100])) 用于初始化 TransD

# ==================================================
# 🚀 第三步：训练 TransD（使用 TransH 的 E 和 R 初始化）
# ==================================================
# [TransD] 开始训练...
# ✅ 使用上游模型的实体嵌入初始化 TransD.E
# ✅ 使用上游模型的关系嵌入初始化 TransD.R
# TransD Epoch 1: 100%|█████████████████████████████████████████████████| 4854/4854 [01:10<00:00, 69.15it/s, loss=0.00182]
# [TransD] Epoch 1 Loss: 0.0018
# TransD Epoch 2: 100%|█████████████████████████████████████████████████| 4854/4854 [01:10<00:00, 68.65it/s, loss=0.00256]
# [TransD] Epoch 2 Loss: 0.0021
# TransD Epoch 3: 100%|█████████████████████████████████████████████████| 4854/4854 [01:10<00:00, 68.55it/s, loss=0.00411]
# [TransD] Epoch 3 Loss: 0.0020
# TransD Epoch 4: 100%|██████████████████████████████████████████████████| 4854/4854 [01:09<00:00, 69.59it/s, loss=0.0018]
# [TransD] Epoch 4 Loss: 0.0019
# TransD Epoch 5: 100%|█████████████████████████████████████████████████| 4854/4854 [01:10<00:00, 68.74it/s, loss=0.00475]
# [TransD] Epoch 5 Loss: 0.0018
# TransD Epoch 6: 100%|█████████████████████████████████████████████████| 4854/4854 [01:08<00:00, 70.60it/s, loss=0.00228]
# [TransD] Epoch 6 Loss: 0.0014
# TransD Epoch 7: 100%|████████████████████████████████████████████████| 4854/4854 [01:11<00:00, 67.70it/s, loss=0.000635]
# [TransD] Epoch 7 Loss: 0.0012
# TransD Epoch 8: 100%|███████████████████████████████████████████████████████| 4854/4854 [01:11<00:00, 68.29it/s, loss=0]
# [TransD] Epoch 8 Loss: 0.0010
# TransD Epoch 9: 100%|█████████████████████████████████████████████████| 4854/4854 [01:10<00:00, 69.08it/s, loss=0.00328]
# [TransD] Epoch 9 Loss: 0.0009
# TransD Epoch 10: 100%|█████████████████████████████████████████████████| 4854/4854 [01:09<00:00, 69.92it/s, loss=0.0004]
# [TransD] Epoch 10 Loss: 0.0009
# [TransD] 模型已保存

# ==================================================
# 📈 开始评估
# ==================================================

# 📊 评估 TransE
# 🔍 开始在开发集上评估模型性能（仅尾实体预测）...
# Evaluating: 100%|███████████████████████████████████████████████████████████████████| 5000/5000 [02:15<00:00, 36.81it/s]
# ✅ 评估完成！
# 📊 HITS@1:  0.2854
# 📊 HITS@3:  0.5398
# 📊 HITS@10: 0.7510
# 📊 MRR:     0.4429

# 📊 评估 TransH (warm-start)
# 🔍 开始在开发集上评估模型性能（仅尾实体预测）...
# Evaluating: 100%|███████████████████████████████████████████████████████████████████| 5000/5000 [02:16<00:00, 36.55it/s]
# ✅ 评估完成！
# 📊 HITS@1:  0.3060
# 📊 HITS@3:  0.5498
# 📊 HITS@10: 0.7530
# 📊 MRR:     0.4588

# 📊 评估 TransD (warm-start)
# 🔍 开始在开发集上评估模型性能（仅尾实体预测）...
# Evaluating: 100%|███████████████████████████████████████████████████████████████████| 5000/5000 [02:15<00:00, 36.96it/s]
# ✅ 评估完成！
# 📊 HITS@1:  0.2998
# 📊 HITS@3:  0.5460
# 📊 HITS@10: 0.7486
# 📊 MRR:     0.4533

# ==================================================
# 🚀 开始评估融合模型 (RRF 融合) - Top10 外答案得分为 0
# ==================================================
# RRF Eval: 100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [03:28<00:00, 24.00it/s]
# ✅ RRF 融合评估完成！
# 📊 RRF HITS@1:  0.3330
# 📊 RRF HITS@3:  0.5624
# 📊 RRF HITS@10: 0.7626
# 📊 RRF MRR:     0.4695
# 🔍 开始融合预测 (RRF 融合 + FAISS 加速) ...
# RRF Predict: 100%|██████████████████████████████████████████████████████████████████| 5000/5000 [01:40<00:00, 49.76it/s]
# ✅ RRF 融合结果已保存至: /mnt/d/forCoding_data/Tianchi_EcommerceKG/preprocessedData/OpenBG500_test.tsv
# ✅ 已压缩为: /mnt/d/forCoding_data/Tianchi_EcommerceKG/preprocessedData/OpenBG500_test__s7_ep10.zip
# 🎉 三阶段串行训练完成！TransE → TransH → TransD 迁移成功。