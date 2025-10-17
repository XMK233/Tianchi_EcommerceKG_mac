import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
import os
import time
from tqdm import tqdm
import zipfile
from pathlib import Path

# 引入 FAISS
import faiss

# 设置随机种子
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ==================== 路径配置 ====================
BASE_DIR = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac"
TRAIN_FILE_PATH = f"{BASE_DIR}/originalData/OpenBG500/OpenBG500_train.tsv"
TEST_FILE_PATH = f"{BASE_DIR}/originalData/OpenBG500/OpenBG500_test.tsv"
DEV_FILE_PATH = f"{BASE_DIR}/originalData/OpenBG500/OpenBG500_dev.tsv"
OUTPUT_FILE_PATH = f"{BASE_DIR}/preprocessedData/OpenBG500_test__ensemble_conv_rot.tsv"

MODEL_DIR = f"{BASE_DIR}/trained_model"
os.makedirs(MODEL_DIR, exist_ok=True)

TRAINED_MODEL_PATHS = {
    'TransE': f"{MODEL_DIR}/trained_model__transE.pth",
    'TransH': f"{MODEL_DIR}/trained_model__transH.pth",
    'TransD': f"{MODEL_DIR}/trained_model__transD.pth",
    'ConvE': f"{MODEL_DIR}/trained_model__convE.pth",
    'RotatE': f"{MODEL_DIR}/trained_model__rotate.pth"
}

# ==================== 超参数 ====================
EMBEDDING_DIM = 100
COMPLEX_DIM = EMBEDDING_DIM // 2  # RotatE 使用复数 (real, imag)
CONV_EMB_2D = (10, 10)  # ConvE reshape 维度
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 1
BATCH_SIZE = 256
NEGATIVE_SAMPLES = 10
MAX_LINES = None
MAX_HEAD_ENTITIES = None

# ==================== 数据集和映射 ====================
class KnowledgeGraphDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, is_test=False, max_lines=None):
        self.triples = []
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


class EntityRelationMapper:
    def __init__(self):
        self.entity_to_id = {}
        self.id_to_entity = {}
        self.relation_to_id = {}
        self.id_to_relation = {}
        self.entity_count = 0
        self.relation_count = 0

    def build_mappings(self, *datasets):
        entities = set()
        relations = set()
        for dataset in datasets:
            for h, r, t in dataset.triples:
                entities.add(h)
                entities.add(t)
                relations.add(r)
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

    def get_entity_embeddings(self):
        return self.E.weight.data.cpu().numpy()

    def get_query_embedding(self, h, r):
        return (self.E(h) + self.R(r)).data.cpu().numpy()


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

    def get_entity_embeddings(self):
        return self.E.weight.data.cpu().numpy()

    def get_query_embedding(self, h, r):
        h_emb = self.project(self.E(h), self.W(r))
        return (h_emb + self.R(r)).data.cpu().numpy()


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
        return torch.norm(h_emb + self.R(r) - t_emb, p=1, dim=1)

    def get_entity_embeddings(self):
        return self.E.weight.data.cpu().numpy()

    def get_query_embedding(self, h, r):
        h_emb = self.project(self.E(h), self.R_proj(r))
        return (h_emb + self.R(r)).data.cpu().numpy()


# ==================== ConvE ====================
class ConvE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, emb_2d=(10, 10)):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.emb_2d = emb_2d
        assert embedding_dim == emb_2d[0] * emb_2d[1]

        self.E = nn.Embedding(num_entities, embedding_dim)
        self.R = nn.Embedding(num_relations, embedding_dim)
        self.conv = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.fc = nn.Linear(32 * emb_2d[0] * emb_2d[1], embedding_dim)
        self.dropout = nn.Dropout(0.2)

        nn.init.xavier_uniform_(self.E.weight)
        nn.init.xavier_uniform_(self.R.weight)

    def forward(self, h, r, t):
        h_emb = self.E(h).view(-1, 1, *self.emb_2d)  # [B, 1, 10, 10]
        r_emb = self.R(r).view(-1, 1, *self.emb_2d)
        x = torch.cat([h_emb, r_emb], dim=2)  # 拼接通道
        x = self.conv(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        t_emb = self.E(t)
        return torch.norm(x - t_emb, p=2, dim=1)

    def get_entity_embeddings(self):
        return self.E.weight.data.view(-1, self.embedding_dim).cpu().numpy()

    def get_query_embedding(self, h, r):
        h_emb = self.E(h).view(-1, 1, *self.emb_2d)
        r_emb = self.R(r).view(-1, 1, *self.emb_2d)
        x = torch.cat([h_emb, r_emb], dim=2)
        x = self.conv(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.data.cpu().numpy()


# ==================== RotatE ====================
class RotatE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.E = nn.Embedding(num_entities, embedding_dim)
        self.R = nn.Embedding(num_relations, embedding_dim)
        nn.init.uniform_(self.E.weight, a=-6.0 / (embedding_dim ** 0.5), b=6.0 / (embedding_dim ** 0.5))
        nn.init.uniform_(self.R.weight, a=-6.0 / (embedding_dim ** 0.5), b=6.0 / (embedding_dim ** 0.5))
        self.E.weight.data.div_(torch.norm(self.E.weight.data, dim=1, keepdim=True))  # 初始化为单位向量

    def forward(self, h, r, t):
        h_re, h_im = torch.chunk(self.E(h), 2, dim=1)
        t_re, t_im = torch.chunk(self.E(t), 2, dim=1)
        r_phase = self.R(r)
        r_re = torch.cos(r_phase)
        r_im = torch.sin(r_phase)
        # (h_re + i h_im) * (r_re + i r_im) = (h_re*r_re - h_im*r_im) + i(...)
        re_score = h_re * r_re - h_im * r_im
        im_score = h_re * r_im + h_im * r_re
        score = torch.stack([re_score, im_score], dim=0)
        t = torch.stack([t_re, t_im], dim=0)
        return -torch.sum(score * t, dim=0)  # 负的余弦相似度

    def get_entity_embeddings(self):
        e = self.E.weight.data
        e_re, e_im = torch.chunk(e, 2, dim=1)
        # 拼接实部和虚部作为向量
        return torch.cat([e_re, e_im], dim=1).cpu().numpy()

    def get_query_embedding(self, h, r):
        h_re, h_im = torch.chunk(self.E(h), 2, dim=1)
        r_phase = self.R(r)
        r_re = torch.cos(r_phase)
        r_im = torch.sin(r_phase)
        re_score = h_re * r_re - h_im * r_im
        im_score = h_re * r_im + h_im * r_im
        return torch.cat([re_score, im_score], dim=1).cpu().numpy()


# ==================== FAISS 检索器 ====================
class FAISSIndex:
    def __init__(self, embeddings, metric="l2"):
        self.dim = embeddings.shape[1]
        self.metric = metric
        if metric == "l2":
            self.index = faiss.IndexFlatL2(self.dim)
        elif metric == "ip":
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            raise ValueError("metric must be 'l2' or 'ip'")
        self.index.add(embeddings)

    def search(self, queries, k=10):
        if self.metric == "ip":
            # 归一化查询向量
            queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        D, I = self.index.search(queries, k)
        return I  # 返回 top-k 索引


# ==================== 训练函数 ====================
def train_model(model, model_name, train_dataset, mapper, device):
    path = TRAINED_MODEL_PATHS[model_name]
    if os.path.exists(path):
        print(f"[{model_name}] 模型已存在，跳过训练")
        return

    print(f"[{model_name}] 开始训练...")
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    model.to(device)
    model.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0
        for h_list, r_list, t_list in tqdm(loader, desc=f"{model_name} Epoch {epoch+1}"):
            h = torch.tensor([mapper.entity_to_id[h] for h in h_list], device=device)
            r = torch.tensor([mapper.relation_to_id[r] for r in r_list], device=device)
            t = torch.tensor([mapper.entity_to_id[t] for t in t_list], device=device))

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
        scheduler.step()
        print(f"[{model_name}] Epoch {epoch+1} Loss: {epoch_loss / len(loader):.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'entity_count': mapper.entity_count,
        'relation_count': mapper.relation_count,
        'embedding_dim': EMBEDDING_DIM,
        'entity_to_id': mapper.entity_to_id,
        'relation_to_id': mapper.relation_to_id,
    }, path)
    print(f"[{model_name}] 模型已保存")


# ==================== 加载模型 ====================
def load_model(ModelClass, path, mapper, device):
    ckpt = torch.load(path, map_location=device)
    model = ModelClass(ckpt['entity_count'], ckpt['relation_count'], EMBEDDING_DIM)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


# ==================== 融合预测（使用 FAISS）====================
def predict_ensemble_with_faiss(models, test_dataset, mapper, device, max_head_entities=None):
    print("🚀 开始使用 FAISS 加速融合预测...")

    # 构建 FAISS 索引（所有模型共享同一实体嵌入空间）
    entity_embeddings = models['TransE'].get_entity_embeddings()  # 取一个模型即可
    faiss_index = FAISSIndex(entity_embeddings, metric="l2")  # 使用 L2 距离

    results = []
    triples = test_dataset.triples
    if max_head_entities:
        triples = triples[:max_head_entities]

    with torch.no_grad():
        for h, r, _ in tqdm(triples, desc="Ensemble FAISS Search"):
            h_id = torch.tensor([mapper.entity_to_id[h]], device=device)
            r_id = torch.tensor([mapper.relation_to_id[r]], device=device)

            # 收集每个模型的查询向量
            query_vecs = []
            for name, model in models.items():
                q = model.get_query_embedding(h_id, r_id)  # [1, dim]
                query_vecs.append(q)

            # 平均查询向量
            fused_query = np.mean(np.stack(query_vecs), axis=0)  # [1, dim]

            # FAISS 检索 Top-10
            topk_indices = faiss_index.search(fused_query, k=10)[0][0]  # [10]

            preds = [mapper.id_to_entity[i] for i in topk_indices]
            results.append('\t'.join([h, r] + preds))

    # 保存结果
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results) + '\n')
    print(f"✅ 融合结果已保存至: {OUTPUT_FILE_PATH}")

    zip_path = OUTPUT_FILE_PATH.replace(".tsv", "__faiss.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUTPUT_FILE_PATH, arcname=os.path.basename(OUTPUT_FILE_PATH))
    print(f"✅ 已压缩为: {zip_path}")


# ==================== 主函数 ====================
def main():
    device = torch.device('mps') if torch.backends.mps.is_available() else \
             torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"🚀 使用设备: {device}")

    # 加载数据
    train_data = KnowledgeGraphDataset(TRAIN_FILE_PATH, max_lines=MAX_LINES)
    test_data = KnowledgeGraphDataset(TEST_FILE_PATH, is_test=True)
    dev_data = KnowledgeGraphDataset(DEV_FILE_PATH)

    mapper = EntityRelationMapper()
    mapper.build_mappings(train_data, test_data, dev_data)
    print(f"实体数: {mapper.entity_count}, 关系数: {mapper.relation_count}")

    # 定义模型
    model_classes = {
        'TransE': TransE,
        'TransH': TransH,
        'TransD': TransD,
        'ConvE': ConvE,
        'RotatE': RotatE
    }

    # 训练所有模型
    for name, Cls in model_classes.items():
        model = Cls(mapper.entity_count, mapper.relation_count, EMBEDDING_DIM)
        train_model(model, name, train_data, mapper, device)

    # 加载所有模型
    loaded_models = {}
    for name, Cls in model_classes.items():
        loaded_models[name] = load_model(Cls, TRAINED_MODEL_PATHS[name], mapper, device)
        print(f"[{name}] 模型已加载")

    # 融合预测（FAISS 加速）
    predict_ensemble_with_faiss(loaded_models, test_data, mapper, device, MAX_HEAD_ENTITIES)

    print("🎉 所有任务完成！融合 + FAISS 预测已生成。")


if __name__ == "__main__":
    main()