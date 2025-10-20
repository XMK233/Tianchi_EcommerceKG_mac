# -*- coding: utf-8 -*-
"""
Knowledge Graph Embedding with Multiple GNNs
✅ 所有模型（GraphSAGE, GCN, GAT, RGCN）均支持：
   - 显式 rel_emb
   - TransE 打分
   - in-batch negative sampling
   - 逆关系建图
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.nn import SAGEConv, GCNConv, RGCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import random
import numpy as np
import os
from tqdm import tqdm
import zipfile
import faiss

# ==================== 固定随机种子 ====================
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ==================== 配置 ====================
scheme_type = "gSage_GCN_RGCN_GAT__relational"
BASE_DIR = "/mnt/d/forCoding_data/Tianchi_EcommerceKG"
TRAIN_FILE_PATH = f"{BASE_DIR}/originalData/OpenBG500/OpenBG500_train.tsv"
DEV_FILE_PATH = f"{BASE_DIR}/originalData/OpenBG500/OpenBG500_dev.tsv"
TEST_FILE_PATH = f"{BASE_DIR}/originalData/OpenBG500/OpenBG500_test.tsv"
OUTPUT_FILE_PATH = f"{BASE_DIR}/preprocessedData/OpenBG500_test.tsv"
MODEL_DIR = f"{BASE_DIR}/trained_models/{scheme_type}"
os.makedirs(MODEL_DIR, exist_ok=True)

TRAINED_MODEL_PATHS = {
    'GraphSAGE': f"{MODEL_DIR}/graphsage.pth",
    'GCN': f"{MODEL_DIR}/gcn.pth",
    'RGCN': f"{MODEL_DIR}/rgcn.pth",
    'GAT': f"{MODEL_DIR}/gat.pth",
    'RGAT': f"{MODEL_DIR}/rgat.pth",          # 新增
    'GNN-Film': f"{MODEL_DIR}/gnn_film.pth",  # 新增
}

# 超参数
EMBEDDING_DIM = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 1
BATCH_SIZE = 1024
MAX_LINES = None
LR_DECAY_STEP = 10
LR_DECAY_FACTOR = 0.5
FORCE_RETRAIN = False

# ==================== 数据集 & 映射器 ====================
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

# ==================== 所有 GNN 模型统一接口：返回 (entity_emb, rel_emb) ====================

class GraphSAGE(nn.Module):
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.ent_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations * 2, dim)  # 显式关系嵌入
        self.conv1 = SAGEConv(dim, dim, aggr='mean')
        self.conv2 = SAGEConv(dim, dim, aggr='mean')
        self.dropout = nn.Dropout(0.3)
        # 初始化
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, x, edge_index):
        x = self.ent_emb(x)
        x = self.dropout(torch.relu(self.conv1(x, edge_index)))
        x = x + self.dropout(self.conv2(x, edge_index))  # residual
        return x

    def get_entity_embeddings(self, device):
        x = torch.arange(self.ent_emb.num_embeddings, device=device, dtype=torch.long)
        return self.forward(x, self.edge_index)

    def get_relation_embeddings(self):
        return self.rel_emb.weight.detach().cpu()

class GCN(nn.Module):
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.ent_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations * 2, dim)
        self.conv1 = GCNConv(dim, dim)
        self.conv2 = GCNConv(dim, dim)
        self.dropout = nn.Dropout(0.3)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, x, edge_index):
        x = self.ent_emb(x)
        x = self.dropout(torch.relu(self.conv1(x, edge_index)))
        x = x + self.dropout(self.conv2(x, edge_index))
        return x

    def get_entity_embeddings(self, device):
        x = torch.arange(self.ent_emb.num_embeddings, device=device, dtype=torch.long)
        return self.forward(x, self.edge_index)

    def get_relation_embeddings(self):
        return self.rel_emb.weight.detach().cpu()

class GAT(nn.Module):
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.ent_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations * 2, dim)
        heads = 4
        out_dim = dim // heads
        self.conv1 = GATConv(dim, out_dim, heads=heads, concat=True, dropout=0.3)
        self.conv2 = GATConv(dim, dim, heads=1, concat=False, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, x, edge_index):
        x = self.ent_emb(x)
        x = self.dropout(torch.relu(self.conv1(x, edge_index)))
        x = x + self.dropout(self.conv2(x, edge_index))
        return x

    def get_entity_embeddings(self, device):
        x = torch.arange(self.ent_emb.num_embeddings, device=device, dtype=torch.long)
        return self.forward(x, self.edge_index)

    def get_relation_embeddings(self):
        return self.rel_emb.weight.detach().cpu()

class RGCN(nn.Module):
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.ent_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations * 2, dim)
        self.conv1 = RGCNConv(dim, dim, num_relations * 2, num_bases=8)
        self.conv2 = RGCNConv(dim, dim, num_relations * 2, num_bases=8)
        self.dropout = nn.Dropout(0.3)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, x, edge_index, edge_type):
        x = self.ent_emb(x)
        x = self.dropout(torch.relu(self.conv1(x, edge_index, edge_type)))
        x = x + self.dropout(self.conv2(x, edge_index, edge_type))
        return x

    def get_entity_embeddings(self, device):
        x = torch.arange(self.ent_emb.num_embeddings, device=device, dtype=torch.long)
        return self.forward(x, self.edge_index, self.edge_type.long())

    def get_relation_embeddings(self):
        return self.rel_emb.weight.detach().cpu()

# ==================== RGAT ====================
class RGAT(nn.Module):
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.ent_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations * 2, dim)
        heads = 4
        out_dim = dim // heads
        self.conv1 = GATConv(dim, out_dim, heads=heads, concat=True, dropout=0.3, add_self_loops=False)
        self.conv2 = GATConv(dim, dim, heads=1, concat=False, dropout=0.3, add_self_loops=False)
        self.dropout = nn.Dropout(0.3)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, x, edge_index, edge_type):
        x = self.ent_emb(x)
        x = self.dropout(torch.relu(self.conv1(x, edge_index, edge_type)))
        x = x + self.dropout(self.conv2(x, edge_index, edge_type))
        return x

    def get_entity_embeddings(self, device):
        x = torch.arange(self.ent_emb.num_embeddings, device=device, dtype=torch.long)
        return self.forward(x, self.edge_index, self.edge_type.long())

    def get_relation_embeddings(self):
        return self.rel_emb.weight.detach().cpu()


# ==================== GNN-FiLM (修复版) ====================
class FiLMConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels, out_channels)
        self.modulation = nn.Linear(in_channels, out_channels * 2)  # 生成 scale 和 shift

    def forward(self, x, edge_index, edge_type, rel_weight):
        """
        Args:
            x: 节点特征 [N, dim]
            edge_index: 图边 [2, E]
            edge_type: 边类型 [E]
            rel_weight: 关系嵌入权重 [num_rel*2, dim]
        """
        row, col = edge_index  # row = target, col = source
        x_j = x[col]  # [E, dim], 邻居节点特征
        r_emb = rel_weight[edge_type]  # [E, dim], 对应的关系嵌入
        scale, shift = self.modulation(r_emb).chunk(2, dim=-1)  # [E, dim] each
        x_j = scale * x_j + shift  # 特征调制
        # 消息聚合（类似 GCN）
        out = torch.zeros_like(x)
        out.scatter_add_(0, row.unsqueeze(-1).expand_as(x_j), x_j)
        out = self.lin(out)
        return out


class GNNFiLM(nn.Module):
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.ent_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations * 2, dim)  # 显式关系嵌入
        self.conv1 = FiLMConv(dim, dim)
        self.conv2 = FiLMConv(dim, dim)
        self.dropout = nn.Dropout(0.3)

        # 初始化
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, x, edge_index, edge_type):
        x = self.ent_emb(x)
        rel_weight = self.rel_emb.weight  # ✅ 取出权重张量
        x = self.dropout(torch.relu(self.conv1(x, edge_index, edge_type, rel_weight)))
        x = x + self.dropout(self.conv2(x, edge_index, edge_type, rel_weight))
        return x

    def get_entity_embeddings(self, device):
        x = torch.arange(self.ent_emb.num_embeddings, device=device, dtype=torch.long)
        return self.forward(x, self.edge_index, self.edge_type.long())

    def get_relation_embeddings(self):
        return self.rel_emb.weight.detach().cpu()  # 返回关系嵌入权重

# def train_gnn_model(ModelClass, model_name, train_triples, mapper, device):
#     """
#     显存优化版 GNN 训练函数，支持 RGCN、RGAT、GNN-FiLM 等关系型 GNN 模型
#     使用 NeighborLoader 进行子图采样，避免全图加载导致的显存爆炸
#     完整实现负采样和损失计算
#     【TODO】这个方法试图解决rgcn爆显存的问题，但存在一些问题。慢慢调试吧。
#     """
#     path = TRAINED_MODEL_PATHS[model_name]
#     if os.path.exists(path) and not FORCE_RETRAIN:
#         print(f"[{model_name}] 模型已存在，跳过训练")
#         ckpt = torch.load(path, map_location=device)
#         return ckpt['entity_emb'], ckpt['rel_emb']
    
#     print(f"[{model_name}] 开始训练...")

#     # ========== 1. 构建图：边和边类型 ==========
#     edge_list = []
#     edge_type_list = []
#     valid_triples = []  # 存储有效的三元组用于训练
    
#     for h, r, t in train_triples:
#         if h in mapper.entity_to_id and r in mapper.relation_to_id and t in mapper.entity_to_id:
#             h_id = mapper.entity_to_id[h]
#             t_id = mapper.entity_to_id[t]
#             r_id = mapper.relation_to_id[r]
#             # 正向
#             edge_list.append([h_id, t_id])
#             edge_type_list.append(r_id)
#             # 逆向（关系 ID 偏移）
#             edge_list.append([t_id, h_id])
#             edge_type_list.append(r_id + mapper.relation_count)
#             # 保存有效三元组
#             valid_triples.append((h_id, r_id, t_id))

#     if len(edge_list) == 0:
#         raise ValueError("No valid edges.")

#     edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
#     edge_type = torch.tensor(edge_type_list, dtype=torch.long)

#     # 将有效三元组转换为tensor
#     valid_triples_tensor = torch.tensor(valid_triples, dtype=torch.long)

#     # ========== 2. 构建 PyG Data 对象 ==========
#     data = Data(
#         num_nodes=mapper.entity_count,
#         edge_index=edge_index,
#         edge_type=edge_type
#     )

#     # ========== 3. 使用 NeighborLoader 进行子图采样 ==========
#     loader = NeighborLoader(
#         data,
#         num_neighbors=[10, 10],
#         batch_size=1024,
#         input_nodes=torch.arange(mapper.entity_count),
#         shuffle=True,
#         num_workers=4,
#     )

#     # ========== 4. 初始化模型 ==========
#     model = ModelClass(mapper.entity_count, mapper.relation_count, EMBEDDING_DIM).to(device)
#     model = model.to(device)

#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_FACTOR)

#     model.train()

#     # ========== 5. 训练主循环 ==========
#     for epoch in range(EPOCHS):
#         epoch_loss = 0.0
#         pbar = tqdm(loader, desc=f"{model_name} Epoch {epoch+1}", total=len(loader))

#         for batch in pbar:
#             batch = batch.to(device)
#             optimizer.zero_grad()

#             # 获取节点特征
#             x = batch.x

#             # 前向传播获取节点嵌入
#             if model_name in ['RGCN', 'RGAT', 'GNN-Film']:
#                 print(x)
#                 z = model(x, batch.edge_index, batch.edge_type)
#             else:
#                 z = model(x, batch.edge_index)

#             # 获取根节点嵌入
#             root_nodes = z[:batch.batch_size]

#             # ========== 6. 负采样和损失计算 ==========
#             # 在当前batch中查找对应的正样本
#             batch_node_ids = batch.n_id[:batch.batch_size].cpu()
#             batch_node_set = set(batch_node_ids.tolist())
            
#             # 查找当前batch中的正样本
#             pos_samples = []
#             for h_id, r_id, t_id in valid_triples:
#                 if h_id in batch_node_set or t_id in batch_node_set:
#                     # 找到在batch中的局部ID
#                     try:
#                         local_h = (batch.n_id == h_id).nonzero(as_tuple=True)[0].item()
#                         local_t = (batch.n_id == t_id).nonzero(as_tuple=True)[0].item()
#                         pos_samples.append((local_h, r_id, local_t))
#                     except:
#                         continue
            
#             if len(pos_samples) == 0:
#                 continue
                
#             pos_samples = torch.tensor(pos_samples, dtype=torch.long).to(device)
            
#             # 提取头尾实体和关系
#             h_idx = pos_samples[:, 0]
#             r_idx = pos_samples[:, 1]
#             t_idx = pos_samples[:, 2]
            
#             # 获取嵌入
#             h_emb = root_nodes[h_idx]
#             r_emb = model.get_relation_embeddings()[r_idx].to(device)
#             t_emb = root_nodes[t_idx]
            
#             # TransE 打分函数: ||h + r - t||_2
#             pos_score = torch.norm(h_emb + r_emb - t_emb, p=2, dim=1)
            
#             # 负采样：替换尾实体
#             neg_t_idx = torch.randint(0, root_nodes.size(0), t_idx.shape, device=device)
#             neg_t_emb = root_nodes[neg_t_idx]
#             neg_score = torch.norm(h_emb + r_emb - neg_t_emb, p=2, dim=1)
            
#             # Margin-based ranking loss
#             margin = 1.0
#             loss = torch.mean(torch.relu(pos_score - neg_score + margin))
            
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#             pbar.set_postfix(loss=loss.item())

#         scheduler.step()
#         print(f"[{model_name}] Epoch {epoch+1} Loss: {epoch_loss / len(pbar):.4f}")

#     # ========== 7. 推理全图实体嵌入 ==========
#     with torch.no_grad():
#         model.eval()
#         embeddings = []
#         for batch in loader:
#             batch = batch.to(device)
#             if model_name in ['RGCN', 'RGAT', 'GNN-Film']:
#                 out = model(batch.x, batch.edge_index, batch.edge_type)
#             else:
#                 out = model(batch.x, batch.edge_index)
#             embeddings.append(out[:batch.batch_size])
#         ent_emb = torch.cat(embeddings, dim=0).cpu()

#         rel_emb = model.get_relation_embeddings()

#     # ========== 8. 保存模型 ==========
#     torch.save({
#         'entity_emb': ent_emb,
#         'rel_emb': rel_emb
#     }, path)
#     print(f"[{model_name}] 模型已保存")

#     return ent_emb, rel_emb

# ==================== 统一训练函数 ====================
def train_gnn_model(ModelClass, model_name, train_triples, mapper, device):
    ## 【TODO】这个方法，在做rgcn的时候会爆显存。但是姑且这么着吧。
    path = TRAINED_MODEL_PATHS[model_name]
    if os.path.exists(path) and not FORCE_RETRAIN:
        print(f"[{model_name}] 模型已存在，跳过训练")
        ckpt = torch.load(path, map_location=device)
        return ckpt['entity_emb'], ckpt['rel_emb']

    print(f"[{model_name}] 开始训练...")

    # 构建图（带逆关系）
    edge_list = []
    edge_type_list = []

    for h, r, t in train_triples:
        if h in mapper.entity_to_id and r in mapper.relation_to_id and t in mapper.entity_to_id:
            h_id = mapper.entity_to_id[h]
            t_id = mapper.entity_to_id[t]
            r_id = mapper.relation_to_id[r]

            # 正向
            edge_list.append([h_id, t_id])
            edge_type_list.append(r_id)
            # 逆向
            edge_list.append([t_id, h_id])
            edge_type_list.append(r_id + mapper.relation_count)

    if len(edge_list) == 0:
        raise ValueError("No valid edges.")

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)
    edge_type = torch.tensor(edge_type_list, dtype=torch.long).to(device)

    # 初始化模型
    model = ModelClass(mapper.entity_count, mapper.relation_count, EMBEDDING_DIM).to(device)
    model.edge_index = edge_index
    if model_name in ['RGCN', 'RGAT', 'GNN-Film']:
        model.edge_type = edge_type

    # DataLoader
    dataset = list(range(len(train_triples)))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_FACTOR)
    model.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"{model_name} Epoch {epoch+1}", total=len(loader))

        for batch_idx, _ in enumerate(pbar):
            optimizer.zero_grad()

            # 构造正样本
            pos_batch = train_triples[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE]
            h_ids = [mapper.entity_to_id[h] for h, r, t in pos_batch if h in mapper.entity_to_id]
            r_ids = [mapper.relation_to_id[r] for h, r, t in pos_batch if r in mapper.relation_to_id]
            t_ids = [mapper.entity_to_id[t] for h, r, t in pos_batch if t in mapper.entity_to_id]

            if len(h_ids) == 0:
                continue

            h_ids = torch.tensor(h_ids, device=device)
            r_ids = torch.tensor(r_ids, device=device)
            t_ids = torch.tensor(t_ids, device=device)

            # 获取嵌入
            z = model.get_entity_embeddings(device)
            h_emb = z[h_ids]
            t_emb = z[t_ids]
            r_emb = model.rel_emb(r_ids)

            # TransE 打分
            pos_score = -torch.norm(h_emb + r_emb - t_emb, p=2, dim=1)

            # In-batch 负采样（替换尾实体）
            neg_t_ids = t_ids.roll(shifts=1, dims=0)
            neg_t_emb = z[neg_t_ids]
            neg_score = -torch.norm(h_emb + r_emb - neg_t_emb, p=2, dim=1)

            # Margin-based loss
            margin = 1.0
            loss = (margin + neg_score - pos_score).clamp(min=0).mean()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        print(f"[{model_name}] Epoch {epoch+1} Loss: {epoch_loss / len(pbar):.4f}")

    # 保存
    with torch.no_grad():
        ent_emb = model.get_entity_embeddings(device).cpu().detach()
        rel_emb = model.get_relation_embeddings()
        torch.save({
            'entity_emb': ent_emb,
            'rel_emb': rel_emb
        }, path)
        print(f"[{model_name}] 模型已保存")

    return ent_emb, rel_emb

# ==================== 统一评估函数 ====================
def evaluate_embedding_relational(ent_emb, rel_emb, dataset, mapper, name):
    print(f"📊 评估 {name} (关系感知, TransE 风格) ...")
    ent_emb = ent_emb.astype(np.float32)
    rel_emb = rel_emb.astype(np.float32)

    hits1 = hits3 = hits10 = mrr = 0
    total = 0
    index = faiss.IndexFlatL2(ent_emb.shape[1])
    index.add(ent_emb)

    for h, r, t in tqdm(dataset.triples, desc=f"Evaluating {name}"):
        if h not in mapper.entity_to_id or r not in mapper.relation_to_id or t not in mapper.entity_to_id:
            continue

        h_id = mapper.entity_to_id[h]
        r_id = mapper.relation_to_id[r]
        t_id = mapper.entity_to_id[t]

        query = (ent_emb[h_id] + rel_emb[r_id]).astype(np.float32).reshape(1, -1)
        _, indices = index.search(query, k=1000)
        rank = np.where(indices[0] == t_id)[0]

        if len(rank) == 0:
            continue
        rank = rank[0] + 1
        total += 1

        hits1 += 1 if rank <= 1 else 0
        hits3 += 1 if rank <= 3 else 0
        hits10 += 1 if rank <= 10 else 0
        mrr += 1.0 / rank

    if total > 0:
        hits1 /= total
        hits3 /= total
        hits10 /= total
        mrr /= total

    print(f"{name} - Hits@1: {hits1:.4f}, Hits@3: {hits3:.4f}, Hits@10: {hits10:.4f}, MRR: {mrr:.4f}")
    return hits1, hits3, hits10, mrr

# ==================== 预测函数（RRF 融合实体嵌入）=====================
def predict_ensemble(embeddings_with_weights, test_dataset, mapper, rrf_k=60):
    print("🔍 开始融合预测 (RRF) ...")
    results = []
    first_emb = list(embeddings_with_weights.values())[0][0]
    embedding_dim = first_emb.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(first_emb.astype(np.float32))

    for h, r, _ in tqdm(test_dataset.triples, desc="Predict"):
        if h not in mapper.entity_to_id:
            continue
        h_id = mapper.entity_to_id[h]
        h_emb = first_emb[h_id].reshape(1, -1).astype(np.float32)
        _, indices = index.search(h_emb, 1000)
        rrf_scores = np.zeros(mapper.entity_count)
        for name, (emb, weight) in embeddings_with_weights.items():
            for rank, idx in enumerate(indices[0]):
                rrf_scores[idx] += weight / (rrf_k + rank + 1)
        ranked = np.argsort(rrf_scores)[::-1][:10]
        preds = [mapper.id_to_entity[i] for i in ranked]
        results.append('\t'.join([h, r] + preds))

    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results) + '\n')
    zip_path = OUTPUT_FILE_PATH.replace(".tsv", "") + f"__{scheme_type}.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(OUTPUT_FILE_PATH, arcname=os.path.basename(OUTPUT_FILE_PATH))
    print(f"✅ 预测结果已保存: {zip_path}")

# ==================== 主函数 ====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")

    train_data = KnowledgeGraphDataset(TRAIN_FILE_PATH, max_lines=MAX_LINES, is_train=True)
    dev_data = KnowledgeGraphDataset(DEV_FILE_PATH, max_lines=MAX_LINES, is_test=False, is_train=False)
    test_data = KnowledgeGraphDataset(TEST_FILE_PATH, max_lines=MAX_LINES, is_test=True, is_train=False)

    mapper = EntityRelationMapper()
    mapper.build_mappings(train_data, dev_data, test_data)
    print(f"实体数: {mapper.entity_count}, 关系数: {mapper.relation_count}")

    # 训练所有模型
    graphsage_ent, graphsage_rel = train_gnn_model(GraphSAGE, 'GraphSAGE', train_data.triples, mapper, device)
    gcn_ent, gcn_rel = train_gnn_model(GCN, 'GCN', train_data.triples, mapper, device)
    gat_ent, gat_rel = train_gnn_model(GAT, 'GAT', train_data.triples, mapper, device)
    # rgcn_ent, rgcn_rel = train_gnn_model(RGCN, 'RGCN', train_data.triples, mapper, device)
    rgat_ent, rgat_rel = train_gnn_model(RGAT, 'RGAT', train_data.triples, mapper, device)
    film_ent, film_rel = train_gnn_model(GNNFiLM, 'GNN-Film', train_data.triples, mapper, device)

    # 评估
    evaluate_embedding_relational(graphsage_ent.cpu().numpy(), graphsage_rel.cpu().numpy(), dev_data, mapper, "GraphSAGE")
    evaluate_embedding_relational(gcn_ent.cpu().numpy(), gcn_rel.cpu().numpy(), dev_data, mapper, "GCN")
    evaluate_embedding_relational(gat_ent.cpu().numpy(), gat_rel.cpu().numpy(), dev_data, mapper, "GAT")
    # evaluate_embedding_relational(rgcn_ent.cpu().numpy(), rgcn_rel.cpu().numpy(), dev_data, mapper, "RGCN")
    evaluate_embedding_relational(rgat_ent.cpu().numpy(), rgat_rel.cpu().numpy(), dev_data, mapper, "RGAT")
    evaluate_embedding_relational(film_ent.cpu().numpy(), film_rel.cpu().numpy(), dev_data, mapper, "GNN-Film")

    # 融合预测（使用实体嵌入）
    embeddings_with_weight = {
        'GraphSAGE': (graphsage_ent.cpu().numpy(), 1.0),
        'GCN': (gcn_ent.cpu().numpy(), 1.2),
        'GAT': (gat_ent.cpu().numpy(), 1.3),
        # 'RGCN': (rgcn_ent.cpu().numpy(), 1.5),
        'RGAT': (rgat_ent.cpu().numpy(), 1.4),
        'GNN-Film': (film_ent.cpu().numpy(), 1.2),
    }
    predict_ensemble(embeddings_with_weight, test_data, mapper)

    print("🎉 所有 GNN 模型训练、评估、预测完成！")

if __name__ == "__main__":
    main()

## 输出结果：
# /home/xiuminke/miniconda3/envs/ml12/lib/python3.11/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
#   import pynvml  # type: ignore[import]
# /home/xiuminke/miniconda3/envs/ml12/lib/python3.11/site-packages/torch_geometric/typing.py:97: UserWarning: An issue occurred while importing 'torch-cluster'. Disabling its usage. Stacktrace: All ufuncs must have type `numpy.ufunc`. Received (<ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>)
#   warnings.warn(f"An issue occurred while importing 'torch-cluster'. "
# /home/xiuminke/miniconda3/envs/ml12/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
#   from .autonotebook import tqdm as notebook_tqdm
# 🚀 使用设备: cuda
# 加载数据: /mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_train.tsv
# 共加载 1242550 个三元组
# 加载数据: /mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_dev.tsv
# 共加载 5000 个三元组
# 加载数据: /mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_test.tsv
# 共加载 5000 个三元组
# 实体数: 249747, 关系数: 500
# [GraphSAGE] 模型已存在，跳过训练
# [GCN] 模型已存在，跳过训练
# [GAT] 模型已存在，跳过训练
# [RGAT] 模型已存在，跳过训练
# [GNN-Film] 模型已存在，跳过训练
# 📊 评估 GraphSAGE (关系感知, TransE 风格) ...
# Evaluating GraphSAGE: 100%|████████████████████████████████████████████████████████| 5000/5000 [00:10<00:00, 467.69it/s]
# GraphSAGE - Hits@1: 0.0036, Hits@3: 0.0107, Hits@10: 0.0178, MRR: 0.0133
# 📊 评估 GCN (关系感知, TransE 风格) ...
# Evaluating GCN: 100%|██████████████████████████████████████████████████████████████| 5000/5000 [00:10<00:00, 474.72it/s]
# GCN - Hits@1: 0.2532, Hits@3: 0.3681, Hits@10: 0.5044, MRR: 0.3366
# 📊 评估 GAT (关系感知, TransE 风格) ...
# Evaluating GAT: 100%|██████████████████████████████████████████████████████████████| 5000/5000 [00:10<00:00, 477.96it/s]
# GAT - Hits@1: 0.0000, Hits@3: 0.0000, Hits@10: 0.0116, MRR: 0.0081
# 📊 评估 RGAT (关系感知, TransE 风格) ...
# Evaluating RGAT: 100%|█████████████████████████████████████████████████████████████| 5000/5000 [00:10<00:00, 474.09it/s]
# RGAT - Hits@1: 0.0000, Hits@3: 0.0025, Hits@10: 0.0123, MRR: 0.0084
# 📊 评估 GNN-Film (关系感知, TransE 风格) ...
# Evaluating GNN-Film: 100%|█████████████████████████████████████████████████████████| 5000/5000 [00:10<00:00, 483.11it/s]
# GNN-Film - Hits@1: 0.0000, Hits@3: 0.0000, Hits@10: 0.0000, MRR: 0.0010
# 🔍 开始融合预测 (RRF) ...
# Predict: 100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:27<00:00, 184.10it/s]
# ✅ 预测结果已保存: /mnt/d/forCoding_data/Tianchi_EcommerceKG/preprocessedData/OpenBG500_test__gSage_GCN_RGCN_GAT__relational.zip
# 🎉 所有 GNN 模型训练、评估、预测完成！