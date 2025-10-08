import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import time
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# 设置随机种子以确保结果可复现
def set_seed(seed=123):
    random.seed(seed)  # 设置Python随机种子
    np.random.seed(seed)  # 设置NumPy随机种子
    torch.manual_seed(seed)  # 设置PyTorch CPU随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置当前GPU随机种子
        torch.cuda.manual_seed_all(seed)  # 设置所有GPU随机种子
        torch.backends.cudnn.deterministic = True  # 确保CUDA卷积操作的确定性

set_seed()

# 数据路径
TRAIN_FILE_PATH = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_train.tsv"
TEST_FILE_PATH = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_test.tsv"
DEV_FILE_PATH = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_dev.tsv" # 开发集路径
OUTPUT_FILE_PATH = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/preprocessedData/OpenBG500_test.tsv"

# 超参数
EMBEDDING_DIM = 200  # 实体和关系嵌入的维度
LEARNING_RATE = 0.001  # 学习率
WEIGHT_DECAY = 1e-5  # 权重衰减（L2正则化）
BATCH_SIZE = 1024  # 训练批次大小
NEGATIVE_SAMPLES = 5  # 每个正样本对应的负样本数量

EPOCHS = 6  # 训练的epoch数量
MAX_LINES = None  # 限制训练数据的行数，None表示使用全部数据
max_head_entities = None  # 限制预测的头实体数量，None表示处理全部

# 学习率调度器参数
LR_DECAY_FACTOR = 0.5  # 学习率衰减因子
LR_DECAY_STEP = 50  # 每隔多少个epoch衰减一次学习率

# 数据加载类
class KnowledgeGraphDataset(Dataset):
    def __init__(self, file_path, is_test=False, is_dev=False, max_lines=None):
        self.is_test = is_test
        self.is_dev = is_dev
        self.triples = []
        
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                if max_lines is not None and line_count >= max_lines:
                    break
                
                parts = line.strip().split('\t')
                if is_test:
                    # 测试文件只有头实体和关系
                    if len(parts) >= 2:
                        h, r = parts[0], parts[1]
                        self.triples.append((h, r, None))
                        line_count += 1
                else:
                    # 训练和开发文件有完整的三元组
                    if len(parts) >= 3:
                        h, r, t = parts[0], parts[1], parts[2]
                        self.triples.append((h, r, t))
                        line_count += 1
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        return self.triples[idx]

# 实体和关系映射管理器
class EntityRelationMapper:
    def __init__(self):
        self.entity_to_id = {}  # 实体到ID的映射字典
        self.id_to_entity = {}  # ID到实体的映射字典
        self.relation_to_id = {}  # 关系到ID的映射字典
        self.id_to_relation = {}  # ID到关系的映射字典
        self.entity_count = 0  # 实体数量
        self.relation_count = 0  # 关系数量
        
    def add_entity(self, entity):
        # 如果实体不存在于映射中，则添加
        if entity not in self.entity_to_id:
            self.entity_to_id[entity] = self.entity_count
            self.id_to_entity[self.entity_count] = entity
            self.entity_count += 1
    
    def add_relation(self, relation):
        # 如果关系不存在于映射中，则添加
        if relation not in self.relation_to_id:
            self.relation_to_id[relation] = self.relation_count
            self.id_to_relation[self.relation_count] = relation
            self.relation_count += 1
    
    def build_mappings(self, train_dataset, test_dataset, dev_dataset=None):
        # 从训练数据构建映射
        for h, r, t in train_dataset.triples:
            self.add_entity(h)
            self.add_entity(t)
            self.add_relation(r)
        
        # 从测试数据构建映射（确保所有实体和关系都被包含）
        for h, r, _ in test_dataset.triples:
            self.add_entity(h)
            self.add_relation(r)
        
        # 从开发集构建映射（确保所有实体和关系都被包含）
        if dev_dataset is not None:
            for h, r, t in dev_dataset.triples:
                self.add_entity(h)
                self.add_entity(t)
                self.add_relation(r)

# GraphSAGE模型实现
class GraphSAGEKG(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=200, hidden_dim=100):
        super(GraphSAGEKG, self).__init__()
        
        # 实体和关系嵌入层
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # GraphSAGE卷积层
        self.conv1 = SAGEConv(embedding_dim, hidden_dim, aggr='mean')
        self.conv2 = SAGEConv(hidden_dim, embedding_dim, aggr='mean')
        
        # 输出层用于评分函数
        self.score_layer = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        
        # 初始化权重
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        
    def forward(self, data):
        # 使用模型自己的实体嵌入作为输入特征
        x = self.entity_embeddings.weight
        edge_index = data.edge_index
        
        # 应用GraphSAGE卷积
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        
        return h
        
    def get_entity_embeddings(self):
        # 返回实体嵌入（用于预测）
        return self.entity_embeddings.weight
        
    def get_relation_embeddings(self):
        # 返回关系嵌入
        return self.relation_embeddings.weight
        
    def score_triple(self, h, r, t):
        # 评分函数：计算三元组的得分
        # 将头实体、关系、尾实体的嵌入连接起来
        combined = torch.cat([h, r, t], dim=1)
        # 通过评分层计算得分
        score = self.score_layer(combined)
        return score

# 创建图数据结构
def create_graph_data(train_dataset, mapper):
    # 获取所有实体和关系的ID
    edges = []
    edge_types = []
    
    # 构建图的边和边类型
    for h, r, t in train_dataset.triples:
        h_id = mapper.entity_to_id[h]
        t_id = mapper.entity_to_id[t]
        r_id = mapper.relation_to_id[r]
        
        # 添加正向边 (h -> t) 和反向边 (t -> h)
        edges.append((h_id, t_id))
        edge_types.append(r_id)
        edges.append((t_id, h_id))
        edge_types.append(r_id + mapper.relation_count)  # 反向关系ID
    
    # 转换为PyTorch Geometric的数据格式
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)
    
    # 初始化实体特征为嵌入向量
    x = torch.nn.Embedding(mapper.entity_count, EMBEDDING_DIM)(torch.arange(mapper.entity_count))
    
    # 创建数据对象
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    
    return data

# 自定义的批处理函数
def collate_fn(batch, mapper):
    h_batch = [item[0] for item in batch]
    r_batch = [item[1] for item in batch]
    t_batch = [item[2] for item in batch]
    
    # 批量转换为ID，减少循环开销
    h_ids = torch.tensor([mapper.entity_to_id[h] for h in h_batch])
    r_ids = torch.tensor([mapper.relation_to_id[r] for r in r_batch])
    
    if t_batch[0] is not None:
        # 训练数据
        t_ids = torch.tensor([mapper.entity_to_id[t] for t in t_batch if t is not None])
        return h_ids, r_ids, t_ids
    else:
        # 测试数据
        return h_ids, r_ids, None

# 训练函数
def train_model(model, train_dataset, graph_data, mapper, device):
    # 创建自定义的collate_fn函数用于批量处理数据
    def custom_collate(batch):
        return collate_fn(batch, mapper)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             collate_fn=custom_collate, pin_memory=True, num_workers=8)
    
    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_FACTOR)
    
    # 启用自动混合精度训练 - 使用新的API
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    
    # 将图数据移至设备
    graph_data = graph_data.to(device)
    
    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} 训练中...")
        
        for batch_idx, batch in enumerate(train_loader):
            h_ids, r_ids, t_ids = batch
            
            # 转移到设备上
            h_ids = h_ids.to(device, non_blocking=True)
            r_ids = r_ids.to(device, non_blocking=True)
            t_ids = t_ids.to(device, non_blocking=True)
            
            # 使用向量化操作生成负样本
            batch_size = h_ids.size(0)
            t_neg_ids = torch.randint(0, mapper.entity_count, 
                                     (batch_size, NEGATIVE_SAMPLES), 
                                     device=device, dtype=torch.long)
            
            # 确保负样本不等于正样本
            pos_tile = t_ids.unsqueeze(1).expand(-1, NEGATIVE_SAMPLES)
            mask = (t_neg_ids == pos_tile)
            
            while mask.any():
                new_neg_ids = torch.randint(0, mapper.entity_count, 
                                          (mask.sum().item(),), 
                                          device=device, dtype=torch.long)
                t_neg_ids[mask] = new_neg_ids
                pos_tile = t_ids.unsqueeze(1).expand(-1, NEGATIVE_SAMPLES)
                mask = (t_neg_ids == pos_tile)
            
            # 使用自动混合精度进行前向传播 - 使用新的API
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                # 通过GraphSAGE获取更新后的实体嵌入
                updated_embeddings = model(graph_data)
                
                # 获取头实体、关系和尾实体的嵌入
                h_emb = updated_embeddings[h_ids]
                r_emb = model.relation_embeddings(r_ids)
                t_emb = updated_embeddings[t_ids]
                
                # 获取负样本尾实体的嵌入
                t_neg_emb = updated_embeddings[t_neg_ids]
                
                # 计算正样本得分
                pos_score = model.score_triple(h_emb, r_emb, t_emb)
                
                # 计算负样本得分
                h_emb_expanded = h_emb.unsqueeze(1).expand(-1, NEGATIVE_SAMPLES, -1)
                r_emb_expanded = r_emb.unsqueeze(1).expand(-1, NEGATIVE_SAMPLES, -1)
                
                # 计算每个负样本的得分
                neg_scores = []
                for i in range(NEGATIVE_SAMPLES):
                    neg_score = model.score_triple(h_emb_expanded[:, i], r_emb_expanded[:, i], t_neg_emb[:, i])
                    neg_scores.append(neg_score)
                
                neg_score = torch.cat(neg_scores, dim=1)
                
                # 计算损失 (max-margin)
                pos_score_expanded = pos_score.expand_as(neg_score)
                individual_loss = torch.relu(neg_score - pos_score_expanded + 1.0)  # 注意这里与TransE相反
                loss = torch.mean(individual_loss) * batch_size
            
            # 反向传播和优化
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                progress = (batch_idx + 1) / len(train_loader) * 100
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f} - Progress: {progress:.1f}%", end='\r')
        
        # 更新学习率
        scheduler.step()
        
        # 打印平均损失和训练速度
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1}/{EPOCHS} - Average Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f} - Time: {epoch_time:.2f}s")

# 评测函数 - 评估GraphSAGE模型在开发集上的表现 (加速版)
def evaluate(model, dev_dataset, graph_data, mapper, device, batch_size=128, max_entities_per_batch=10000):
    model.eval()
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_10 = 0
    mrr_score = 0
    total_count = 0
    
    total_dev_triples = len(dev_dataset.triples)
    print(f"\n开始在开发集上评估模型...")
    print(f"将评估 {total_dev_triples} 个三元组")
    
    with torch.no_grad():
        # 获取更新后的实体嵌入
        updated_embeddings = model(graph_data)
        
        # 创建批次进行处理
        for batch_start in range(0, total_dev_triples, batch_size):
            batch_end = min(batch_start + batch_size, total_dev_triples)
            batch_triples = dev_dataset.triples[batch_start:batch_end]
            current_batch_size = len(batch_triples)
            
            # 获取批次中的头实体、关系和尾实体
            h_list = [h for h, _, _ in batch_triples]
            r_list = [r for _, r, _ in batch_triples]
            t_list = [t for _, _, t in batch_triples]
            
            # 转换为ID和张量
            h_ids = torch.tensor([mapper.entity_to_id[h] for h in h_list], device=device)
            r_ids = torch.tensor([mapper.relation_to_id[r] for r in r_list], device=device)
            t_ids = torch.tensor([mapper.entity_to_id[t] for t in t_list], device=device)
            
            # 获取头实体和关系的嵌入
            h_emb = updated_embeddings[h_ids]  # [current_batch_size, embedding_dim]
            r_emb = model.relation_embeddings(r_ids)  # [current_batch_size, embedding_dim]
            
            # 分块处理实体，避免一次性加载所有实体导致显存溢出
            batch_scores = torch.zeros(current_batch_size, mapper.entity_count, device=device)
            
            # 分块计算得分 - 向量化实现
            for entity_start in range(0, mapper.entity_count, max_entities_per_batch):
                entity_end = min(entity_start + max_entities_per_batch, mapper.entity_count)
                entity_chunk_size = entity_end - entity_start
                
                # 获取当前块的实体嵌入
                entity_ids = torch.arange(entity_start, entity_end, device=device)
                entity_embeddings = updated_embeddings[entity_ids]  # [entity_chunk_size, embedding_dim]
                
                # 向量化计算当前块的得分
                h_expanded = h_emb.unsqueeze(1)  # [current_batch_size, 1, embedding_dim]
                r_expanded = r_emb.unsqueeze(1)  # [current_batch_size, 1, embedding_dim]
                t_expanded = entity_embeddings.unsqueeze(0)  # [1, entity_chunk_size, embedding_dim]
                
                # 扩展维度以便广播
                h_broadcast = h_expanded.expand(-1, entity_chunk_size, -1)
                r_broadcast = r_expanded.expand(-1, entity_chunk_size, -1)
                t_broadcast = t_expanded.expand(current_batch_size, -1, -1)
                
                # 准备计算批次得分
                chunk_scores = []
                for i in range(current_batch_size):
                    # 向量化计算一个批次的得分
                    h_chunk = h_broadcast[i]
                    r_chunk = r_broadcast[i]
                    t_chunk = t_broadcast[i]
                    
                    # 向量化计算得分
                    combined = torch.cat([h_chunk, r_chunk, t_chunk], dim=1)
                    scores = model.score_layer(combined).squeeze(1)
                    chunk_scores.append(scores)
                
                chunk_scores_tensor = torch.stack(chunk_scores)  # [current_batch_size, entity_chunk_size]
                batch_scores[:, entity_start:entity_end] = chunk_scores_tensor
                
                # 清理中间变量，释放显存
                del entity_embeddings, h_broadcast, r_broadcast, t_broadcast, chunk_scores_tensor
            
            # 查找每个样本中正确尾实体的得分
            correct_scores = torch.gather(batch_scores, 1, t_ids.view(-1, 1))  # [current_batch_size, 1]
            
            # 计算排名（得分越高排名越高） - 向量化实现
            # 计算有多少实体得分比正确实体高（排名即这个数量+1）
            num_better_entities = (batch_scores > correct_scores).sum(dim=1)  # [current_batch_size]
            ranks = num_better_entities + 1  # [current_batch_size] - 排名从1开始
            
            # 计算指标
            hits_at_1 += (ranks == 1).sum().item()
            hits_at_3 += (ranks <= 3).sum().item()
            hits_at_10 += (ranks <= 10).sum().item()
            
            # 对于排名n>10的实体，得分为0
            mrr_contributions = torch.where(ranks <= 10, 1.0 / ranks, torch.tensor(0.0, device=device))
            mrr_score += mrr_contributions.sum().item()
            total_count += current_batch_size
            
            # 显示进度条
            progress = min(total_count, total_dev_triples) / total_dev_triples * 100
            bar_length = 50
            filled_length = int(bar_length * progress // 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"\r  评估进度: |{bar}| {progress:.1f}% ({min(total_count, total_dev_triples)}/{total_dev_triples})", end='')
            
            # 清理显存
            del h_emb, r_emb, batch_scores, correct_scores, num_better_entities, ranks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print()  # 确保进度条完成后换行
        
        # 清理所有剩余的显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 计算最终指标
    hits_at_1 = hits_at_1 / total_count
    hits_at_3 = hits_at_3 / total_count
    hits_at_10 = hits_at_10 / total_count
    mrr_score = mrr_score / total_count
    
    print(f"\n开发集评估结果:")
    print(f"HITS@1: {hits_at_1:.4f}")
    print(f"HITS@3: {hits_at_3:.4f}")
    print(f"HITS@10: {hits_at_10:.4f}")
    print(f"MRR: {mrr_score:.4f}")
    
    return hits_at_1, hits_at_3, hits_at_10, mrr_score

# 预测函数 - 优化版本
def predict_tail_entities(model, test_dataset, graph_data, mapper, device, max_head_entities=None, batch_size=128, max_entities_per_batch=5000):
    model.eval()
    results = []
    total_test_triples = len(test_dataset.triples)
    
    process_count = total_test_triples if max_head_entities is None else min(max_head_entities, total_test_triples)
    print(f"\n开始预测尾实体...")
    print(f"将处理 {process_count} 个头实体/关系对")
    
    max_entities_per_batch = min(5000, mapper.entity_count)  # 限制最大实体数，防止显存溢出
    
    with torch.no_grad():
        # 获取更新后的实体嵌入
        updated_embeddings = model(graph_data)
        
        # 创建批次进行处理
        for batch_start in range(0, process_count, batch_size):
            batch_end = min(batch_start + batch_size, process_count)
            batch_triples = test_dataset.triples[batch_start:batch_end]
            
            # 获取批次中的头实体和关系
            h_list = [h for h, _, _ in batch_triples]
            r_list = [r for _, r, _ in batch_triples]
            
            # 转换为ID和张量
            h_ids = torch.tensor([mapper.entity_to_id[h] for h in h_list], device=device)
            r_ids = torch.tensor([mapper.relation_to_id[r] for r in r_list], device=device)
            
            # 获取嵌入向量
            h_emb = updated_embeddings[h_ids]  # [batch_size, embedding_dim]
            r_emb = model.relation_embeddings(r_ids)  # [batch_size, embedding_dim]
            
            # 分块处理实体，避免一次性加载所有实体导致显存溢出
            batch_scores = torch.zeros(len(batch_triples), mapper.entity_count, device=device)
            
            # 分块计算得分 - 向量化实现
            for entity_start in range(0, mapper.entity_count, max_entities_per_batch):
                entity_end = min(entity_start + max_entities_per_batch, mapper.entity_count)
                entity_chunk_size = entity_end - entity_start
                
                # 获取当前块的实体嵌入
                entity_ids = torch.arange(entity_start, entity_end, device=device)
                entity_embeddings = updated_embeddings[entity_ids]  # [chunk_size, embedding_dim]
                
                # 向量化计算当前块的得分
                h_expanded = h_emb.unsqueeze(1)  # [batch_size, 1, embedding_dim]
                r_expanded = r_emb.unsqueeze(1)  # [batch_size, 1, embedding_dim]
                t_expanded = entity_embeddings.unsqueeze(0)  # [1, chunk_size, embedding_dim]
                
                # 扩展维度以便广播
                h_broadcast = h_expanded.expand(-1, entity_chunk_size, -1)
                r_broadcast = r_expanded.expand(-1, entity_chunk_size, -1)
                t_broadcast = t_expanded.expand(len(batch_triples), -1, -1)
                
                # 准备计算批次得分
                chunk_scores = []
                for i in range(len(batch_triples)):
                    # 向量化计算一个批次的得分
                    h_chunk = h_broadcast[i]
                    r_chunk = r_broadcast[i]
                    t_chunk = t_broadcast[i]
                    
                    # 向量化计算得分
                    combined = torch.cat([h_chunk, r_chunk, t_chunk], dim=1)
                    scores = model.score_layer(combined).squeeze(1)
                    chunk_scores.append(scores)
                
                chunk_scores = torch.stack(chunk_scores)  # [batch_size, chunk_size]
                
                # 存储到完整得分矩阵中
                batch_scores[:, entity_start:entity_end] = chunk_scores
                
                # 清理中间变量，释放显存
                del entity_embeddings, chunk_scores
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 获取每个头实体/关系对的top10尾实体（得分最高的）
            _, top10_indices = torch.topk(batch_scores, k=10, dim=1)
            
            # 处理批次结果
            for i in range(len(batch_triples)):
                h, r, _ = batch_triples[i]
                top10_t_ids = top10_indices[i].tolist()
                
                # 转换回实体ID字符串
                top10_entities = [mapper.id_to_entity[t_id] for t_id in top10_t_ids]
                
                # 构建结果行
                result_line = [h, r] + top10_entities
                results.append('\t'.join(result_line))
            
            # 显示进度条
            progress = batch_end / process_count * 100
            bar_length = 50
            filled_length = int(bar_length * progress // 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"\r  预测进度: |{bar}| {progress:.1f}% ({batch_end}/{process_count})", end='')
            
            # 定期清理GPU内存
            if batch_start % (batch_size * 10) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    print()  # 确保进度条完成后换行
    
    # 写入结果文件
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')
    
    print(f"预测结果已保存到 {OUTPUT_FILE_PATH}")

# 主函数
def main():
    # 设置CUDA优化选项
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("正在加载数据集...")
    train_dataset = KnowledgeGraphDataset(TRAIN_FILE_PATH, max_lines=MAX_LINES)
    test_dataset = KnowledgeGraphDataset(TEST_FILE_PATH, is_test=True)
    dev_dataset = KnowledgeGraphDataset(DEV_FILE_PATH, is_dev=True)  # 加载开发集
    
    print(f"训练数据大小: {len(train_dataset)}")
    print(f"测试数据大小: {len(test_dataset)}")
    print(f"开发数据大小: {len(dev_dataset)}")
    
    # 构建实体和关系映射
    print("正在构建实体和关系映射...")
    mapper = EntityRelationMapper()
    mapper.build_mappings(train_dataset, test_dataset, dev_dataset)
    
    print(f"实体数量: {mapper.entity_count}")
    print(f"关系数量: {mapper.relation_count}")
    
    # 创建图数据
    print("正在创建图数据结构...")
    graph_data = create_graph_data(train_dataset, mapper)
    
    # 创建GraphSAGE模型
    print("正在创建GraphSAGE模型...")
    model = GraphSAGEKG(mapper.entity_count, mapper.relation_count, EMBEDDING_DIM)
    model.to(device)
    
    # 训练模型
    print("开始训练模型...")
    train_model(model, train_dataset, graph_data, mapper, device)
    
    # 在开发集上评估模型
    print("\n训练完成，开始在开发集上评估模型性能...")
    evaluate(model, dev_dataset, graph_data, mapper, device)
    
    # 预测尾实体
    predict_tail_entities(model, test_dataset, graph_data, mapper, device, max_head_entities)
    
    print("任务完成！")

if __name__ == "__main__":
    main()