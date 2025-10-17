import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
import time
from collections import defaultdict
import gc

# 设置随机种子以确保可重复性
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 检测MPS可用性并设置相应的后端优化
has_mps = False
try:
    has_mps = torch.backends.mps.is_available()
    print(f"MPS 可用性: {has_mps}")
    # MPS优化设置
    if has_mps:
        torch.backends.mps.enabled = True
        # 对于MPS，我们不需要CUDA相关的种子设置
        print("已启用MPS加速")
    else:
        # 对于CUDA设备，保持原有的设置
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(42) if torch.cuda.is_available() else None
except:
    has_mps = False
    print("MPS 不可用，将尝试使用CPU或CUDA")

# # 数据文件路径
# TRAIN_FILE_PATH = 'data/train.txt'
# TEST_FILE_PATH = 'data/test.txt'
# DEV_FILE_PATH = 'data/dev.txt'

# # 输出文件路径
# OUTPUT_FILE_PATH = 'output/transh_results.txt'
# TRAINED_MODEL_PATH = 'models/transh_model.pth'

# 数据路径
scheme_type = 'th_mac_mps_try'
# 数据路径 - 保持原有路径，确保与用户环境匹配
TRAIN_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/originalData/OpenBG500/OpenBG500_train.tsv"
TEST_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/originalData/OpenBG500/OpenBG500_test.tsv"
DEV_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/originalData/OpenBG500/OpenBG500_dev.tsv" # 开发集路径
OUTPUT_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/preprocessedData/OpenBG500_test.tsv"

# 更新模型保存路径到用户的本地路径
FORCE_RETRAIN = True
TRAINED_MODEL_PATH = f"/Users/minkexiu/Documents/GitHub/Tianchi_EcommerceKG_mac/trained_models/trained_model__{scheme_type}.pth"

# 超参数设置 - 针对MPS优化的参数配置
EMBEDDING_DIM = 100
MARGIN = 1.0
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 1
# 增大批次大小以更好地利用MPS并行能力
BATCH_SIZE = 512  # M3 Pro可以处理更大的批次
NEGATIVE_SAMPLES = 10
MAX_LINES = None  # 设置为None以加载全部数据，或设置具体数字如100000
MAX_HEAD_ENTITIES = None  # 设置为None以处理全部测试数据，或设置具体数字如1000
LR_DECAY_STEP = 5  # 学习率衰减步长
LR_DECAY_FACTOR = 0.1  # 学习率衰减因子

# 自定义数据集类
class KnowledgeGraphDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, is_test=False, is_dev=False, max_lines=None):
        self.triples = []
        self.is_test = is_test
        self.is_dev = is_dev
        self._load_data(file_path, max_lines)
    
    def _load_data(self, file_path, max_lines):
        print(f"正在加载数据从 {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if max_lines is not None:
                lines = lines[:max_lines]
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 3:
                    h, r, t = parts
                    self.triples.append((h, r, t))
                elif self.is_test and len(parts) == 2:
                    h, r = parts
                    # 测试数据没有尾实体，用占位符代替
                    self.triples.append((h, r, "<UNK>"))
        print(f"已加载 {len(self.triples)} 个三元组")
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        return self.triples[idx]

# 自定义collate_fn函数，用于批量处理数据
def collate_fn(batch):
    # 解包批次中的三元组
    h_list, r_list, t_list = zip(*batch)
    return list(h_list), list(r_list), list(t_list)

# 实体和关系映射管理器
class EntityRelationMapper:
    def __init__(self):
        self.entity_to_id = {}
        self.id_to_entity = {}
        self.relation_to_id = {}
        self.id_to_relation = {}
        self.entity_count = 0
        self.relation_count = 0
    
    def build_mappings(self, *datasets):
        entity_set = set()
        relation_set = set()
        
        # 收集所有实体和关系
        for dataset in datasets:
            for h, r, t in dataset.triples:
                entity_set.add(h)
                entity_set.add(t)
                relation_set.add(r)
        
        # 构建实体映射
        for entity in entity_set:
            if entity not in self.entity_to_id:
                self.entity_to_id[entity] = self.entity_count
                self.id_to_entity[self.entity_count] = entity
                self.entity_count += 1
        
        # 构建关系映射
        for relation in relation_set:
            if relation not in self.relation_to_id:
                self.relation_to_id[relation] = self.relation_count
                self.id_to_relation[self.relation_count] = relation
                self.relation_count += 1

# TransH模型定义
class TransH(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransH, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        # 实体嵌入
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        
        # 关系嵌入
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        # 关系超平面法向量
        self.normal_vectors = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.normal_vectors.weight)
    
    def _projection(self, embeddings, normals):
        # 将实体向量投影到关系特定的超平面上
        # embeddings: [batch_size, embedding_dim]
        # normals: [batch_size, embedding_dim]
        # 计算投影: e' = e - (e · w) * w，其中w是单位法向量
        # 使用clamp防止在MPS上可能出现的NaN值
        norm_normals = nn.functional.normalize(normals, p=2, dim=1)  # [batch_size, embedding_dim]
        # 在MPS上确保数值稳定性
        norm_normals = torch.clamp(norm_normals, min=-1.0, max=1.0)
        dot_product = torch.sum(embeddings * norm_normals, dim=1, keepdim=True)  # [batch_size, 1]
        projections = embeddings - dot_product * norm_normals  # [batch_size, embedding_dim]
        return projections
    
    def forward(self, h_ids, r_ids, t_ids, t_neg_ids=None):
        # 获取头实体、关系和尾实体的嵌入
        h = self.entity_embeddings(h_ids)  # [batch_size, embedding_dim]
        r = self.relation_embeddings(r_ids)  # [batch_size, embedding_dim]
        t = self.entity_embeddings(t_ids)  # [batch_size, embedding_dim]
        
        # 获取关系的法向量
        r_norm = self.normal_vectors(r_ids)  # [batch_size, embedding_dim]
        
        # 投影实体到关系特定的超平面
        h_proj = self._projection(h, r_norm)  # [batch_size, embedding_dim]
        t_proj = self._projection(t, r_norm)  # [batch_size, embedding_dim]
        
        # 计算正样本得分
        pos_score = torch.norm(h_proj + r - t_proj, p=1, dim=1)  # [batch_size]
        
        if t_neg_ids is not None:
            # 处理负样本
            batch_size, neg_samples = t_neg_ids.size()
            
            # 扩展头实体和关系嵌入以匹配负样本数量
            h_proj_expanded = h_proj.unsqueeze(1).expand(-1, neg_samples, -1).contiguous().view(-1, self.embedding_dim)  # [batch_size*neg_samples, embedding_dim]
            r_expanded = r.unsqueeze(1).expand(-1, neg_samples, -1).contiguous().view(-1, self.embedding_dim)  # [batch_size*neg_samples, embedding_dim]
            r_norm_expanded = r_norm.unsqueeze(1).expand(-1, neg_samples, -1).contiguous().view(-1, self.embedding_dim)  # [batch_size*neg_samples, embedding_dim]
            
            # 获取负样本尾实体嵌入
            t_neg = self.entity_embeddings(t_neg_ids.view(-1))  # [batch_size*neg_samples, embedding_dim]
            
            # 投影负样本尾实体
            t_neg_proj = self._projection(t_neg, r_norm_expanded)  # [batch_size*neg_samples, embedding_dim]
            
            # 计算负样本得分
            neg_score = torch.norm(h_proj_expanded + r_expanded - t_neg_proj, p=1, dim=1)  # [batch_size*neg_samples]
            neg_score = neg_score.view(batch_size, neg_samples)  # [batch_size, neg_samples]
            
            return pos_score, neg_score
        
        return pos_score
    
    def normalize_entities(self):
        # 归一化实体嵌入
        with torch.no_grad():
            # 确保在MPS上的数值稳定性
            norms = torch.norm(self.entity_embeddings.weight, dim=1, keepdim=True)
            # 防止除以零
            norms = torch.maximum(norms, torch.tensor(1e-12, device=norms.device))
            self.entity_embeddings.weight.div_(norms)

def train_model(model, train_dataset, mapper, device, epochs=EPOCHS, batch_size=BATCH_SIZE):
    # 根据设备类型设置适当的数据加载器参数
    pin_memory = device.type != 'mps'  # MPS不支持pin_memory
    # Mac上使用多进程数据加载器可能会有问题，设置为0或2
    num_workers = 0 if device.type == 'mps' else 2
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    
    # 定义优化器 - 改为AdamW优化器，通常比SGD更容易收敛
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_FACTOR)
    
    # 根据设备类型设置适当的混合精度训练
    use_amp = False  # MPS不支持autocast，设置为False
    scaler = None
    
    # 只有CUDA设备使用autocast
    if device.type == 'cuda':
        use_amp = True
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()  # 记录epoch开始时间
        
        # 为每个epoch打印进度条
        print(f"\nEpoch {epoch+1}/{epochs} 训练中...")
        
        # 使用enumerate来获取batch索引
        for batch_idx, batch in enumerate(train_loader):
            h_ids, r_ids, t_ids = batch
            
            # 直接在collate_fn中转换为张量，这里只需要转移到设备上
            h_ids = torch.tensor([mapper.entity_to_id[h] for h in h_ids], device=device, dtype=torch.long)
            r_ids = torch.tensor([mapper.relation_to_id[r] for r in r_ids], device=device, dtype=torch.long)
            t_ids = torch.tensor([mapper.entity_to_id[t] for t in t_ids], device=device, dtype=torch.long)
            
            # 使用向量化操作生成负样本，替代Python循环
            batch_size_current = h_ids.size(0)
            
            # 向量化生成负样本 - 优化1：大幅提高负样本生成效率
            # 创建负样本索引矩阵 [batch_size, negative_samples]
            t_neg_ids = torch.randint(0, mapper.entity_count, 
                                     (batch_size_current, NEGATIVE_SAMPLES), 
                                     device=device, dtype=torch.long)
            
            # 确保负样本不等于正样本
            # 找出t_neg_ids等于t_ids的位置
            pos_tile = t_ids.unsqueeze(1).expand(-1, NEGATIVE_SAMPLES)
            mask = (t_neg_ids == pos_tile)
            
            # 统计每个样本需要替换的负样本数量
            while mask.any():
                # 为需要替换的位置生成新的随机索引
                new_neg_ids = torch.randint(0, mapper.entity_count, 
                                          (mask.sum().item(),), 
                                          device=device, dtype=torch.long)
                
                # 替换无效的负样本
                t_neg_ids[mask] = new_neg_ids
                
                # 重新检查是否还有等于正样本的负样本
                pos_tile = t_ids.unsqueeze(1).expand(-1, NEGATIVE_SAMPLES)
                mask = (t_neg_ids == pos_tile)
            
            # 根据设备类型选择适当的计算方式
            if use_amp and device.type == 'cuda':
                # CUDA设备使用自动混合精度
                with torch.cuda.amp.autocast(enabled=True):
                    # 前向传播
                    pos_score, neg_score = model(h_ids, r_ids, t_ids, t_neg_ids)
                    
                    # 计算损失（基于margin的max-margin损失）
                    # 将pos_score扩展为与neg_score相同的维度
                    pos_score_expanded = pos_score.unsqueeze(1).expand_as(neg_score)
                    
                    # 计算每个样本的损失，并按批次大小进行平均，避免大批次带来的损失爆炸
                    individual_loss = torch.relu(pos_score_expanded - neg_score + MARGIN)
                    loss = torch.mean(individual_loss)  # 使用mean而不是sum，更稳定
                    
                    # 乘以批次大小以保持损失量级，便于与之前的训练比较
                    loss = loss * batch_size_current
                
                # 反向传播和优化 - 使用scaler进行梯度缩放
                optimizer.zero_grad()
                scaler.scale(loss).backward()  # 缩放损失并反向传播
                scaler.step(optimizer)  # 更新参数
                scaler.update()  # 更新缩放器
            else:
                # MPS或CPU设备使用普通计算方式
                # 前向传播
                pos_score, neg_score = model(h_ids, r_ids, t_ids, t_neg_ids)
                
                # 计算损失（基于margin的max-margin损失）
                # 将pos_score扩展为与neg_score相同的维度
                pos_score_expanded = pos_score.unsqueeze(1).expand_as(neg_score)
                
                # 计算每个样本的损失，并按批次大小进行平均，避免大批次带来的损失爆炸
                individual_loss = torch.relu(pos_score_expanded - neg_score + MARGIN)
                loss = torch.mean(individual_loss)  # 使用mean而不是sum，更稳定
                
                # 乘以批次大小以保持损失量级，便于与之前的训练比较
                loss = loss * batch_size_current
            
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 归一化实体嵌入
            model.normalize_entities()
            
            total_loss += loss.item()
            
            # 每个batch打印一次进度 - 优化：减少打印频率
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                progress = (batch_idx + 1) / len(train_loader) * 100
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f} - Progress: {progress:.1f}%", end='\r')
        
        # 每个epoch结束后更新学习率
        scheduler.step()
        
        # 每个epoch结束后打印平均损失和训练速度
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f} - Time: {epoch_time:.2f}s")

def evaluate(model, dev_dataset, mapper, device, batch_size=128, max_entities_per_batch=2000):
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
            h_ids = torch.tensor([mapper.entity_to_id[h] for h in h_list], device=device, dtype=torch.long)
            r_ids = torch.tensor([mapper.relation_to_id[r] for r in r_list], device=device, dtype=torch.long)
            t_ids = torch.tensor([mapper.entity_to_id[t] for t in t_list], device=device, dtype=torch.long)
            
            # 获取嵌入向量并计算h + r
            h_emb = model.entity_embeddings(h_ids)  # [current_batch_size, embedding_dim]
            r_emb = model.relation_embeddings(r_ids)  # [current_batch_size, embedding_dim]
            r_norm = model.normal_vectors(r_ids)  # [current_batch_size, embedding_dim]
            
            # 投影头实体
            h_proj = model._projection(h_emb, r_norm)  # [current_batch_size, embedding_dim]
            
            # 计算h_proj + r
            h_plus_r = h_proj + r_emb  # [current_batch_size, embedding_dim]
            
            # 分块处理实体，避免一次性加载所有实体导致显存溢出
            batch_scores = torch.zeros(current_batch_size, mapper.entity_count, device=device)
            
            # 分块计算得分 - 向量化实现
            for entity_start in range(0, mapper.entity_count, max_entities_per_batch):
                entity_end = min(entity_start + max_entities_per_batch, mapper.entity_count)
                entity_chunk_size = entity_end - entity_start
                
                # 获取当前块的实体嵌入
                entity_ids = torch.arange(entity_start, entity_end, device=device, dtype=torch.long)
                entity_embeddings = model.entity_embeddings(entity_ids)  # [entity_chunk_size, embedding_dim]
                
                # 投影实体嵌入到关系特定的超平面
                # 需要为每个实体复制关系法向量
                # 先扩展entity_embeddings以匹配r_norm的批次大小
                entity_embeddings_expanded = entity_embeddings.repeat(current_batch_size, 1)
                r_norm_expanded = r_norm.unsqueeze(1).expand(-1, entity_chunk_size, -1).contiguous().view(-1, model.embedding_dim)
                entity_embeddings_proj = model._projection(entity_embeddings_expanded, r_norm_expanded)
                
                # 计算当前块的得分 - 向量化操作
                h_plus_r_expanded = h_plus_r.unsqueeze(1)  # [current_batch_size, 1, embedding_dim]
                # 修复张量形状问题：将entity_embeddings_proj重新reshape为正确的形状
                entity_embeddings_proj_reshaped = entity_embeddings_proj.view(current_batch_size, entity_chunk_size, -1)
                chunk_scores = torch.norm(
                    h_plus_r_expanded - entity_embeddings_proj_reshaped, 
                    p=1, 
                    dim=2
                )  # [current_batch_size, entity_chunk_size]
                
                # 存储到完整得分矩阵中
                batch_scores[:, entity_start:entity_end] = chunk_scores
                
                # 清理中间变量，释放内存
                del entity_embeddings, entity_embeddings_proj, chunk_scores
                clear_memory(device)
            
            # 查找每个样本中正确尾实体的得分
            correct_scores = torch.gather(batch_scores, 1, t_ids.view(-1, 1))  # [current_batch_size, 1]
            
            # 计算排名（得分越低排名越高） - 通过向量化比较实现
            # 计算有多少实体得分比正确实体低（排名即这个数量+1）
            num_better_entities = (batch_scores < correct_scores).sum(dim=1)  # [current_batch_size]
            ranks = num_better_entities + 1  # [current_batch_size] - 排名从1开始
            
            # 根据要求修改MRR计算：排名超过10的得分为0
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
            
            # 定期清理内存
            if batch_start % (batch_size * 5) == 0:
                clear_memory(device)
            
            # 清理内存
            del h_emb, r_emb, r_norm, h_proj, h_plus_r, batch_scores, correct_scores, num_better_entities, ranks
            clear_memory(device)
    
    print()  # 确保进度条完成后换行
    
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

def predict_tail_entities(model, test_dataset, mapper, device, max_head_entities=None, batch_size=128, max_entities_per_batch=2000):
    model.eval()
    results = []
    total_test_triples = len(test_dataset.triples)
    
    # 确定要处理的头实体数量
    process_count = total_test_triples if max_head_entities is None else min(max_head_entities, total_test_triples)
    print(f"\n开始预测尾实体...")
    print(f"将处理 {process_count} 个头实体/关系对")
    
    # 计算每个批次能处理的最大实体数量（避免显存溢出）
    # 根据显存大小动态调整，假设16GB显存，每个实体需要embedding_dim*4字节
    max_entities_per_batch = min(2000, mapper.entity_count)  # 限制最大实体数以适应显存
    
    with torch.no_grad():
        # 创建批次进行处理，大幅提高GPU利用率
        for batch_start in range(0, process_count, batch_size):
            batch_end = min(batch_start + batch_size, process_count)
            batch_triples = test_dataset.triples[batch_start:batch_end]
            
            # 获取批次中的头实体和关系
            h_list = [h for h, _, _ in batch_triples]
            r_list = [r for _, r, _ in batch_triples]
            
            # 转换为ID和张量
            h_ids = torch.tensor([mapper.entity_to_id[h] for h in h_list], device=device, dtype=torch.long)
            r_ids = torch.tensor([mapper.relation_to_id[r] for r in r_list], device=device, dtype=torch.long)
            
            # 获取嵌入向量
            h_emb = model.entity_embeddings(h_ids)  # [batch_size, embedding_dim]
            r_emb = model.relation_embeddings(r_ids)  # [batch_size, embedding_dim]
            r_norm = model.normal_vectors(r_ids)  # [batch_size, embedding_dim]
            
            # 投影头实体
            h_proj = model._projection(h_emb, r_norm)  # [batch_size, embedding_dim]
            
            # 计算h_proj + r
            h_plus_r = h_proj + r_emb  # [batch_size, embedding_dim]
            
            # 分块处理实体，避免一次性加载所有实体导致显存溢出
            batch_scores = torch.zeros(len(batch_triples), mapper.entity_count, device=device)
            
            # 分块计算得分
            for entity_start in range(0, mapper.entity_count, max_entities_per_batch):
                entity_end = min(entity_start + max_entities_per_batch, mapper.entity_count)
                
                # 获取当前块的实体嵌入
                entity_ids = torch.arange(entity_start, entity_end, device=device, dtype=torch.long)
                entity_embeddings = model.entity_embeddings(entity_ids)  # [chunk_size, embedding_dim]
                
                # 投影实体嵌入到关系特定的超平面
                # 需要为每个实体复制关系法向量
                entity_chunk_size = entity_end - entity_start
                batch_size_current = len(batch_triples)
                # 先扩展entity_embeddings以匹配r_norm的批次大小
                entity_embeddings_expanded = entity_embeddings.repeat(batch_size_current, 1)
                r_norm_expanded = r_norm.unsqueeze(1).expand(-1, entity_chunk_size, -1).contiguous().view(-1, model.embedding_dim)
                entity_embeddings_proj = model._projection(entity_embeddings_expanded, r_norm_expanded)
                
                # 计算当前块的得分
                h_plus_r_expanded = h_plus_r.unsqueeze(1)  # [batch_size, 1, embedding_dim]
                # 修复张量形状问题：将entity_embeddings_proj重新reshape为正确的形状
                entity_embeddings_proj_reshaped = entity_embeddings_proj.view(batch_size_current, entity_chunk_size, -1)
                chunk_scores = torch.norm(h_plus_r_expanded - entity_embeddings_proj_reshaped, p=1, dim=2)  # [batch_size, chunk_size]
                
                # 存储到完整得分矩阵中
                batch_scores[:, entity_start:entity_end] = chunk_scores
                
                # 清理中间变量，释放内存
                del entity_embeddings, entity_embeddings_proj, chunk_scores
                clear_memory(device)
            
            # 获取每个头实体/关系对的top10尾实体
            _, top10_indices = torch.topk(-batch_scores, k=10, dim=1)  # 使用负数以获取最小的10个值
            
            # 处理批次结果
            for i in range(len(batch_triples)):
                h, r, _ = batch_triples[i]  # 正确解包3元素元组
                top10_t_ids = top10_indices[i].tolist()
                
                # 转换回实体ID字符串
                top10_entities = [mapper.id_to_entity[t_id] for t_id in top10_t_ids]
                
                # 构建结果行
                result_line = [h, r] + top10_entities
                results.append('\t'.join(result_line))
            
            # 显示进度条 - 改进版
            progress = batch_end / process_count * 100
            bar_length = 50
            filled_length = int(bar_length * progress // 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"\r  预测进度: |{bar}| {progress:.1f}% ({batch_end}/{process_count})", end='')
            
            # 定期清理内存
            if batch_start % (batch_size * 10) == 0:
                clear_memory(device)
    
    print()  # 确保进度条完成后换行
    
    # 写入结果文件
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')
    
    print(f"预测结果已保存到 {OUTPUT_FILE_PATH}")
    
    zip_file_path = OUTPUT_FILE_PATH.replace(".tsv", "") + f"__{scheme_type}.zip" 
    
    from pathlib import Path
    file_a = Path(OUTPUT_FILE_PATH)
    
    import zipfile
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(file_a, file_a.name)
    
    print(f"文件已成功压缩为: {zip_file_path}")

def clear_memory(device):
    """根据设备类型清理内存"""
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        # 对于MPS，尝试清理Metal缓存
        torch.mps.empty_cache()
        # print("MPS缓存已清理")

def main():
    # 优先使用MPS设备，其次是CUDA，最后是CPU
    if has_mps:
        device = torch.device('mps')
        print("使用MPS设备")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("使用CUDA设备")
        # 设置CUDA优化选项
        # 启用cuDNN基准测试以选择最佳卷积算法
        # torch.backends.cudnn.benchmark = True
        # 如果内存不足，可以考虑禁用这个选项
        torch.backends.cudnn.enabled = False
        
        # 设置PyTorch CUDA内存分配器以减少碎片化
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    else:
        device = torch.device('cpu')
        print("使用CPU设备")
    
    print(f"使用设备: {device}")
    
    # 对MPS设备进行内存管理设置
    if device.type == 'mps':
        # 设置内存管理优化
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 启用MPS回退到CPU的选项
    
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
    
    # 创建模型
    print("正在创建TransH模型...")
    model = TransH(mapper.entity_count, mapper.relation_count, EMBEDDING_DIM)
    model.to(device)
    
    # 检测OUTPUT_FILE_PATH的目录是否存在，如果不存在就创建
    output_dir = os.path.dirname(OUTPUT_FILE_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 检测模型保存目录是否存在，如果不存在就创建
    model_dir = os.path.dirname(TRAINED_MODEL_PATH)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"创建模型保存目录: {model_dir}")
    
    # MPS特定优化 - 预分配内存
    if device.type == 'mps':
        # 预热MPS设备
        print("预热MPS设备...")
        dummy_model = TransH(10, 5, EMBEDDING_DIM).to(device)
        dummy_h = torch.randint(0, 10, (10,), device=device)
        dummy_r = torch.randint(0, 5, (10,), device=device)
        dummy_t = torch.randint(0, 10, (10,), device=device)
        _ = dummy_model(dummy_h, dummy_r, dummy_t)
        del dummy_model, dummy_h, dummy_r, dummy_t
        clear_memory(device)
        print("MPS设备预热完成")
    
    # 检查是否强制重新训练或模型不存在
    if FORCE_RETRAIN or not os.path.exists(TRAINED_MODEL_PATH):
        # 训练模型
        print("开始训练模型...")
        train_model(model, train_dataset, mapper, device)
        
        # 保存训练好的模型
        print(f"保存训练好的模型到 {TRAINED_MODEL_PATH}")
        torch.save({
            'model_state_dict': model.state_dict(),
            'entity_count': mapper.entity_count,
            'relation_count': mapper.relation_count,
            'embedding_dim': EMBEDDING_DIM,
            'entity_to_id': mapper.entity_to_id,
            'relation_to_id': mapper.relation_to_id,
            'id_to_entity': mapper.id_to_entity,
            'id_to_relation': mapper.id_to_relation
        }, TRAINED_MODEL_PATH)
        print("模型保存完成")
    else:
        # 加载已训练的模型
        print(f"加载已训练的模型从 {TRAINED_MODEL_PATH}")
        checkpoint = torch.load(TRAINED_MODEL_PATH, map_location=device)
        
        # 确保模型结构与保存的模型一致
        if (mapper.entity_count != checkpoint['entity_count'] or 
            mapper.relation_count != checkpoint['relation_count'] or
            EMBEDDING_DIM != checkpoint['embedding_dim']):
            print("警告：模型参数不匹配，将使用新训练的模型")
            print("开始训练模型...")
            train_model(model, train_dataset, mapper, device)
            
            # 保存训练好的模型
            print(f"保存训练好的模型到 {TRAINED_MODEL_PATH}")
            torch.save({
                'model_state_dict': model.state_dict(),
                'entity_count': mapper.entity_count,
                'relation_count': mapper.relation_count,
                'embedding_dim': EMBEDDING_DIM,
                'entity_to_id': mapper.entity_to_id,
                'relation_to_id': mapper.relation_to_id,
                'id_to_entity': mapper.id_to_entity,
                'id_to_relation': mapper.id_to_relation
            }, TRAINED_MODEL_PATH)
            print("模型保存完成")
        else:
            # 加载模型参数
            model.load_state_dict(checkpoint['model_state_dict'])
            print("模型加载完成")
    
    # 在开发集上评估模型
    print("\n开始在开发集上评估模型性能...")
    evaluate(model, dev_dataset, mapper, device)
    
    # 预测尾实体
    # 可以通过max_head_entities参数控制处理的头实体个数，None表示处理全部
    predict_tail_entities(model, test_dataset, mapper, device, MAX_HEAD_ENTITIES)
    
    print("任务完成！")

if __name__ == "__main__":
    main()