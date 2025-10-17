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
print("使用CPU模式运行")

# 数据路径
scheme_type = 'th_cpu_try'
# 数据路径 - 保持原有路径，确保与用户环境匹配
TRAIN_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/originalData/OpenBG500/OpenBG500_train.tsv"
TEST_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/originalData/OpenBG500/OpenBG500_test.tsv"
DEV_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/originalData/OpenBG500/OpenBG500_dev.tsv" # 开发集路径
OUTPUT_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/preprocessedData/OpenBG500_test.tsv"

# 更新模型保存路径到用户的本地路径
FORCE_RETRAIN = True
TRAINED_MODEL_PATH = f"/Users/minkexiu/Documents/GitHub/Tianchi_EcommerceKG_mac/trained_models/trained_model__{scheme_type}.pth"

# 超参数设置 - 针对CPU优化的参数配置
EMBEDDING_DIM = 100
MARGIN = 1.0
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 1
# CPU版本使用较小的批次大小以避免内存问题
BATCH_SIZE = 64  # CPU版本使用较小批次
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
        norm_normals = nn.functional.normalize(normals, p=2, dim=1)  # [batch_size, embedding_dim]
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
            norms = torch.norm(self.entity_embeddings.weight, dim=1, keepdim=True)
            # 防止除以零
            norms = torch.maximum(norms, torch.tensor(1e-12))
            self.entity_embeddings.weight.div_(norms)

def train_model(model, train_dataset, mapper, device, epochs=EPOCHS, batch_size=BATCH_SIZE):
    # CPU模式下的数据加载器参数
    pin_memory = False  # CPU不需要pin_memory
    num_workers = 0  # CPU模式下使用0个worker避免潜在问题
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers
    )
    
    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_FACTOR)
    
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
            
            # 转换为ID和张量
            h_ids = torch.tensor([mapper.entity_to_id[h] for h in h_ids], device=device, dtype=torch.long)
            r_ids = torch.tensor([mapper.relation_to_id[r] for r in r_ids], device=device, dtype=torch.long)
            t_ids = torch.tensor([mapper.entity_to_id[t] for t in t_ids], device=device, dtype=torch.long)
            
            # 生成负样本
            batch_size_current = h_ids.size(0)
            
            # 创建负样本索引矩阵 [batch_size, negative_samples]
            t_neg_ids = torch.randint(0, mapper.entity_count, 
                                     (batch_size_current, NEGATIVE_SAMPLES), 
                                     device=device, dtype=torch.long)
            
            # 确保负样本不等于正样本
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
            
            # 前向传播
            pos_score, neg_score = model(h_ids, r_ids, t_ids, t_neg_ids)
            
            # 计算损失（基于margin的max-margin损失）
            pos_score_expanded = pos_score.unsqueeze(1).expand_as(neg_score)
            
            # 计算每个样本的损失，并按批次大小进行平均
            individual_loss = torch.relu(pos_score_expanded - neg_score + MARGIN)
            loss = torch.mean(individual_loss)
            
            # 乘以批次大小以保持损失量级
            loss = loss * batch_size_current
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 归一化实体嵌入
            model.normalize_entities()
            
            total_loss += loss.item()
            
            # 每个batch打印一次进度
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
            
            # 分块处理实体，避免一次性加载所有实体导致内存溢出
            batch_scores = torch.zeros(current_batch_size, mapper.entity_count, device=device)
            
            # 分块计算得分
            for entity_start in range(0, mapper.entity_count, max_entities_per_batch):
                entity_end = min(entity_start + max_entities_per_batch, mapper.entity_count)
                entity_chunk_size = entity_end - entity_start
                
                # 获取当前块的实体嵌入
                entity_ids = torch.arange(entity_start, entity_end, device=device, dtype=torch.long)
                entity_embeddings = model.entity_embeddings(entity_ids)  # [entity_chunk_size, embedding_dim]
                
                # 投影实体嵌入到关系特定的超平面
                entity_embeddings_expanded = entity_embeddings.repeat(current_batch_size, 1)
                r_norm_expanded = r_norm.unsqueeze(1).expand(-1, entity_chunk_size, -1).contiguous().view(-1, model.embedding_dim)
                entity_embeddings_proj = model._projection(entity_embeddings_expanded, r_norm_expanded)
                
                # 计算当前块的得分
                h_plus_r_expanded = h_plus_r.unsqueeze(1)  # [current_batch_size, 1, embedding_dim]
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
                gc.collect()
            
            # 查找每个样本中正确尾实体的得分
            correct_scores = torch.gather(batch_scores, 1, t_ids.view(-1, 1))  # [current_batch_size, 1]
            
            # 计算排名（得分越小，排名越高）
            # 统计得分小于正确得分的实体数量（即排名）
            worse_scores = batch_scores < correct_scores  # [current_batch_size, entity_count]
            ranks = worse_scores.sum(dim=1) + 1  # +1是因为排名从1开始
            
            # 更新评估指标
            hits_at_1 += (ranks <= 1).sum().item()
            hits_at_3 += (ranks <= 3).sum().item()
            hits_at_10 += (ranks <= 10).sum().item()
            mrr_score += (1.0 / ranks.float()).sum().item()
            total_count += current_batch_size
            
            # 打印进度
            progress = total_count / total_dev_triples * 100
            print(f"  评估进度: {progress:.1f}% - Hits@1: {hits_at_1/total_count:.4f} - Hits@3: {hits_at_3/total_count:.4f} - Hits@10: {hits_at_10/total_count:.4f} - MRR: {mrr_score/total_count:.4f}", end='\r')
    
    # 计算最终指标
    hits_at_1 /= total_count
    hits_at_3 /= total_count
    hits_at_10 /= total_count
    mrr_score /= total_count
    
    print(f"\n评估完成!")
    print(f"Hits@1: {hits_at_1:.4f}")
    print(f"Hits@3: {hits_at_3:.4f}")
    print(f"Hits@10: {hits_at_10:.4f}")
    print(f"MRR: {mrr_score:.4f}")
    
    return {
        'hits_at_1': hits_at_1,
        'hits_at_3': hits_at_3,
        'hits_at_10': hits_at_10,
        'mrr': mrr_score
    }

def predict_tail_entities(model, test_dataset, mapper, device, output_file=None, max_head_entities=None):
    model.eval()
    
    total_test_triples = len(test_dataset.triples)
    
    # 确定要处理的头实体数量
    process_count = total_test_triples if max_head_entities is None else min(max_head_entities, total_test_triples)
    print(f"\n开始预测尾实体...")
    print(f"将处理 {process_count} 个查询三元组")
    
    # 存储预测结果
    results = []
    
    with torch.no_grad():
        # 按批次处理测试数据
        batch_size = 64  # CPU模式下使用较小的批次
        for batch_start in range(0, process_count, batch_size):
            batch_end = min(batch_start + batch_size, process_count)
            batch_triples = test_dataset.triples[batch_start:batch_end]
            current_batch_size = len(batch_triples)
            
            # 获取批次中的头实体和关系
            h_list = [h for h, r, _ in batch_triples]
            r_list = [r for h, r, _ in batch_triples]
            
            # 转换为ID和张量
            h_ids = torch.tensor([mapper.entity_to_id[h] for h in h_list], device=device, dtype=torch.long)
            r_ids = torch.tensor([mapper.relation_to_id[r] for r in r_list], device=device, dtype=torch.long)
            
            # 获取嵌入向量并计算h + r
            h_emb = model.entity_embeddings(h_ids)  # [current_batch_size, embedding_dim]
            r_emb = model.relation_embeddings(r_ids)  # [current_batch_size, embedding_dim]
            r_norm = model.normal_vectors(r_ids)  # [current_batch_size, embedding_dim]
            
            # 投影头实体
            h_proj = model._projection(h_emb, r_norm)  # [current_batch_size, embedding_dim]
            
            # 计算h_proj + r
            h_plus_r = h_proj + r_emb  # [current_batch_size, embedding_dim]
            
            # 分块处理实体，避免一次性加载所有实体导致内存溢出
            batch_scores = torch.zeros(current_batch_size, mapper.entity_count, device=device)
            
            # 分块计算得分
            max_entities_per_batch = 1000  # CPU模式下使用较小的块
            for entity_start in range(0, mapper.entity_count, max_entities_per_batch):
                entity_end = min(entity_start + max_entities_per_batch, mapper.entity_count)
                entity_chunk_size = entity_end - entity_start
                
                # 获取当前块的实体嵌入
                entity_ids = torch.arange(entity_start, entity_end, device=device, dtype=torch.long)
                entity_embeddings = model.entity_embeddings(entity_ids)  # [entity_chunk_size, embedding_dim]
                
                # 投影实体嵌入到关系特定的超平面
                entity_embeddings_expanded = entity_embeddings.repeat(current_batch_size, 1)
                r_norm_expanded = r_norm.unsqueeze(1).expand(-1, entity_chunk_size, -1).contiguous().view(-1, model.embedding_dim)
                entity_embeddings_proj = model._projection(entity_embeddings_expanded, r_norm_expanded)
                
                # 计算当前块的得分
                h_plus_r_expanded = h_plus_r.unsqueeze(1)  # [current_batch_size, 1, embedding_dim]
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
                gc.collect()
            
            # 找出得分最小的前10个实体（因为我们使用L1距离作为得分）
            top10_scores, top10_indices = torch.topk(batch_scores, 10, dim=1, largest=False)
            
            # 处理批次结果
            for i in range(current_batch_size):
                h = h_list[i]
                r = r_list[i]
                top10_entities = [mapper.id_to_entity[idx.item()] for idx in top10_indices[i]]
                results.append((h, r, top10_entities))
            
            # 打印进度
            progress = batch_end / process_count * 100
            print(f"  预测进度: {progress:.1f}%", end='\r')
    
    # 写入结果到文件
    if output_file:
        print(f"\n\n正在写入预测结果到 {output_file}...")
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for h, r, top10_entities in results:
                # 写入格式: h\tr\tt1\tt2\t...\tt10
                line = f"{h}\t{r}\t" + "\t".join(top10_entities) + "\n"
                f.write(line)
        print(f"预测结果已写入 {output_file}")
    
    return results

def clear_memory():
    """清理CPU内存"""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

def main():
    # 选择设备 - CPU模式
    device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("\n加载数据集...")
    train_dataset = KnowledgeGraphDataset(TRAIN_FILE_PATH, max_lines=MAX_LINES)
    test_dataset = KnowledgeGraphDataset(TEST_FILE_PATH, is_test=True, max_lines=MAX_LINES)
    dev_dataset = KnowledgeGraphDataset(DEV_FILE_PATH, is_dev=True, max_lines=MAX_LINES)
    
    # 打印数据集大小
    print(f"\n数据集统计:")
    print(f"训练集大小: {len(train_dataset.triples)}")
    print(f"测试集大小: {len(test_dataset.triples)}")
    print(f"开发集大小: {len(dev_dataset.triples)}")
    
    # 构建实体和关系映射
    print("\n构建实体和关系映射...")
    mapper = EntityRelationMapper()
    mapper.build_mappings(train_dataset, test_dataset, dev_dataset)
    
    # 打印实体和关系数量
    print(f"实体数量: {mapper.entity_count}")
    print(f"关系数量: {mapper.relation_count}")
    
    # 创建模型
    print("\n创建TransH模型...")
    model = TransH(
        num_entities=mapper.entity_count,
        num_relations=mapper.relation_count,
        embedding_dim=EMBEDDING_DIM
    ).to(device)
    
    # 创建模型保存目录
    model_dir = os.path.dirname(TRAINED_MODEL_PATH)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 训练或加载模型
    if FORCE_RETRAIN or not os.path.exists(TRAINED_MODEL_PATH):
        print("\n开始训练模型...")
        train_model(model, train_dataset, mapper, device)
        
        # 保存模型
        print(f"\n保存模型到 {TRAINED_MODEL_PATH}...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'entity_to_id': mapper.entity_to_id,
            'relation_to_id': mapper.relation_to_id,
            'id_to_entity': mapper.id_to_entity,
            'id_to_relation': mapper.id_to_relation
        }, TRAINED_MODEL_PATH)
        print("模型保存成功!")
    else:
        print(f"\n加载预训练模型从 {TRAINED_MODEL_PATH}...")
        checkpoint = torch.load(TRAINED_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # 加载映射
        mapper.entity_to_id = checkpoint['entity_to_id']
        mapper.relation_to_id = checkpoint['relation_to_id']
        mapper.id_to_entity = checkpoint['id_to_entity']
        mapper.id_to_relation = checkpoint['id_to_relation']
        mapper.entity_count = len(mapper.entity_to_id)
        mapper.relation_count = len(mapper.relation_to_id)
        print("模型加载成功!")
    
    # 在开发集上评估模型
    print("\n在开发集上评估模型...")
    evaluate(model, dev_dataset, mapper, device)
    
    # 预测尾实体
    print("\n预测尾实体...")
    predict_tail_entities(model, test_dataset, mapper, device, OUTPUT_FILE_PATH, MAX_HEAD_ENTITIES)
    
    print("\n所有任务完成!")

if __name__ == "__main__":
    main()