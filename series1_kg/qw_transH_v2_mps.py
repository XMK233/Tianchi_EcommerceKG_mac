import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
import time
from collections import defaultdict
import gc
import zipfile
from pathlib import Path

# 设置随机种子以确保结果可复现
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

scheme_type = "transH_mac_try"

# 数据文件路径 (请根据您的实际路径修改)
TRAIN_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/originalData/OpenBG500/OpenBG500_train.tsv"
TEST_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/originalData/OpenBG500/OpenBG500_test.tsv"
DEV_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/originalData/OpenBG500/OpenBG500_dev.tsv"
OUTPUT_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/preprocessedData/OpenBG500_test.tsv"
TRAINED_MODEL_PATH = f"/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/trained_model/trained_model__{scheme_type}.pth"
FORCE_RETRAIN = True

# 超参数设置
EMBEDDING_DIM = 100
MARGIN = 1.0
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 1
BATCH_SIZE = 256
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
            h_proj_expanded = h_proj.unsqueeze(1).expand(-1, neg_samples, -1).contiguous().view(-1, self.embedding_dim)
            r_expanded = r.unsqueeze(1).expand(-1, neg_samples, -1).contiguous().view(-1, self.embedding_dim)
            r_norm_expanded = r_norm.unsqueeze(1).expand(-1, neg_samples, -1).contiguous().view(-1, self.embedding_dim)
            
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
            self.entity_embeddings.weight.div_(torch.norm(self.entity_embeddings.weight, dim=1, keepdim=True))

def get_device():
    """获取可用设备，优先使用 MPS (Mac GPU)，然后是 CUDA，最后是 CPU。"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def train_model(model, train_dataset, mapper, device, epochs=EPOCHS, batch_size=BATCH_SIZE):
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=False,  # MPS 不需要 pin_memory
        num_workers=0  # MPS 上多进程可能有问题，设为0
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
            
            # 直接在collate_fn中转换为张量，这里只需要转移到设备上
            h_ids = torch.tensor([mapper.entity_to_id[h] for h in h_ids], device=device, dtype=torch.long)
            r_ids = torch.tensor([mapper.relation_to_id[r] for r in r_ids], device=device, dtype=torch.long)
            t_ids = torch.tensor([mapper.entity_to_id[t] for t in t_ids], device=device, dtype=torch.long)
            
            # 使用向量化操作生成负样本
            batch_size_current = h_ids.size(0)
            t_neg_ids = torch.randint(0, mapper.entity_count, 
                                     (batch_size_current, NEGATIVE_SAMPLES), 
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
            
            # 前向传播 (移除AMP，MPS不支持)
            pos_score, neg_score = model(h_ids, r_ids, t_ids, t_neg_ids)
            
            # 计算损失
            pos_score_expanded = pos_score.unsqueeze(1).expand_as(neg_score)
            individual_loss = torch.relu(pos_score_expanded - neg_score + MARGIN)
            loss = torch.mean(individual_loss) * batch_size_current
            
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

def evaluate(model, dev_dataset, mapper, device, batch_size=128, max_entities_per_batch=4000):
    """
    评估模型性能。通过增大批次和分块大小来加速推理。
    """
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
            h_emb = model.entity_embeddings(h_ids)
            r_emb = model.relation_embeddings(r_ids)
            r_norm = model.normal_vectors(r_ids)
            
            # 投影头实体
            h_proj = model._projection(h_emb, r_norm)
            
            # 计算h_proj + r
            h_plus_r = h_proj + r_emb  # [current_batch_size, embedding_dim]
            
            # 分块处理实体，避免显存溢出
            batch_scores = torch.zeros(current_batch_size, mapper.entity_count, device=device)
            
            # 分块计算得分 - 向量化实现
            for entity_start in range(0, mapper.entity_count, max_entities_per_batch):
                entity_end = min(entity_start + max_entities_per_batch, mapper.entity_count)
                entity_chunk_size = entity_end - entity_start
                
                # 获取当前块的实体嵌入
                entity_ids = torch.arange(entity_start, entity_end, device=device, dtype=torch.long)
                entity_embeddings = model.entity_embeddings(entity_ids)  # [entity_chunk_size, embedding_dim]
                
                # 投影实体嵌入到关系特定的超平面
                # 扩展以匹配批次大小
                entity_embeddings_expanded = entity_embeddings.unsqueeze(0).expand(current_batch_size, -1, -1)
                r_norm_expanded = r_norm.unsqueeze(1).expand(-1, entity_chunk_size, -1)
                entity_embeddings_proj = model._projection(entity_embeddings_expanded.reshape(-1, model.embedding_dim), 
                                                          r_norm_expanded.reshape(-1, model.embedding_dim))
                entity_embeddings_proj = entity_embeddings_proj.reshape(current_batch_size, entity_chunk_size, -1)
                
                # 计算当前块的得分
                h_plus_r_expanded = h_plus_r.unsqueeze(1)  # [current_batch_size, 1, embedding_dim]
                chunk_scores = torch.norm(
                    h_plus_r_expanded - entity_embeddings_proj, 
                    p=1, 
                    dim=2
                )  # [current_batch_size, entity_chunk_size]
                
                # 存储到完整得分矩阵中
                batch_scores[:, entity_start:entity_end] = chunk_scores
            
            # 查找每个样本中正确尾实体的得分
            correct_scores = torch.gather(batch_scores, 1, t_ids.view(-1, 1))  # [current_batch_size, 1]
            
            # 计算排名
            num_better_entities = (batch_scores < correct_scores).sum(dim=1)  # [current_batch_size]
            ranks = num_better_entities + 1  # [current_batch_size]
            
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

def predict_tail_entities(model, test_dataset, mapper, device, max_head_entities=None, batch_size=256, max_entities_per_batch=4000):
    """
    预测尾实体。通过增大批次和分块大小来加速推理。
    """
    model.eval()
    results = []
    total_test_triples = len(test_dataset.triples)
    
    # 确定要处理的头实体数量
    process_count = total_test_triples if max_head_entities is None else min(max_head_entities, total_test_triples)
    print(f"\n开始预测尾实体...")
    print(f"将处理 {process_count} 个头实体/关系对")
    
    with torch.no_grad():
        # 创建批次进行处理
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
            h_emb = model.entity_embeddings(h_ids)
            r_emb = model.relation_embeddings(r_ids)
            r_norm = model.normal_vectors(r_ids)
            
            # 投影头实体
            h_proj = model._projection(h_emb, r_norm)
            
            # 计算h_proj + r
            h_plus_r = h_proj + r_emb  # [batch_size, embedding_dim]
            
            # 分块处理实体
            batch_scores = torch.zeros(len(batch_triples), mapper.entity_count, device=device)
            
            # 分块计算得分
            for entity_start in range(0, mapper.entity_count, max_entities_per_batch):
                entity_end = min(entity_start + max_entities_per_batch, mapper.entity_count)
                
                # 获取当前块的实体嵌入
                entity_ids = torch.arange(entity_start, entity_end, device=device, dtype=torch.long)
                entity_embeddings = model.entity_embeddings(entity_ids)
                
                # 投影实体嵌入到关系特定的超平面
                entity_chunk_size = entity_end - entity_start
                batch_size_current = len(batch_triples)
                # 扩展以匹配批次大小
                entity_embeddings_expanded = entity_embeddings.unsqueeze(0).expand(batch_size_current, -1, -1)
                r_norm_expanded = r_norm.unsqueeze(1).expand(-1, entity_chunk_size, -1)
                entity_embeddings_proj = model._projection(entity_embeddings_expanded.reshape(-1, model.embedding_dim), 
                                                          r_norm_expanded.reshape(-1, model.embedding_dim))
                entity_embeddings_proj = entity_embeddings_proj.reshape(batch_size_current, entity_chunk_size, -1)
                
                # 计算当前块的得分
                h_plus_r_expanded = h_plus_r.unsqueeze(1)
                chunk_scores = torch.norm(h_plus_r_expanded - entity_embeddings_proj, p=1, dim=2)
                
                # 存储到完整得分矩阵中
                batch_scores[:, entity_start:entity_end] = chunk_scores
            
            # 获取每个头实体/关系对的top10尾实体
            _, top10_indices = torch.topk(-batch_scores, k=10, dim=1)
            
            # 处理批次结果
            for i in range(len(batch_triples)):
                h, r, _ = batch_triples[i]
                top10_t_ids = top10_indices[i].tolist()
                top10_entities = [mapper.id_to_entity[t_id] for t_id in top10_t_ids]
                result_line = [h, r] + top10_entities
                results.append('\t'.join(result_line))
            
            # 显示进度条
            progress = batch_end / process_count * 100
            bar_length = 50
            filled_length = int(bar_length * progress // 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"\r  预测进度: |{bar}| {progress:.1f}% ({batch_end}/{process_count})", end='')
    
    print()  # 确保进度条完成后换行
    
    # 写入结果文件
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')
    
    print(f"预测结果已保存到 {OUTPUT_FILE_PATH}")
    
    zip_file_path = OUTPUT_FILE_PATH.replace(".tsv", "") + f"__{scheme_type}.zip" 
    file_a = Path(OUTPUT_FILE_PATH)
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(file_a, file_a.name)
    
    print(f"文件已成功压缩为: {zip_file_path}")

def main():
    # 获取设备 (优先使用 MPS)
    device = get_device()
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("正在加载数据集...")
    train_dataset = KnowledgeGraphDataset(TRAIN_FILE_PATH, max_lines=MAX_LINES)
    test_dataset = KnowledgeGraphDataset(TEST_FILE_PATH, is_test=True)
    dev_dataset = KnowledgeGraphDataset(DEV_FILE_PATH, is_dev=True)
    
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
    
    # 检测输出和模型保存目录
    output_dir = os.path.dirname(OUTPUT_FILE_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    model_dir = os.path.dirname(TRAINED_MODEL_PATH)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"创建模型保存目录: {model_dir}")
    
    # 训练或加载模型
    if FORCE_RETRAIN or not os.path.exists(TRAINED_MODEL_PATH):
        print("开始训练模型...")
        train_model(model, train_dataset, mapper, device)
        
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
        print(f"加载已训练的模型从 {TRAINED_MODEL_PATH}")
        checkpoint = torch.load(TRAINED_MODEL_PATH, map_location=device)
        
        if (mapper.entity_count != checkpoint['entity_count'] or 
            mapper.relation_count != checkpoint['relation_count'] or
            EMBEDDING_DIM != checkpoint['embedding_dim']):
            print("警告：模型参数不匹配，将使用新训练的模型")
            print("开始训练模型...")
            train_model(model, train_dataset, mapper, device)
            
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
            model.load_state_dict(checkpoint['model_state_dict'])
            print("模型加载完成")
    
    # 在开发集上评估模型
    print("\n开始在开发集上评估模型性能...")
    evaluate(model, dev_dataset, mapper, device)
    
    # 预测尾实体
    predict_tail_entities(model, test_dataset, mapper, device, MAX_HEAD_ENTITIES)
    
    print("任务完成！")

if __name__ == "__main__":
    main()