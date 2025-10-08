import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, default_collate
import random
import time

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
LEARNING_RATE = 0.001  # 学习率（降低学习率以改善收敛）
MARGIN = 1.0  # Max-margin损失的margin值
BATCH_SIZE = 1024  # 训练批次大小
NEGATIVE_SAMPLES = 5  # 每个正样本对应的负样本数量
WEIGHT_DECAY = 1e-5  # 权重衰减（L2正则化）

MAX_LINES = None  # 限制训练数据的行数，None表示使用全部数据
epochs = 5  # 训练的epoch数量
max_head_entities = None  # 限制预测的头实体数量，None表示处理全部

# 学习率调度器参数
LR_DECAY_FACTOR = 0.5  # 学习率衰减因子
LR_DECAY_STEP = 50  # 每隔多少个epoch衰减一次学习率

# 数据加载类 - 优化版：预处理实体和关系ID以减少运行时计算
class KnowledgeGraphDataset(Dataset):
    def __init__(self, file_path, is_test=False, is_dev=False, max_lines=None, mapper=None):
        self.is_test = is_test
        self.is_dev = is_dev
        self.triples = []
        self.mapper = mapper
        
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                # 如果设置了最大行数限制且已达到限制，则停止读取
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
        return len(self.triples)  # 返回数据集大小
    
    def __getitem__(self, idx):
        return self.triples[idx]  # 返回索引对应的三元组

# 自定义的批处理函数，用于在DataLoader中并行转换实体和关系为ID
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

# TransE模型实现
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransE, self).__init__()
        # 初始化实体和关系嵌入层
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # 使用Xavier均匀分布初始化嵌入向量
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        
        # 归一化实体嵌入
        self.normalize_entities()
    
    def normalize_entities(self):
        # 对实体嵌入进行L2归一化
        with torch.no_grad():
            norms = torch.norm(self.entity_embeddings.weight, p=2, dim=1, keepdim=True)
            self.entity_embeddings.weight.data = self.entity_embeddings.weight.data / norms
    
    def forward(self, h, r, t, t_neg=None):
        # 获取嵌入向量 [batch_size, embedding_dim]
        h_emb = self.entity_embeddings(h)  # 头实体嵌入 [batch_size, embedding_dim]
        r_emb = self.relation_embeddings(r)  # 关系嵌入 [batch_size, embedding_dim]
        t_emb = self.entity_embeddings(t)  # 尾实体嵌入 [batch_size, embedding_dim]
        
        # 计算正样本得分：||h + r - t|| [batch_size]
        pos_score = torch.norm(h_emb + r_emb - t_emb, p=1, dim=1)
        
        if t_neg is not None:
            # 计算负样本得分
            t_neg_emb = self.entity_embeddings(t_neg)  # 负样本尾实体嵌入 [batch_size, negative_samples, embedding_dim]
            h_emb_expanded = h_emb.unsqueeze(1).expand(-1, NEGATIVE_SAMPLES, -1)  # 扩展头实体嵌入维度
            r_emb_expanded = r_emb.unsqueeze(1).expand(-1, NEGATIVE_SAMPLES, -1)  # 扩展关系嵌入维度
            
            # 计算负样本得分 [batch_size, negative_samples]
            neg_score = torch.norm(h_emb_expanded + r_emb_expanded - t_neg_emb, p=1, dim=2)
            return pos_score, neg_score  # 返回正样本得分和负样本得分
        
        return pos_score  # 只返回正样本得分

# 训练函数 - 优化版：提高GPU利用率
def train_model(model, train_dataset, mapper, device):
    # 创建自定义的collate_fn函数用于批量处理数据
    def custom_collate(batch):
        return collate_fn(batch, mapper)
    
    # 创建数据加载器 - 优化：使用pin_memory=True和更大的num_workers
    # pin_memory=True: 加速CPU到GPU的数据传输
    # num_workers: 并行数据加载的进程数
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             collate_fn=custom_collate, pin_memory=True, num_workers=8)
    
    # 定义优化器 - 改为AdamW优化器，通常比SGD更容易收敛
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_FACTOR)
    
    # 启用CUDA自动混合精度训练，减少内存使用并加速训练
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
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
            h_ids = h_ids.to(device, non_blocking=True)  # 非阻塞传输，提高效率
            r_ids = r_ids.to(device, non_blocking=True)
            t_ids = t_ids.to(device, non_blocking=True)
            
            # 使用向量化操作生成负样本，替代Python循环
            batch_size = h_ids.size(0)
            
            # 向量化生成负样本 - 优化1：大幅提高负样本生成效率
            # 创建负样本索引矩阵 [batch_size, negative_samples]
            t_neg_ids = torch.randint(0, mapper.entity_count, 
                                     (batch_size, NEGATIVE_SAMPLES), 
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
            
            # 使用自动混合精度进行前向传播
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # 前向传播
                pos_score, neg_score = model(h_ids, r_ids, t_ids, t_neg_ids)
                
                # 计算损失（基于margin的max-margin损失）
                # 将pos_score扩展为与neg_score相同的维度
                pos_score_expanded = pos_score.unsqueeze(1).expand_as(neg_score)
                
                # 计算每个样本的损失，并按批次大小进行平均，避免大批次带来的损失爆炸
                individual_loss = torch.relu(pos_score_expanded - neg_score + MARGIN)
                loss = torch.mean(individual_loss)  # 使用mean而不是sum，更稳定
                
                # 乘以批次大小以保持损失量级，便于与之前的训练比较
                loss = loss * batch_size
            
            # 反向传播和优化 - 使用scaler进行梯度缩放
            optimizer.zero_grad()
            scaler.scale(loss).backward()  # 缩放损失并反向传播
            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新缩放器
            
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

# 评测函数 - 优化版：向量化计算提升评估速度
def evaluate(model, dev_dataset, mapper, device, batch_size=128, max_entities_per_batch=10000):
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
            h_ids = torch.tensor([mapper.entity_to_id[h] for h in h_list], device=device)
            r_ids = torch.tensor([mapper.relation_to_id[r] for r in r_list], device=device)
            t_ids = torch.tensor([mapper.entity_to_id[t] for t in t_list], device=device)
            
            # 获取嵌入向量并计算h + r
            h_emb = model.entity_embeddings(h_ids)  # [current_batch_size, embedding_dim]
            r_emb = model.relation_embeddings(r_ids)  # [current_batch_size, embedding_dim]
            h_plus_r = h_emb + r_emb  # [current_batch_size, embedding_dim]
            
            # 分块处理实体，避免一次性加载所有实体导致显存溢出
            batch_scores = torch.zeros(current_batch_size, mapper.entity_count, device=device)
            
            # 分块计算得分 - 向量化实现
            for entity_start in range(0, mapper.entity_count, max_entities_per_batch):
                entity_end = min(entity_start + max_entities_per_batch, mapper.entity_count)
                entity_chunk_size = entity_end - entity_start
                
                # 获取当前块的实体嵌入
                entity_ids = torch.arange(entity_start, entity_end, device=device)
                entity_embeddings = model.entity_embeddings(entity_ids)  # [entity_chunk_size, embedding_dim]
                
                # 计算当前块的得分 - 向量化操作
                h_plus_r_expanded = h_plus_r.unsqueeze(1)  # [current_batch_size, 1, embedding_dim]
                chunk_scores = torch.norm(
                    h_plus_r_expanded - entity_embeddings.unsqueeze(0), 
                    p=1, 
                    dim=2
                )  # [current_batch_size, entity_chunk_size]
                
                # 存储到完整得分矩阵中
                batch_scores[:, entity_start:entity_end] = chunk_scores
                
                # 清理中间变量，释放显存
                del entity_embeddings, chunk_scores
            
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
            
            # 显示进度
            progress = min(total_count, total_dev_triples) / total_dev_triples * 100
            if total_count % 500 == 0 or total_count >= total_dev_triples:
                print(f"  评估进度: {progress:.1f}% ({min(total_count, total_dev_triples)}/{total_dev_triples})")
            
            # 清理显存
            del h_emb, r_emb, h_plus_r, batch_scores, correct_scores, num_better_entities, ranks
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

# 预测函数 - 优化版本（向量化计算，带进度条和显存优化）
def predict_tail_entities(model, test_dataset, mapper, device, max_head_entities=None, batch_size=128, max_entities_per_batch=10000):
    model.eval()
    results = []
    total_test_triples = len(test_dataset.triples)
    
    # 确定要处理的头实体数量
    process_count = total_test_triples if max_head_entities is None else min(max_head_entities, total_test_triples)
    print(f"\n开始预测尾实体...")
    print(f"将处理 {process_count} 个头实体/关系对")
    
    # 计算每个批次能处理的最大实体数量（避免显存溢出）
    # 根据显存大小动态调整，假设16GB显存，每个实体需要embedding_dim*4字节
    max_entities_per_batch = min(10000, mapper.entity_count)  # 限制最大实体数
    
    with torch.no_grad():
        # 创建批次进行处理，大幅提高GPU利用率
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
            h_emb = model.entity_embeddings(h_ids)  # [batch_size, embedding_dim]
            r_emb = model.relation_embeddings(r_ids)  # [batch_size, embedding_dim]
            
            # 计算h + r
            h_plus_r = h_emb + r_emb  # [batch_size, embedding_dim]
            
            # 分块处理实体，避免一次性加载所有实体导致显存溢出
            batch_scores = torch.zeros(len(batch_triples), mapper.entity_count, device=device)
            
            # 分块计算得分
            for entity_start in range(0, mapper.entity_count, max_entities_per_batch):
                entity_end = min(entity_start + max_entities_per_batch, mapper.entity_count)
                
                # 获取当前块的实体嵌入
                entity_ids = torch.arange(entity_start, entity_end, device=device)
                entity_embeddings = model.entity_embeddings(entity_ids)  # [chunk_size, embedding_dim]
                
                # 计算当前块的得分
                h_plus_r_expanded = h_plus_r.unsqueeze(1)  # [batch_size, 1, embedding_dim]
                chunk_scores = torch.norm(h_plus_r_expanded - entity_embeddings.unsqueeze(0), p=1, dim=2)  # [batch_size, chunk_size]
                
                # 存储到完整得分矩阵中
                batch_scores[:, entity_start:entity_end] = chunk_scores
                
                # 清理中间变量，释放显存
                del entity_embeddings, chunk_scores
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
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
        # 启用cuDNN基准测试以选择最佳卷积算法
        # torch.backends.cudnn.benchmark = True
        # 如果内存不足，可以考虑禁用这个选项
        torch.backends.cudnn.enabled = False
    
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
    
    # 创建模型
    print("正在创建TransE模型...")
    model = TransE(mapper.entity_count, mapper.relation_count, EMBEDDING_DIM)
    model.to(device)
    
    # 训练模型
    print("开始训练模型...")
    train_model(model, train_dataset, mapper, device)
    
    # 在开发集上评估模型
    print("\n训练完成，开始在开发集上评估模型性能...")
    evaluate(model, dev_dataset, mapper, device)
    
    # 预测尾实体
    # 可以通过max_head_entities参数控制处理的头实体个数，None表示处理全部
    predict_tail_entities(model, test_dataset, mapper, device, max_head_entities)
    
    print("任务完成！")

if __name__ == "__main__":
    main()