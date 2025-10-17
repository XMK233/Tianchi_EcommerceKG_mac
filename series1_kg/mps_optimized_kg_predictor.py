import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import time
import gc
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# 设置随机种子，保证结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 设置MPS随机种子 - 使用正确的API路径
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.backends.mps.manual_seed(seed)

set_seed(42)

# 数据路径配置
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/originalData/OpenBG500/OpenBG500_train.tsv"
TEST_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/originalData/OpenBG500/OpenBG500_test.tsv"
DEV_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/originalData/OpenBG500/OpenBG500_dev.tsv" # 开发集路径
OUTPUT_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/preprocessedData/OpenBG500_test.tsv"

# 确保输出目录存在
os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)

# MPS优化配置
def setup_mps_optimization():
    """为MPS设备设置优化参数"""
    # 启用PyTorch的JIT编译以优化MPS操作
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)
    
    # 对于MPS设备，设置合适的内存分配策略 - 使用正确的API路径
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # 启用内存池管理
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.95'

# 超参数设置 - 针对M3 Pro优化
EMBEDDING_DIM = 200  # 嵌入维度
LEARNING_RATE = 0.001  # 学习率
BATCH_SIZE = 2048  # 更大的批次大小以利用M3 Pro的并行计算能力
NUM_EPOCHS = 20  # 训练轮数
MARGIN = 1.0  # 间隔值
NEGATIVE_SAMPLES = 1  # 负样本数量
MAX_LINES = None  # 限制加载的行数，None表示加载全部
MAX_HEAD_ENTITIES = None  # 限制预测的头实体数量，None表示全部处理

# CPU核心数，用于并行处理
NUM_WORKERS = min(mp.cpu_count(), 8)  # 使用8个或系统可用CPU核心数

# 数据加载器的collate函数 - 移到顶层以解决pickle问题
def custom_collate(batch):
    # 分离正样本和负样本
    positive_triples = []
    negative_triples = []
    for item in batch:
        if isinstance(item, tuple) and len(item) == 3:
            # 正样本
            positive_triples.append(item)
        elif isinstance(item, list) and len(item) > 0:
            # 负样本列表
            negative_triples.extend(item)
    return positive_triples, negative_triples

# 并行数据加载和预处理
class ParallelKnowledgeGraphDataset(Dataset):
    def __init__(self, file_path, is_test=False, is_dev=False, max_lines=None):
        self.triples = []
        self.is_test = is_test
        self.is_dev = is_dev
        
        # 使用多线程加载数据
        self._load_data_parallel(file_path, max_lines)
    
    def _load_data_parallel(self, file_path, max_lines):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if max_lines is not None:
                    lines = lines[:max_lines]
                
            # 使用线程池并行处理数据行
            chunk_size = max(1000, len(lines) // NUM_WORKERS)
            
            def process_chunk(chunk):
                results = []
                for line in chunk:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        h, r, t = parts[0], parts[1], parts[2]
                        results.append((h, r, t))
                return results
            
            chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
            
            # 使用线程池并行处理
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                processed_chunks = list(executor.map(process_chunk, chunks))
            
            # 合并结果
            for chunk in processed_chunks:
                self.triples.extend(chunk)
                
        except Exception as e:
            print(f"加载数据集时出错: {e}")
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        return self.triples[idx]

# 实体和关系映射管理器
class EntityRelationMapper:
    def __init__(self):
        self.entity_to_id = {}
        self.id_to_entity = []
        self.relation_to_id = {}
        self.id_to_relation = []
        self.entity_count = 0
        self.relation_count = 0
        
    # 并行构建映射
    def build_mappings_parallel(self, train_dataset, test_dataset, dev_dataset):
        # 收集所有实体和关系
        all_entities = set()
        all_relations = set()
        
        # 创建数据集列表
        datasets = [train_dataset, test_dataset, dev_dataset]
        dataset_types = ['train', 'test', 'dev']
        
        # 定义处理单个数据集的函数
        def process_dataset(dataset, dataset_type):
            entities = set()
            relations = set()
            
            if dataset_type == 'train' or dataset_type == 'dev':
                for h, r, t in dataset.triples:
                    entities.add(h)
                    entities.add(t)
                    relations.add(r)
            else:  # test dataset
                for h, r, _ in dataset.triples:
                    entities.add(h)
                    relations.add(r)
            
            return entities, relations
        
        # 使用线程池并行处理数据集
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = list(executor.map(process_dataset, datasets, dataset_types))
        
        # 合并结果
        for entities, relations in results:
            all_entities.update(entities)
            all_relations.update(relations)
        
        # 构建映射
        self.entity_to_id = {entity: i for i, entity in enumerate(sorted(all_entities))}
        self.id_to_entity = sorted(all_entities)
        self.relation_to_id = {relation: i for i, relation in enumerate(sorted(all_relations))}
        self.id_to_relation = sorted(all_relations)
        
        # 更新计数
        self.entity_count = len(self.id_to_entity)
        self.relation_count = len(self.id_to_relation)

# TransE模型 - MPS优化版
class TransE(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, device):
        super(TransE, self).__init__()
        self.entity_embeddings = torch.nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(num_relations, embedding_dim)
        self.embedding_dim = embedding_dim
        self.device = device
        
        # 初始化嵌入向量
        self.initialize_embeddings()
        
        # 使用更高效的内存格式
        self.entity_embeddings.weight.data = self.entity_embeddings.weight.data.contiguous()
        self.relation_embeddings.weight.data = self.relation_embeddings.weight.data.contiguous()
    
    def initialize_embeddings(self):
        # 使用均匀分布初始化
        torch.nn.init.uniform_(self.entity_embeddings.weight.data, -6 / np.sqrt(self.embedding_dim), 6 / np.sqrt(self.embedding_dim))
        torch.nn.init.uniform_(self.relation_embeddings.weight.data, -6 / np.sqrt(self.embedding_dim), 6 / np.sqrt(self.embedding_dim))
        
        # 归一化实体嵌入
        self.normalize_entity_embeddings()
    
    def normalize_entity_embeddings(self):
        # 对实体嵌入进行L2归一化
        with torch.no_grad():
            norm = torch.norm(self.entity_embeddings.weight, p=2, dim=1, keepdim=True)
            self.entity_embeddings.weight.data = self.entity_embeddings.weight.data / norm
    
    def forward(self, positive_triples, negative_triples, mapper):
        # 预分配张量以减少内存碎片
        h_pos = torch.tensor([mapper.entity_to_id[h] for h, _, _ in positive_triples], device=self.device)
        r_pos = torch.tensor([mapper.relation_to_id[r] for _, r, _ in positive_triples], device=self.device)
        t_pos = torch.tensor([mapper.entity_to_id[t] for _, _, t in positive_triples], device=self.device)
        
        # 获取嵌入向量
        h_emb_pos = self.entity_embeddings(h_pos)
        r_emb_pos = self.relation_embeddings(r_pos)
        t_emb_pos = self.entity_embeddings(t_pos)
        
        # 计算正样本得分 (L1距离) - 使用MPS优化的向量化操作
        pos_scores = torch.norm(h_emb_pos + r_emb_pos - t_emb_pos, p=1, dim=1)
        
        # 处理负样本
        h_neg = torch.tensor([mapper.entity_to_id[h] for h, _, _ in negative_triples], device=self.device)
        r_neg = torch.tensor([mapper.relation_to_id[r] for _, r, _ in negative_triples], device=self.device)
        t_neg = torch.tensor([mapper.entity_to_id[t] for _, _, t in negative_triples], device=self.device)
        
        # 获取嵌入向量
        h_emb_neg = self.entity_embeddings(h_neg)
        r_emb_neg = self.relation_embeddings(r_neg)
        t_emb_neg = self.entity_embeddings(t_neg)
        
        # 计算负样本得分 (L1距离) - 使用MPS优化的向量化操作
        neg_scores = torch.norm(h_emb_neg + r_emb_neg - t_emb_neg, p=1, dim=1)
        
        # 计算损失 (基于间隔的损失函数)
        # 确保正样本得分尽可能小，负样本得分尽可能大
        loss = torch.mean(torch.relu(pos_scores - neg_scores + MARGIN))
        
        # 定期归一化实体嵌入
        if self.training:
            self.normalize_entity_embeddings()
        
        return loss

# 并行负采样函数
def generate_negative_samples_parallel(triple, mapper, num_negatives=1, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    h, r, t = triple
    negative_samples = []
    
    for _ in range(num_negatives):
        # 随机选择替换头实体或尾实体
        if random.random() < 0.5:
            # 替换头实体
            neg_h = random.choice(mapper.id_to_entity)
            while neg_h == h:  # 确保生成的负样本不同于正样本
                neg_h = random.choice(mapper.id_to_entity)
            negative_samples.append((neg_h, r, t))
        else:
            # 替换尾实体
            neg_t = random.choice(mapper.id_to_entity)
            while neg_t == t:  # 确保生成的负样本不同于正样本
                neg_t = random.choice(mapper.id_to_entity)
            negative_samples.append((h, r, neg_t))
    
    return negative_samples

# 训练函数 - MPS优化版
def train_model(model, train_dataset, mapper, device):
    # 定义优化器 - 使用更适合MPS的参数
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
    
    # 启用梯度累积以提高吞吐量
    gradient_accumulation_steps = 2  # 根据显存大小调整
    
    # 创建批次索引以避免重复计算
    total_batches = (len(train_dataset) + BATCH_SIZE - 1) // BATCH_SIZE
    batch_indices = list(range(0, len(train_dataset), BATCH_SIZE))
    
    # 训练循环
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        # 手动分批次处理数据，避免使用多进程DataLoader
        for batch_idx, batch_start in enumerate(batch_indices):
            batch_end = min(batch_start + BATCH_SIZE, len(train_dataset))
            batch_triples = train_dataset.triples[batch_start:batch_end]
            
            # 生成负样本 - 使用线程池并行生成
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                # 为每个triple设置不同的随机种子以确保结果可复现
                seeds = [batch_idx * len(batch_triples) + i for i in range(len(batch_triples))]
                gen_func = partial(generate_negative_samples_parallel, mapper=mapper, num_negatives=NEGATIVE_SAMPLES)
                
                # 并行生成负样本
                negative_samples_list = list(executor.map(
                    lambda triple_seed: gen_func(triple=triple_seed[0], seed=triple_seed[1]),
                    zip(batch_triples, seeds)
                ))
            
            # 收集正负样本
            positive_triples = batch_triples
            negative_triples = [sample for sublist in negative_samples_list for sample in sublist]
            
            # 前向传播
            loss = model(positive_triples, negative_triples, mapper)
            
            # 梯度累积
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # 梯度裁剪以防止梯度爆炸
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            # 显示批次进度
            if (batch_end) % (BATCH_SIZE * 10) == 0:
                progress = batch_end / len(train_dataset) * 100
                print(f"  批次进度: {progress:.1f}% ({batch_end}/{len(train_dataset)})", end='\r')
        
        # 如果还有未更新的梯度，进行最后一次更新
        if total_batches % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # MPS特定的内存清理
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.backends.mps.empty_cache()
        gc.collect()
        
        # 打印轮次信息
        epoch_time = time.time() - start_time
        avg_loss = total_loss / total_batches
        print(f"\n轮次 {epoch+1}/{NUM_EPOCHS} - 平均损失: {avg_loss:.4f} - 耗时: {epoch_time:.2f}秒")
    
    return model

# 评估函数 - MPS优化版本
def evaluate(model, dev_dataset, mapper, device, batch_size=256, max_entities_per_batch=20000):
    model.eval()
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_10 = 0
    mrr_score = 0
    total_count = 0
    total_dev_triples = len(dev_dataset.triples)
    
    print(f"将评估 {total_dev_triples} 个三元组")
    
    with torch.no_grad():
        # 创建批次进行处理 - 更大的批次大小以利用MPS并行性
        for batch_start in range(0, total_dev_triples, batch_size):
            batch_end = min(batch_start + batch_size, total_dev_triples)
            batch_triples = dev_dataset.triples[batch_start:batch_end]
            current_batch_size = len(batch_triples)
            
            # 获取批次中的头实体、关系和尾实体
            h_list = [h for h, _, _ in batch_triples]
            r_list = [r for _, r, _ in batch_triples]
            t_list = [t for _, _, t in batch_triples]
            
            # 转换为ID和张量并直接发送到设备
            h_ids = torch.tensor([mapper.entity_to_id[h] for h in h_list], device=device)
            r_ids = torch.tensor([mapper.relation_to_id[r] for r in r_list], device=device)
            t_ids = torch.tensor([mapper.entity_to_id[t] for t in t_list], device=device)
            
            # 获取嵌入向量并计算h + r - 使用MPS优化的向量化操作
            h_emb = model.entity_embeddings(h_ids)  # [current_batch_size, embedding_dim]
            r_emb = model.relation_embeddings(r_ids)  # [current_batch_size, embedding_dim]
            h_plus_r = h_emb + r_emb  # [current_batch_size, embedding_dim]
            
            # 使用预分配的固定大小张量以减少MPS上的内存碎片
            batch_scores = torch.zeros((current_batch_size, mapper.entity_count), device=device)
            
            # 分块处理实体，增加块大小以提高MPS利用率
            for entity_start in range(0, mapper.entity_count, max_entities_per_batch):
                entity_end = min(entity_start + max_entities_per_batch, mapper.entity_count)
                
                # 获取当前块的实体嵌入
                entity_ids = torch.arange(entity_start, entity_end, device=device)
                entity_embeddings = model.entity_embeddings(entity_ids)  # [chunk_size, embedding_dim]
                
                # 计算当前块的得分 - 使用MPS优化的向量化操作
                h_plus_r_expanded = h_plus_r.unsqueeze(1)  # [current_batch_size, 1, embedding_dim]
                entity_embeddings_expanded = entity_embeddings.unsqueeze(0)  # [1, chunk_size, embedding_dim]
                
                # 使用广播操作计算距离
                chunk_scores = torch.norm(h_plus_r_expanded - entity_embeddings_expanded, p=1, dim=2)  # [current_batch_size, chunk_size]
                
                # 存储到完整得分矩阵中
                batch_scores[:, entity_start:entity_end] = chunk_scores
                
                # 清理中间变量，释放内存
                del entity_embeddings, chunk_scores, entity_embeddings_expanded
            
            # 查找每个样本中正确尾实体的得分
            correct_scores = torch.gather(batch_scores, 1, t_ids.view(-1, 1))  # [current_batch_size, 1]
            
            # 计算排名（得分越低排名越高） - 使用向量化比较以提高MPS效率
            num_better_entities = (batch_scores < correct_scores).sum(dim=1)  # [current_batch_size]
            ranks = num_better_entities + 1  # [current_batch_size] - 排名从1开始
            
            # 使用位运算和向量化操作计算指标
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
            
            # MPS特定的内存清理 - 使用正确的API路径
            del h_emb, r_emb, h_plus_r, batch_scores, correct_scores, num_better_entities, ranks
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.backends.mps.empty_cache()
            gc.collect()
    
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

# 预测函数 - MPS优化版本
def predict_tail_entities(model, test_dataset, mapper, device, max_head_entities=None, batch_size=256, max_entities_per_batch=20000):
    model.eval()
    results = []
    total_test_triples = len(test_dataset.triples)
    
    # 确定要处理的头实体数量
    process_count = total_test_triples if max_head_entities is None else min(max_head_entities, total_test_triples)
    print(f"\n开始预测尾实体...")
    print(f"将处理 {process_count} 个头实体/关系对")
    
    # 增加最大实体数每批次以利用MPS并行性
    max_entities_per_batch = min(max_entities_per_batch, mapper.entity_count)
    
    # 创建结果缓冲区以减少内存分配
    results_buffer = []
    buffer_size = 1000  # 每1000个结果写入一次文件
    
    with torch.no_grad():
        # 创建批次进行处理，使用更大的批次大小
        for batch_start in range(0, process_count, batch_size):
            batch_end = min(batch_start + batch_size, process_count)
            batch_triples = test_dataset.triples[batch_start:batch_end]
            current_batch_size = len(batch_triples)
            
            # 获取批次中的头实体和关系
            h_list = [h for h, _, _ in batch_triples]
            r_list = [r for _, r, _ in batch_triples]
            
            # 转换为ID和张量并直接发送到设备
            h_ids = torch.tensor([mapper.entity_to_id[h] for h in h_list], device=device)
            r_ids = torch.tensor([mapper.relation_to_id[r] for r in r_list], device=device)
            
            # 获取嵌入向量
            h_emb = model.entity_embeddings(h_ids)  # [batch_size, embedding_dim]
            r_emb = model.relation_embeddings(r_ids)  # [batch_size, embedding_dim]
            
            # 计算h + r
            h_plus_r = h_emb + r_emb  # [batch_size, embedding_dim]
            
            # 预分配固定大小的得分张量
            batch_scores = torch.zeros((current_batch_size, mapper.entity_count), device=device)
            
            # 分块计算得分 - 增加块大小以提高MPS利用率
            for entity_start in range(0, mapper.entity_count, max_entities_per_batch):
                entity_end = min(entity_start + max_entities_per_batch, mapper.entity_count)
                
                # 获取当前块的实体嵌入
                entity_ids = torch.arange(entity_start, entity_end, device=device)
                entity_embeddings = model.entity_embeddings(entity_ids)  # [chunk_size, embedding_dim]
                
                # 计算当前块的得分 - 使用MPS优化的向量化操作
                h_plus_r_expanded = h_plus_r.unsqueeze(1)  # [batch_size, 1, embedding_dim]
                entity_embeddings_expanded = entity_embeddings.unsqueeze(0)  # [1, chunk_size, embedding_dim]
                
                # 使用广播操作计算距离
                chunk_scores = torch.norm(h_plus_r_expanded - entity_embeddings_expanded, p=1, dim=2)  # [batch_size, chunk_size]
                
                # 存储到完整得分矩阵中
                batch_scores[:, entity_start:entity_end] = chunk_scores
                
                # 清理中间变量，释放内存
                del entity_embeddings, chunk_scores, entity_embeddings_expanded
            
            # 获取每个头实体/关系对的top10尾实体 - 使用MPS优化的topk操作
            _, top10_indices = torch.topk(-batch_scores, k=10, dim=1)  # 使用负数以获取最小的10个值
            
            # 处理批次结果
            for i in range(current_batch_size):
                h, r, _ = batch_triples[i]
                top10_t_ids = top10_indices[i].tolist()
                
                # 转换回实体ID字符串
                top10_entities = [mapper.id_to_entity[t_id] for t_id in top10_t_ids]
                
                # 构建结果行
                result_line = [h, r] + top10_entities
                results_buffer.append('\t'.join(result_line))
                
                # 当缓冲区达到一定大小时，写入文件以减少内存使用
                if len(results_buffer) >= buffer_size:
                    results.extend(results_buffer)
                    results_buffer = []
            
            # 显示进度条
            progress = batch_end / process_count * 100
            bar_length = 50
            filled_length = int(bar_length * progress // 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"\r  预测进度: |{bar}| {progress:.1f}% ({batch_end}/{process_count})", end='')
            
            # 定期清理内存
            if batch_start % (batch_size * 10) == 0:
                del h_emb, r_emb, h_plus_r, batch_scores
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.backends.mps.empty_cache()
                gc.collect()
    
    # 添加剩余的结果
    if results_buffer:
        results.extend(results_buffer)
    
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


# 主函数
def main():
    # 设置MPS优化
    setup_mps_optimization()
    
    # 选择最佳可用设备 - 优先使用MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("使用MPS设备加速计算")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("使用CUDA设备加速计算")
    else:
        device = torch.device('cpu')
        print("使用CPU设备")
    
    print(f"使用设备: {device}")
    print(f"并行工作线程数: {NUM_WORKERS}")
    
    # 加载数据集 - 使用并行加载器
    print("正在并行加载数据集...")
    start_time = time.time()
    
    # 使用ProcessPoolExecutor进行并行数据集加载
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_train = executor.submit(ParallelKnowledgeGraphDataset, TRAIN_FILE_PATH, False, False, MAX_LINES)
        future_test = executor.submit(ParallelKnowledgeGraphDataset, TEST_FILE_PATH, True, False)
        future_dev = executor.submit(ParallelKnowledgeGraphDataset, DEV_FILE_PATH, False, True)
        
        train_dataset = future_train.result()
        test_dataset = future_test.result()
        dev_dataset = future_dev.result()
    
    load_time = time.time() - start_time
    print(f"数据集加载完成，耗时: {load_time:.2f}秒")
    
    print(f"训练数据大小: {len(train_dataset)}")
    print(f"测试数据大小: {len(test_dataset)}")
    print(f"开发数据大小: {len(dev_dataset)}")
    
    # 构建实体和关系映射 - 使用并行构建
    print("正在并行构建实体和关系映射...")
    mapper = EntityRelationMapper()
    mapper.build_mappings_parallel(train_dataset, test_dataset, dev_dataset)
    
    print(f"实体数量: {mapper.entity_count}")
    print(f"关系数量: {mapper.relation_count}")
    
    # 创建模型 - MPS优化版
    print("正在创建TransE模型...")
    model = TransE(mapper.entity_count, mapper.relation_count, EMBEDDING_DIM, device)
    model.to(device)
    
    # 训练模型 - MPS优化版
    print("开始训练模型...")
    model = train_model(model, train_dataset, mapper, device)
    
    # 在开发集上评估模型 - MPS优化版
    print("\n训练完成，开始在开发集上评估模型性能...")
    evaluate(model, dev_dataset, mapper, device)
    
    # 清理不再需要的数据集以释放内存
    del train_dataset, dev_dataset
    gc.collect()
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.backends.mps.empty_cache()
    
    # 预测尾实体 - MPS优化版
    predict_tail_entities(model, test_dataset, mapper, device, MAX_HEAD_ENTITIES)
    
    print("任务完成！")

if __name__ == "__main__":
    main()