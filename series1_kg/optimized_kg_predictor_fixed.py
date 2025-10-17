import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import time
import gc

# 设置随机种子，保证结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 数据路径配置
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/originalData/OpenBG500/OpenBG500_train.tsv"
TEST_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/originalData/OpenBG500/OpenBG500_test.tsv"
DEV_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/originalData/OpenBG500/OpenBG500_dev.tsv" # 开发集路径
OUTPUT_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/preprocessedData/OpenBG500_test.tsv"

# 确保输出目录存在
os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)

# 超参数设置
EMBEDDING_DIM = 200  # 嵌入维度
LEARNING_RATE = 0.001  # 学习率
BATCH_SIZE = 1024  # 批次大小（针对CPU可适当调小）
NUM_EPOCHS = 1  # 训练轮数
MARGIN = 1.0  # 间隔值
NEGATIVE_SAMPLES = 1  # 负样本数量
MAX_LINES = None  # 限制加载的行数，None表示加载全部
MAX_HEAD_ENTITIES = None  # 限制预测的头实体数量，None表示全部处理

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

# 数据集类
class KnowledgeGraphDataset(Dataset):
    def __init__(self, file_path, is_test=False, is_dev=False, max_lines=None):
        self.triples = []
        self.is_test = is_test
        self.is_dev = is_dev
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if max_lines is not None:
                    lines = lines[:max_lines]
                
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        h, r, t = parts[0], parts[1], parts[2]
                        self.triples.append((h, r, t))
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
    
    def build_mappings(self, train_dataset, test_dataset, dev_dataset):
        # 收集所有实体和关系
        all_entities = set()
        all_relations = set()
        
        # 处理训练集
        for h, r, t in train_dataset.triples:
            all_entities.add(h)
            all_entities.add(t)
            all_relations.add(r)
        
        # 处理测试集
        for h, r, _ in test_dataset.triples:
            all_entities.add(h)
            all_relations.add(r)
        
        # 处理开发集
        for h, r, t in dev_dataset.triples:
            all_entities.add(h)
            all_entities.add(t)
            all_relations.add(r)
        
        # 构建映射
        self.entity_to_id = {entity: i for i, entity in enumerate(sorted(all_entities))}
        self.id_to_entity = sorted(all_entities)
        self.relation_to_id = {relation: i for i, relation in enumerate(sorted(all_relations))}
        self.id_to_relation = sorted(all_relations)
        
        # 更新计数
        self.entity_count = len(self.id_to_entity)
        self.relation_count = len(self.id_to_relation)

# TransE模型
class TransE(torch.nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransE, self).__init__()
        self.entity_embeddings = torch.nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(num_relations, embedding_dim)
        self.embedding_dim = embedding_dim
        
        # 初始化嵌入向量
        self.initialize_embeddings()
    
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
        # 处理正样本
        h_pos = torch.tensor([mapper.entity_to_id[h] for h, _, _ in positive_triples])
        r_pos = torch.tensor([mapper.relation_to_id[r] for _, r, _ in positive_triples])
        t_pos = torch.tensor([mapper.entity_to_id[t] for _, _, t in positive_triples])
        
        # 获取嵌入向量
        h_emb_pos = self.entity_embeddings(h_pos)
        r_emb_pos = self.relation_embeddings(r_pos)
        t_emb_pos = self.entity_embeddings(t_pos)
        
        # 计算正样本得分 (L1距离)
        pos_scores = torch.norm(h_emb_pos + r_emb_pos - t_emb_pos, p=1, dim=1)
        
        # 处理负样本
        h_neg = torch.tensor([mapper.entity_to_id[h] for h, _, _ in negative_triples])
        r_neg = torch.tensor([mapper.relation_to_id[r] for _, r, _ in negative_triples])
        t_neg = torch.tensor([mapper.entity_to_id[t] for _, _, t in negative_triples])
        
        # 获取嵌入向量
        h_emb_neg = self.entity_embeddings(h_neg)
        r_emb_neg = self.relation_embeddings(r_neg)
        t_emb_neg = self.entity_embeddings(t_neg)
        
        # 计算负样本得分 (L1距离)
        neg_scores = torch.norm(h_emb_neg + r_emb_neg - t_emb_neg, p=1, dim=1)
        
        # 计算损失 (基于间隔的损失函数)
        # 确保正样本得分尽可能小，负样本得分尽可能大
        loss = torch.mean(torch.relu(pos_scores - neg_scores + MARGIN))
        
        # 定期归一化实体嵌入
        if self.training:
            self.normalize_entity_embeddings()
        
        return loss

# 负采样函数
def generate_negative_samples(triple, mapper, num_negatives=1):
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

# 训练函数
def train_model(model, train_dataset, mapper, device):
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练循环
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        # 手动分批次处理数据，避免使用多进程DataLoader
        for batch_start in range(0, len(train_dataset), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(train_dataset))
            batch_triples = train_dataset.triples[batch_start:batch_end]
            
            # 生成负样本
            positive_triples = []
            negative_triples = []
            
            for triple in batch_triples:
                positive_triples.append(triple)
                negatives = generate_negative_samples(triple, mapper, NEGATIVE_SAMPLES)
                negative_triples.extend(negatives)
            
            # 前向传播
            loss = model(positive_triples, negative_triples, mapper)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 显示批次进度
            if batch_end % (BATCH_SIZE * 10) == 0:
                progress = batch_end / len(train_dataset) * 100
                print(f"  批次进度: {progress:.1f}% ({batch_end}/{len(train_dataset)})", end='\r')
        
        # 清理内存
        gc.collect()
        
        # 打印轮次信息
        epoch_time = time.time() - start_time
        avg_loss = total_loss / (len(train_dataset) / BATCH_SIZE)
        print(f"\n轮次 {epoch+1}/{NUM_EPOCHS} - 平均损失: {avg_loss:.4f} - 耗时: {epoch_time:.2f}秒")
    
    return model

# 评估函数 - 优化版本
def evaluate(model, dev_dataset, mapper, device, batch_size=128, max_entities_per_batch=10000):
    model.eval()
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_10 = 0
    mrr_score = 0
    total_count = 0
    total_dev_triples = len(dev_dataset.triples)
    
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
            h_ids = torch.tensor([mapper.entity_to_id[h] for h in h_list])
            r_ids = torch.tensor([mapper.relation_to_id[r] for r in r_list])
            t_ids = torch.tensor([mapper.entity_to_id[t] for t in t_list])
            
            # 获取嵌入向量并计算h + r
            h_emb = model.entity_embeddings(h_ids)  # [current_batch_size, embedding_dim]
            r_emb = model.relation_embeddings(r_ids)  # [current_batch_size, embedding_dim]
            h_plus_r = h_emb + r_emb  # [current_batch_size, embedding_dim]
            
            # 分块处理实体，避免一次性加载所有实体导致内存溢出
            batch_scores = torch.zeros(current_batch_size, mapper.entity_count)
            
            # 分块计算得分 - 向量化实现
            for entity_start in range(0, mapper.entity_count, max_entities_per_batch):
                entity_end = min(entity_start + max_entities_per_batch, mapper.entity_count)
                entity_chunk_size = entity_end - entity_start
                
                # 获取当前块的实体嵌入
                entity_ids = torch.arange(entity_start, entity_end)
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
                
                # 清理中间变量，释放内存
                del entity_embeddings, chunk_scores
            
            # 查找每个样本中正确尾实体的得分
            correct_scores = torch.gather(batch_scores, 1, t_ids.view(-1, 1))  # [current_batch_size, 1]
            
            # 计算排名（得分越低排名越高） - 通过向量化比较实现
            # 计算有多少实体得分比正确实体低（排名即这个数量+1）
            num_better_entities = (batch_scores < correct_scores).sum(dim=1)  # [current_batch_size]
            ranks = num_better_entities + 1  # [current_batch_size] - 排名从1开始
            
            # 计算指标
            hits_at_1 += (ranks == 1).sum().item()
            hits_at_3 += (ranks <= 3).sum().item()
            hits_at_10 += (ranks <= 10).sum().item()
            
            # 对于排名n>10的实体，得分为0
            mrr_contributions = torch.where(ranks <= 10, 1.0 / ranks, torch.tensor(0.0))
            mrr_score += mrr_contributions.sum().item()
            total_count += current_batch_size
            
            # 显示进度
            progress = min(total_count, total_dev_triples) / total_dev_triples * 100
            if total_count % 500 == 0 or total_count >= total_dev_triples:
                print(f"  评估进度: {progress:.1f}% ({min(total_count, total_dev_triples)}/{total_dev_triples})")
            
            # 清理内存
            del h_emb, r_emb, h_plus_r, batch_scores, correct_scores, num_better_entities, ranks
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

# 预测函数 - 优化版本
def predict_tail_entities(model, test_dataset, mapper, device, max_head_entities=None, batch_size=128, max_entities_per_batch=10000):
    model.eval()
    results = []
    total_test_triples = len(test_dataset.triples)
    
    # 确定要处理的头实体数量
    process_count = total_test_triples if max_head_entities is None else min(max_head_entities, total_test_triples)
    print(f"\n开始预测尾实体...")
    print(f"将处理 {process_count} 个头实体/关系对")
    
    # 计算每个批次能处理的最大实体数量（避免内存溢出）
    max_entities_per_batch = min(10000, mapper.entity_count)  # 限制最大实体数
    
    with torch.no_grad():
        # 创建批次进行处理
        for batch_start in range(0, process_count, batch_size):
            batch_end = min(batch_start + batch_size, process_count)
            batch_triples = test_dataset.triples[batch_start:batch_end]
            
            # 获取批次中的头实体和关系
            h_list = [h for h, _, _ in batch_triples]
            r_list = [r for _, r, _ in batch_triples]
            
            # 转换为ID和张量
            h_ids = torch.tensor([mapper.entity_to_id[h] for h in h_list])
            r_ids = torch.tensor([mapper.relation_to_id[r] for r in r_list])
            
            # 获取嵌入向量
            h_emb = model.entity_embeddings(h_ids)  # [batch_size, embedding_dim]
            r_emb = model.relation_embeddings(r_ids)  # [batch_size, embedding_dim]
            
            # 计算h + r
            h_plus_r = h_emb + r_emb  # [batch_size, embedding_dim]
            
            # 分块处理实体，避免一次性加载所有实体导致内存溢出
            batch_scores = torch.zeros(len(batch_triples), mapper.entity_count)
            
            # 分块计算得分
            for entity_start in range(0, mapper.entity_count, max_entities_per_batch):
                entity_end = min(entity_start + max_entities_per_batch, mapper.entity_count)
                
                # 获取当前块的实体嵌入
                entity_ids = torch.arange(entity_start, entity_end)
                entity_embeddings = model.entity_embeddings(entity_ids)  # [chunk_size, embedding_dim]
                
                # 计算当前块的得分
                h_plus_r_expanded = h_plus_r.unsqueeze(1)  # [batch_size, 1, embedding_dim]
                chunk_scores = torch.norm(h_plus_r_expanded - entity_embeddings.unsqueeze(0), p=1, dim=2)  # [batch_size, chunk_size]
                
                # 存储到完整得分矩阵中
                batch_scores[:, entity_start:entity_end] = chunk_scores
                
                # 清理中间变量，释放内存
                del entity_embeddings, chunk_scores
                gc.collect()
            
            # 获取每个头实体/关系对的top10尾实体
            _, top10_indices = torch.topk(-batch_scores, k=10, dim=1)  # 使用负数以获取最小的10个值
            
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
            
            # 定期清理内存
            if batch_start % (batch_size * 10) == 0:
                gc.collect()
    
    print()  # 确保进度条完成后换行
    
    # 写入结果文件
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')
    
    print(f"预测结果已保存到 {OUTPUT_FILE_PATH}")

# 主函数
def main():
    # 选择CPU设备
    device = torch.device('cpu')
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
    predict_tail_entities(model, test_dataset, mapper, device, MAX_HEAD_ENTITIES)
    
    print("任务完成！")

if __name__ == "__main__":
    main()