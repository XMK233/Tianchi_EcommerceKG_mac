import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# 设置随机种子以确保结果可复现
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed()

# 数据路径
TRAIN_FILE_PATH = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_train.tsv"
TEST_FILE_PATH = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_test.tsv"
OUTPUT_FILE_PATH = "./rst.tsv"

# 超参数
EMBEDDING_DIM = 200
LEARNING_RATE = 0.01
MARGIN = 1.0
BATCH_SIZE = 1024
NEGATIVE_SAMPLES = 5

MAX_LINES = None ## 拿多少个样本训练。
EPOCHS = 5 # 200 ## 训练几个epoch。
max_head_entities = None  # 可以修改为任意整数，如1000表示只处理前1000个

# 数据加载类
class KnowledgeGraphDataset(Dataset):
    def __init__(self, file_path, is_test=False, max_lines=None):
        self.is_test = is_test
        self.triples = []
        
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
                    # 训练文件有完整的三元组
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
        self.entity_to_id = {}
        self.id_to_entity = {}
        self.relation_to_id = {}
        self.id_to_relation = {}
        self.entity_count = 0
        self.relation_count = 0
    
    def add_entity(self, entity):
        if entity not in self.entity_to_id:
            self.entity_to_id[entity] = self.entity_count
            self.id_to_entity[self.entity_count] = entity
            self.entity_count += 1
    
    def add_relation(self, relation):
        if relation not in self.relation_to_id:
            self.relation_to_id[relation] = self.relation_count
            self.id_to_relation[self.relation_count] = relation
            self.relation_count += 1
    
    def build_mappings(self, train_dataset, test_dataset):
        # 从训练数据构建映射
        for h, r, t in train_dataset.triples:
            self.add_entity(h)
            self.add_entity(t)
            self.add_relation(r)
        
        # 从测试数据构建映射（确保所有实体和关系都被包含）
        for h, r, _ in test_dataset.triples:
            self.add_entity(h)
            self.add_relation(r)

# TransE模型实现
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransE, self).__init__()
        # 初始化实体和关系嵌入
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # 初始化嵌入向量
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
        # 获取嵌入向量
        h_emb = self.entity_embeddings(h)
        r_emb = self.relation_embeddings(r)
        t_emb = self.entity_embeddings(t)
        
        # 计算得分：||h + r - t||
        pos_score = torch.norm(h_emb + r_emb - t_emb, p=1, dim=1)
        
        if t_neg is not None:
            # 计算负样本得分
            t_neg_emb = self.entity_embeddings(t_neg)
            h_emb_expanded = h_emb.unsqueeze(1).expand(-1, NEGATIVE_SAMPLES, -1)
            r_emb_expanded = r_emb.unsqueeze(1).expand(-1, NEGATIVE_SAMPLES, -1)
            
            neg_score = torch.norm(h_emb_expanded + r_emb_expanded - t_neg_emb, p=1, dim=2)
            return pos_score, neg_score
        
        return pos_score

# 训练函数
def train_model(model, train_dataset, mapper, device):
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 定义优化器和损失函数
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # 为每个epoch打印进度条
        print(f"\nEpoch {epoch+1}/{EPOCHS} 训练中...")
        
        # 使用enumerate来获取batch索引
        for batch_idx, batch in enumerate(train_loader):
            h_batch, r_batch, t_batch = batch
            
            # 将实体和关系转换为ID
            h_ids = torch.tensor([mapper.entity_to_id[h] for h in h_batch], device=device)
            r_ids = torch.tensor([mapper.relation_to_id[r] for r in r_batch], device=device)
            t_ids = torch.tensor([mapper.entity_to_id[t] for t in t_batch], device=device)
            
            # 生成负样本
            batch_size = h_ids.size(0)
            t_neg_ids = []
            for i in range(batch_size):
                # 为每个正样本生成多个负样本
                neg_samples = []
                while len(neg_samples) < NEGATIVE_SAMPLES:
                    # 随机选择一个实体作为负样本尾实体
                    neg_id = random.randint(0, mapper.entity_count - 1)
                    # 确保负样本不是正样本
                    if neg_id != t_ids[i].item():
                        neg_samples.append(neg_id)
                t_neg_ids.append(neg_samples)
            
            t_neg_ids = torch.tensor(t_neg_ids, device=device)
            
            # 前向传播
            pos_score, neg_score = model(h_ids, r_ids, t_ids, t_neg_ids)
            
            # 计算损失（基于margin的max-margin损失）
            # 将pos_score扩展为与neg_score相同的维度
            pos_score_expanded = pos_score.unsqueeze(1).expand_as(neg_score)
            loss = torch.sum(torch.relu(pos_score_expanded - neg_score + MARGIN))
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 归一化实体嵌入
            model.normalize_entities()
            
            total_loss += loss.item()
            
            # 每个batch打印一次进度
            progress = (batch_idx + 1) / len(train_loader) * 100
            print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f} - Progress: {progress:.1f}%", end='\r')
        
        # 每个epoch结束后打印平均损失
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{EPOCHS} - Average Loss: {avg_loss:.4f}")

# 预测函数 - 优化版本（向量化计算）
def predict_tail_entities(model, test_dataset, mapper, device, max_head_entities=None, batch_size=128):
    model.eval()
    results = []
    total_test_triples = len(test_dataset.triples)
    
    # 确定要处理的头实体数量
    process_count = total_test_triples if max_head_entities is None else min(max_head_entities, total_test_triples)
    print(f"\n开始预测尾实体...")
    print(f"将处理 {process_count} 个头实体/关系对")
    
    # 预计算所有实体嵌入（只需一次，避免重复计算）
    all_entities = torch.arange(mapper.entity_count, device=device)
    all_entity_embeddings = model.entity_embeddings(all_entities)
    
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
            
            # 使用广播机制一次性计算所有可能的尾实体得分
            # 这是向量化操作的关键部分，避免了循环遍历每个尾实体
            h_plus_r_expanded = h_plus_r.unsqueeze(1)  # [batch_size, 1, embedding_dim]
            scores = torch.norm(h_plus_r_expanded - all_entity_embeddings.unsqueeze(0), p=1, dim=2)  # [batch_size, entity_count]
            
            # 获取每个头实体/关系对的top10尾实体
            _, top10_indices = torch.topk(-scores, k=10, dim=1)  # 使用负数以获取最小的10个值
            
            # 处理批次结果
            for i in range(len(batch_triples)):
                h, r, _ = batch_triples[i]  # 正确解包3元素元组
                top10_t_ids = top10_indices[i].tolist()
                
                # 转换回实体ID字符串
                top10_entities = [mapper.id_to_entity[t_id] for t_id in top10_t_ids]
                
                # 构建结果行
                result_line = [h, r] + top10_entities
                results.append('\t'.join(result_line))
            
            # 显示进度
            progress = batch_end / process_count * 100
            print(f"  预测进度: {batch_end}/{process_count} - {progress:.1f}%", end='\r')
    
    print()  # 确保进度条完成后换行
    
    # 写入结果文件
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')
    
    print(f"预测结果已保存到 {OUTPUT_FILE_PATH}")

# 主函数
def main():
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("正在加载数据集...")
    # 限制训练数据为10000行以节省训练时间
    train_dataset = KnowledgeGraphDataset(TRAIN_FILE_PATH, max_lines=MAX_LINES)
    test_dataset = KnowledgeGraphDataset(TEST_FILE_PATH, is_test=True)
    
    print(f"训练数据大小: {len(train_dataset)}")
    print(f"测试数据大小: {len(test_dataset)}")
    
    # 构建实体和关系映射
    print("正在构建实体和关系映射...")
    mapper = EntityRelationMapper()
    mapper.build_mappings(train_dataset, test_dataset)
    
    print(f"实体数量: {mapper.entity_count}")
    print(f"关系数量: {mapper.relation_count}")
    
    # 创建模型
    print("正在创建TransE模型...")
    model = TransE(mapper.entity_count, mapper.relation_count, EMBEDDING_DIM)
    model.to(device)
    
    # 训练模型
    print("开始训练模型...")
    train_model(model, train_dataset, mapper, device)
    
    # 预测尾实体
    # 可以通过max_head_entities参数控制处理的头实体个数，None表示处理全部
    predict_tail_entities(model, test_dataset, mapper, device, max_head_entities)
    
    print("任务完成！")

if __name__ == "__main__":
    main()