import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, default_collate
import random
import time
import jieba
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import re

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

# 中文文本数据路径
ENTITY_TEXT_PATH = "D:\\forCoding_data\\Tianchi_EcommerceKG\\originalData\\OpenBG500\\OpenBG500_entity2text.tsv"
RELATION_TEXT_PATH = "D:\\forCoding_data\\Tianchi_EcommerceKG\\originalData\\OpenBG500\\OpenBG500_relation2text.tsv"

# 预训练词向量路径（假设使用中文维基百科预训练的Word2Vec模型）
# 注意：用户需要下载这个模型，这里提供的是假设路径
WORD2VEC_PATH = "D:\\pretrained_models\\wiki.zh.word2vec.bin"

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

# NLP参数
TEXT_EMBEDDING_DIM = 300  # 文本嵌入维度
TEXT_WEIGHT = 0.3  # 文本特征在最终评分中的权重
TOP_K_CANDIDATES = 100  # 在文本排序阶段考虑的候选实体数量

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

# 实体和关系映射管理器 - 增强版：支持中文文本信息
class EnhancedEntityRelationMapper:
    def __init__(self):
        self.entity_to_id = {}  # 实体到ID的映射字典
        self.id_to_entity = {}  # ID到实体的映射字典
        self.relation_to_id = {}  # 关系到ID的映射字典
        self.id_to_relation = {}  # ID到关系的映射字典
        self.entity_count = 0  # 实体数量
        self.relation_count = 0  # 关系数量
        
        # 添加中文文本映射
        self.entity_to_text = {}  # 实体到中文文本的映射
        self.relation_to_text = {}  # 关系到中文文本的映射
        self.entity_text_embeddings = {}  # 实体中文文本的向量表示
        self.relation_text_embeddings = {}  # 关系中文文本的向量表示
    
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
    
    def load_text_mappings(self, entity_text_path, relation_text_path):
        """加载实体和关系的中文文本信息"""
        # 加载实体中文文本
        try:
            with open(entity_text_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        entity_id = parts[0]
                        entity_text = parts[1]
                        self.entity_to_text[entity_id] = entity_text
            print(f"已加载 {len(self.entity_to_text)} 个实体的中文文本信息")
        except Exception as e:
            print(f"加载实体文本信息时出错: {e}")
        
        # 加载关系中文文本
        try:
            with open(relation_text_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        relation_id = parts[0]
                        relation_text = parts[1]
                        self.relation_to_text[relation_id] = relation_text
            print(f"已加载 {len(self.relation_to_text)} 个关系的中文文本信息")
        except Exception as e:
            print(f"加载关系文本信息时出错: {e}")
    
    def init_text_embeddings(self, word2vec_path=None):
        """初始化中文文本的向量表示"""
        # 尝试加载预训练词向量模型
        word_vectors = None
        try:
            if os.path.exists(word2vec_path):
                print(f"正在加载预训练词向量模型: {word2vec_path}")
                word_vectors = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
                print("词向量模型加载完成")
            else:
                print(f"预训练词向量模型文件不存在: {word2vec_path}")
                print("将使用随机初始化的向量")
        except Exception as e:
            print(f"加载预训练词向量模型时出错: {e}")
            print("将使用随机初始化的向量")
        
        # 为实体文本创建向量表示
        for entity_id, text in self.entity_to_text.items():
            self.entity_text_embeddings[entity_id] = self._get_text_embedding(text, word_vectors)
        
        # 为关系文本创建向量表示
        for relation_id, text in self.relation_to_text.items():
            self.relation_text_embeddings[relation_id] = self._get_text_embedding(text, word_vectors)
    
    def _get_text_embedding(self, text, word_vectors):
        """获取文本的向量表示"""
        # 清理文本
        text = self._clean_text(text)
        
        # 分词
        words = list(jieba.cut(text))
        
        if not words:
            return np.random.rand(TEXT_EMBEDDING_DIM)
        
        # 使用词向量计算文本向量
        if word_vectors is not None:
            vectors = []
            for word in words:
                if word in word_vectors:
                    vectors.append(word_vectors[word])
            
            if vectors:
                return np.mean(vectors, axis=0)
        
        # 如果没有词向量或词汇不在词向量中，返回随机向量
        return np.random.rand(TEXT_EMBEDDING_DIM)
    
    def _clean_text(self, text):
        """清理文本，去除特殊字符"""
        # 保留中文、英文、数字和常见标点
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。, .]', ' ', text)
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def get_text_similarity(self, h_text, r_text, t_text):
        """计算头实体、关系和尾实体之间的文本语义相似度"""
        if h_text in self.entity_text_embeddings and r_text in self.relation_text_embeddings and t_text in self.entity_text_embeddings:
            h_emb = self.entity_text_embeddings[h_text].reshape(1, -1)
            r_emb = self.relation_text_embeddings[r_text].reshape(1, -1)
            t_emb = self.entity_text_embeddings[t_text].reshape(1, -1)
            
            # 计算语义匹配度：h + r 与 t 的相似度
            h_r_emb = (h_emb + r_emb) / 2  # 简单融合头实体和关系的文本向量
            similarity = cosine_similarity(h_r_emb, t_emb)[0][0]
            return similarity
        else:
            return 0.0

# TransE模型实现 - 增强版：支持文本特征融合
class EnhancedTransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, mapper=None):
        super(EnhancedTransE, self).__init__()
        # 初始化实体和关系嵌入层
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # 使用Xavier均匀分布初始化嵌入向量
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        
        # 归一化实体嵌入
        self.normalize_entities()
        
        # 存储mapper引用，用于访问文本信息
        self.mapper = mapper
    
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

# 预测函数 - 增强版：结合结构特征和文本特征
def predict_tail_entities_with_text(model, test_dataset, mapper, device, max_head_entities=None, batch_size=128, max_entities_per_batch=10000):
    model.eval()
    results = []
    total_test_triples = len(test_dataset.triples)
    
    # 确定要处理的头实体数量
    process_count = total_test_triples if max_head_entities is None else min(max_head_entities, total_test_triples)
    print(f"\n开始预测尾实体...")
    print(f"将处理 {process_count} 个头实体/关系对")
    print(f"使用文本特征增强预测，文本特征权重: {TEXT_WEIGHT}")
    
    # 计算每个批次能处理的最大实体数量（避免显存溢出）
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
            
            # 处理批次结果，结合文本特征
            for i in range(len(batch_triples)):
                h, r, _ = batch_triples[i]  # 正确解包3元素元组
                
                # 首先获取TransE模型的top K候选实体
                _, top_k_indices = torch.topk(-batch_scores[i], k=TOP_K_CANDIDATES, dim=0)  # 使用负数以获取最小的K个值
                top_k_ids = top_k_indices.tolist()
                top_k_entities = [mapper.id_to_entity[t_id] for t_id in top_k_ids]
                
                # 计算每个候选实体的文本相似度得分
                text_scores = []
                for candidate in top_k_entities:
                    # 获取原始ID（不是内部索引）
                    original_h_id = h
                    original_r_id = r
                    original_t_id = candidate
                    
                    # 计算文本相似度得分
                    similarity = mapper.get_text_similarity(original_h_id, original_r_id, original_t_id)
                    text_scores.append(similarity)
                
                # 融合结构得分和文本得分
                combined_scores = []
                for j, t_id in enumerate(top_k_ids):
                    # TransE得分（越低越好）转换为（越高越好）
                    structure_score = 1.0 / (1.0 + batch_scores[i][t_id].item())
                    # 文本得分（已经是越高越好）
                    text_score = text_scores[j]
                    # 加权融合
                    combined_score = (1 - TEXT_WEIGHT) * structure_score + TEXT_WEIGHT * text_score
                    combined_scores.append((combined_score, t_id))
                
                # 根据融合得分排序，取top10
                combined_scores.sort(reverse=True, key=lambda x: x[0])
                top10_t_ids = [t_id for _, t_id in combined_scores[:10]]
                
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
    
    # 构建实体和关系映射 - 使用增强版映射管理器
    print("正在构建实体和关系映射...")
    mapper = EnhancedEntityRelationMapper()
    mapper.build_mappings(train_dataset, test_dataset, dev_dataset)
    
    # 加载实体和关系的中文文本信息
    print("正在加载实体和关系的中文文本信息...")
    mapper.load_text_mappings(ENTITY_TEXT_PATH, RELATION_TEXT_PATH)
    
    # 初始化文本嵌入
    print("正在初始化文本嵌入...")
    mapper.init_text_embeddings(WORD2VEC_PATH)
    
    print(f"实体数量: {mapper.entity_count}")
    print(f"关系数量: {mapper.relation_count}")
    
    # 创建模型 - 使用增强版TransE模型
    print("正在创建增强版TransE模型...")
    model = EnhancedTransE(mapper.entity_count, mapper.relation_count, EMBEDDING_DIM, mapper)
    model.to(device)
    
    # 训练模型
    print("开始训练模型...")
    train_model(model, train_dataset, mapper, device)
    
    # 在开发集上评估模型
    print("\n训练完成，开始在开发集上评估模型性能...")
    evaluate(model, dev_dataset, mapper, device)
    
    # 预测尾实体 - 使用结合文本特征的预测函数
    # 可以通过max_head_entities参数控制处理的头实体个数，None表示处理全部
    predict_tail_entities_with_text(model, test_dataset, mapper, device, max_head_entities)
    
    print("任务完成！")

if __name__ == "__main__":
    main()