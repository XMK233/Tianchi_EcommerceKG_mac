# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, BatchSampler, SequentialSampler
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
import numpy as np
import time
import psutil
import gc
import sys
from tqdm import tqdm

# 设置随机种子以保证结果可复现
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 内存优化工具函数
def release_memory(obj):
    """安全释放对象内存"""
    if isinstance(obj, torch.Tensor):
        obj.detach_()
        if obj.is_cuda:
            obj = obj.cpu()
    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 监控内存使用的装饰器
def memory_monitor(func):
    """监控函数的内存使用情况"""
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            before_alloc = torch.cuda.memory_allocated()
            before_reserved = torch.cuda.memory_reserved()
        
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            after_alloc = torch.cuda.memory_allocated()
            after_reserved = torch.cuda.memory_reserved()
            used = after_alloc - before_alloc
            reserved = after_reserved - before_reserved
            if abs(used) > 1024**2:  # 大于1MB时才打印
                print(f"  函数 {func.__name__} 内存变化: {used/1024**2:.2f}MB (分配), {reserved/1024**2:.2f}MB (预留)")
        
        return result
    return wrapper

# 数据路径
TRAIN_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/originalData/OpenBG500/OpenBG500_train.tsv"
TEST_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/originalData/OpenBG500/OpenBG500_test.tsv"
DEV_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/originalData/OpenBG500/OpenBG500_dev.tsv" # 开发集路径
OUTPUT_FILE_PATH = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/preprocessedData/OpenBG500_test.tsv"

# 超参数 - 内存优化版
EMBEDDING_DIM = 200  # 实体和关系嵌入的维度
LEARNING_RATE = 0.001  # 学习率
WEIGHT_DECAY = 1e-5  # 权重衰减（L2正则化）

# 内存优化配置
BATCH_SIZE = 64  # 更小的批次大小（从256减少到64）
NEGATIVE_SAMPLES = 2  # 减少负样本数量（从5减少到2）
NEG_CHUNK_SIZE = 2  # 负样本分块处理（更小的块大小）

# 训练配置
EPOCHS = 6  # 训练的epoch数量
MAX_LINES = None  # 限制训练数据的行数，None表示使用全部数据
MAX_HEAD_ENTITIES = None  # 限制预测的头实体数量，None表示处理全部

# 内存和显存限制配置
MAX_GPU_MEMORY_GB = 32  # 最大GPU显存使用限制（GB）
MAX_CPU_MEMORY_GB = 16  # 最大CPU内存使用限制（GB）
GRADIENT_ACCUMULATION_STEPS = 8  # 梯度累积步数（增加到8以保持有效批次大小）
MIXED_PRECISION = False  # 禁用混合精度训练以减少内存碎片化

# 学习率调度器参数
LR_DECAY_FACTOR = 0.5  # 学习率衰减因子
LR_DECAY_STEP = 50  # 每隔多少个epoch衰减一次学习率

# 评估和预测配置
EVAL_BATCH_SIZE = 16  # 评估批次大小
PREDICT_BATCH_SIZE = 8  # 预测批次大小
MAX_ENTITIES_PER_BATCH = 500  # 最大实体批次大小（进一步减小）

# 迭代式数据加载函数 - 减少内存占用
def load_triples_from_file(file_path, is_test=False, max_lines=None):
    """从文件中逐行加载三元组数据，避免一次性加载全部到内存"""
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
                    yield (h, r, None)
                    line_count += 1
            else:
                # 训练和开发文件有完整的三元组
                if len(parts) >= 3:
                    h, r, t = parts[0], parts[1], parts[2]
                    yield (h, r, t)
                    line_count += 1

# 数据集类 - 内存优化版本
class KnowledgeGraphDataset(Dataset):
    def __init__(self, file_path, is_test=False, is_dev=False, max_lines=None):
        """初始化数据集，使用缓存机制减少内存使用"""
        self.file_path = file_path
        self.is_test = is_test
        self.is_dev = is_dev
        self.max_lines = max_lines
        
        # 使用生成器函数加载三元组，而不是一次性加载全部
        self.triples = []
        self._load_triples_with_cache()
        
    def _load_triples_with_cache(self):
        """使用生成器和缓存加载三元组"""
        for triple in self.load_triples_from_file():
            self.triples.append(triple)
            # 如果达到最大行数限制，停止加载
            if self.max_lines is not None and len(self.triples) >= self.max_lines:
                break
    
    def load_triples_from_file(self):
        """生成器函数，逐行读取文件并返回三元组"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if self.max_lines is not None and line_num >= self.max_lines:
                    break
                
                # 去除行尾的换行符并分割
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h = parts[0]
                    r = parts[1]
                    # 对于测试集，尾实体可能为空
                    t = None if self.is_test else parts[2]
                    yield (h, r, t)
                else:
                    # 忽略格式不正确的行
                    continue
                    
    def __len__(self):
        return len(self.triples)
        
    def __getitem__(self, idx):
        # 实现随机访问
        return self.triples[idx]

# 流式数据集类 - 完全基于迭代器，不保存完整数据到内存
class StreamingKnowledgeGraphDataset(IterableDataset):
    def __init__(self, file_path, is_test=False, is_dev=False, buffer_size=10000):
        """初始化流式数据集"""
        self.file_path = file_path
        self.is_test = is_test
        self.is_dev = is_dev
        self.buffer_size = buffer_size  # 缓冲区大小，用于在有限内存中实现shuffle
    
    def __iter__(self):
        """返回迭代器，支持流式读取数据"""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # 单进程模式
            return self._single_process_iterator()
        else:
            # 多进程模式
            return self._multi_process_iterator(worker_info)
    
    def _single_process_iterator(self):
        """单进程模式的迭代器"""
        buffer = []
        for triple in self.load_triples_from_file():
            buffer.append(triple)
            # 当缓冲区达到指定大小时，打乱并逐个返回
            if len(buffer) >= self.buffer_size:
                np.random.shuffle(buffer)
                for item in buffer:
                    yield item
                buffer = []
        # 返回剩余的缓冲区数据
        if buffer:
            np.random.shuffle(buffer)
            for item in buffer:
                yield item
    
    def _multi_process_iterator(self, worker_info):
        """多进程模式的迭代器，每个进程处理文件的不同部分"""
        # 注意：对于流式处理，很难实现真正的文件分片
        # 这里只是将每个worker使用独立的缓冲区进行shuffle
        buffer = []
        for triple in self.load_triples_from_file():
            # 简单的worker过滤策略，可能不够均衡
            if hash(triple[0]) % worker_info.num_workers == worker_info.id:
                buffer.append(triple)
                # 当缓冲区达到指定大小时，打乱并逐个返回
                if len(buffer) >= self.buffer_size:
                    np.random.shuffle(buffer)
                    for item in buffer:
                        yield item
                    buffer = []
        # 返回剩余的缓冲区数据
        if buffer:
            np.random.shuffle(buffer)
            for item in buffer:
                yield item
                
    def load_triples_from_file(self):
        """生成器函数，逐行读取文件并返回三元组"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 去除行尾的换行符并分割
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    h = parts[0]
                    r = parts[1]
                    # 对于测试集，尾实体可能为空
                    t = None if self.is_test else parts[2]
                    yield (h, r, t)
                else:
                    # 忽略格式不正确的行
                    continue

# 内存优化的数据加载器包装器
class MemoryEfficientDataLoader:
    """内存优化的数据加载器，支持渐进式数据加载和分批处理"""
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        
        # 确定使用哪种数据加载器
        if isinstance(dataset, IterableDataset):
            self.loader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=False,  # 禁用内存锁定以节省内存
                drop_last=False
            )
        else:
            # 对于常规Dataset，使用BatchSampler控制批次大小
            sampler = SequentialSampler(dataset)
            if shuffle:
                sampler = torch.utils.data.RandomSampler(dataset)
                
            self.loader = DataLoader(
                dataset,
                batch_sampler=BatchSampler(sampler, batch_size=batch_size, drop_last=False),
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),  # 只在GPU可用时启用
                persistent_workers=False  # 避免持久化工作进程以节省内存
            )
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self):
        if isinstance(self.dataset, IterableDataset):
            # 对于IterableDataset，无法精确知道长度
            # 我们可以返回一个估计值或0
            return 0
        return len(self.loader)

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

# RGCN模型实现 - 内存优化版本
class RGCNKG(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim=200, hidden_dim=100, num_bases=30, chunk_size=500, low_memory_mode=False):
        super(RGCNKG, self).__init__()
        
        # 配置参数
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size  # 内存优化的分块大小
        self.low_memory_mode = low_memory_mode  # 低内存模式开关
        
        # 实体和关系嵌入层 - 使用较小的初始嵌入以节省内存
        # 如果是低内存模式，使用半精度浮点数
        dtype = torch.float16 if low_memory_mode and torch.cuda.is_available() else torch.float32
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim, dtype=dtype)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim, dtype=dtype)
        
        # RGCN卷积层 - 使用基分解来减少参数数量
        self.conv1 = RGCNConv(embedding_dim, hidden_dim, num_relations=num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_dim, embedding_dim, num_relations=num_relations * 2, num_bases=num_bases)
        
        # 输出层用于评分函数 - 简化网络结构以减少内存使用
        self.score_layer = nn.Linear(embedding_dim * 3, 1)
        
        # 初始化权重 - 使用方差更小的初始化方法
        nn.init.normal_(self.entity_embeddings.weight.data, std=0.01)
        nn.init.normal_(self.relation_embeddings.weight.data, std=0.01)
        nn.init.xavier_uniform_(self.score_layer.weight)
        nn.init.zeros_(self.score_layer.bias)
        
        # 注册缓冲区以存储中间结果，避免重复分配
        self.register_buffer('edge_index_cache', None)
        self.register_buffer('edge_type_cache', None)
        
    def memory_efficient_forward(self, data, chunk_size=None, recompute_graph=False):
        """内存高效的正向传播，使用更精细的分块策略"""
        # 使用模型自己的实体嵌入作为输入特征
        x = self.entity_embeddings.weight
        device = x.device
        
        # 缓存图结构以避免重复传输
        if self.edge_index_cache is None or recompute_graph:
            self.edge_index_cache = data.edge_index.to(device)
            self.edge_type_cache = data.edge_type.to(device)
        
        edge_index = self.edge_index_cache
        edge_type = self.edge_type_cache
        
        # 使用实例参数或默认参数
        current_chunk_size = chunk_size or self.chunk_size
        num_nodes = x.size(0)
        
        # 如果是低内存模式，进一步减小分块大小
        if self.low_memory_mode:
            current_chunk_size = max(1, current_chunk_size // 2)
        
        # 预分配结果张量
        h = torch.zeros_like(x)
        
        # 为每个节点分块计算
        for i in range(0, num_nodes, current_chunk_size):
            end_idx = min(i + current_chunk_size, num_nodes)
            chunk_size_actual = end_idx - i
            
            if chunk_size_actual == 0:
                continue
            
            # 仅处理当前块中的节点，而不是整个图
            chunk_nodes = torch.arange(i, end_idx, device=device)
            
            # 获取与当前块节点相关的边（入边和出边）
            in_mask = torch.isin(edge_index[1], chunk_nodes)
            out_mask = torch.isin(edge_index[0], chunk_nodes)
            edge_mask = in_mask | out_mask
            
            if edge_mask.sum() > 0:
                # 提取相关的边和边类型
                sub_edge_index = edge_index[:, edge_mask]
                sub_edge_type = edge_type[edge_mask]
                
                # 使用原始嵌入的副本
                sub_x = x.clone()
                
                # 应用卷积层，但只保留当前块的结果
                with torch.no_grad():
                    temp_h = self.conv1(sub_x, sub_edge_index, sub_edge_type)
                    temp_h = F.relu(temp_h)
                    temp_h = self.conv2(temp_h, sub_edge_index, sub_edge_type)
                
                # 只保留当前块的结果
                h[i:end_idx] = temp_h[i:end_idx].detach()
            else:
                # 如果没有边，使用原始嵌入
                h[i:end_idx] = x[i:end_idx]
            
            # 每处理几个块后进行一次内存清理
            if (i // current_chunk_size) % 5 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        return h
        
    def forward(self, data):
        # 对于大图，强制使用内存高效的正向传播
        h = self.memory_efficient_forward(data)
        return h
        
    def get_entity_embeddings(self, dtype=None):
        """返回实体嵌入（用于预测），支持类型转换以节省内存"""
        embeddings = self.entity_embeddings.weight
        if dtype is not None:
            embeddings = embeddings.to(dtype)
        return embeddings
        
    def get_relation_embeddings(self, dtype=None):
        """返回关系嵌入，支持类型转换以节省内存"""
        embeddings = self.relation_embeddings.weight
        if dtype is not None:
            embeddings = embeddings.to(dtype)
        return embeddings
        
    @memory_monitor
    def score_triple(self, h, r, t):
        """评分函数：计算三元组的得分"""
        # 避免创建过多临时张量
        combined = torch.cat([h, r, t], dim=1)
        score = self.score_layer(combined)
        return score
        
    @memory_monitor
    def score_triple_batch(self, h_ids, r_ids, t_ids, entity_embeddings=None):
        """批量计算三元组得分，避免重复计算嵌入"""
        # 如果提供了预先计算的嵌入，则使用它们
        if entity_embeddings is not None:
            h = entity_embeddings[h_ids]
            t = entity_embeddings[t_ids]
        else:
            h = self.entity_embeddings(h_ids)
            t = self.entity_embeddings(t_ids)
        
        r = self.relation_embeddings(r_ids)
        return self.score_triple(h, r, t)
        
    def clear_caches(self):
        """清除所有缓存以释放内存"""
        self.edge_index_cache = None
        self.edge_type_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

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

# 训练函数 - 内存优化版本
@memory_monitor
@release_memory
def train_model(model, train_dataset, graph_data, mapper, device):
    # 创建自定义的collate_fn函数用于批量处理数据
    def custom_collate(batch):
        return collate_fn(batch, mapper)
    
    # 创建内存优化的数据加载器
    train_loader = MemoryEfficientDataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=custom_collate,
        pin_memory=False,  # 禁用内存锁定以减少内存占用
        num_workers=0 if torch.cuda.is_available() else 2,  # GPU模式下不使用多进程
        low_memory=True
    )
    
    # 定义优化器 - 使用AdamW并启用权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_FACTOR)
    
    # 启用自动混合精度训练 - 兼容旧版本PyTorch
    scaler = None
    if torch.cuda.is_available() and MIXED_PRECISION:
        try:
            # 尝试使用旧版API（PyTorch < 1.10）
            scaler = torch.cuda.amp.GradScaler()
        except AttributeError:
            try:
                # 尝试使用新版API（PyTorch >= 1.10）
                scaler = torch.amp.GradScaler('cuda', enabled=True)
            except AttributeError:
                print("警告: 混合精度训练不可用，将使用常规精度训练")
                MIXED_PRECISION = False
    
    # 将图数据移至设备
    graph_data = graph_data.to(device)
    
    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} 训练中...")
        
        # 预先清零梯度
        optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True更高效
        
        # 定期清理内存和缓存
        if epoch > 0:
            model.clear_caches()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 获取更新后的实体嵌入（每个epoch只计算一次）
        with torch.no_grad():
            updated_embeddings = model(graph_data)
        
        for batch_idx, batch in enumerate(train_loader):
            h_ids, r_ids, t_ids = batch
            
            # 转移到设备上，使用非阻塞传输
            h_ids = h_ids.to(device, non_blocking=True)
            r_ids = r_ids.to(device, non_blocking=True)
            t_ids = t_ids.to(device, non_blocking=True)
            
            # 使用更内存高效的方式生成负样本
            batch_size = h_ids.size(0)
            
            # 批次负样本分块大小
            neg_chunk_size = min(8, NEGATIVE_SAMPLES)  # 减小负样本计算的分块大小
            total_neg_scores = torch.zeros(batch_size, NEGATIVE_SAMPLES, device=device)
            
            # 分块处理负样本，减少内存峰值使用
            for neg_start in range(0, NEGATIVE_SAMPLES, neg_chunk_size):
                neg_end = min(neg_start + neg_chunk_size, NEGATIVE_SAMPLES)
                current_neg_size = neg_end - neg_start
                
                # 为当前块生成负样本ID
                t_neg_ids_chunk = torch.randint(0, mapper.entity_count, 
                                             (batch_size, current_neg_size), 
                                             device=device, dtype=torch.long)
                
                # 确保负样本不等于正样本
                pos_tile = t_ids.unsqueeze(1).expand(-1, current_neg_size)
                mask = (t_neg_ids_chunk == pos_tile)
                
                # 优化负样本生成，避免while循环
                if mask.any():
                    # 一次性生成足够的替换ID
                    num_replacements = mask.sum().item()
                    if num_replacements > 0:
                        # 生成2倍于所需的候选ID以增加找到有效ID的概率
                        candidate_ids = torch.randint(0, mapper.entity_count, 
                                                    (num_replacements * 2,), 
                                                    device=device, dtype=torch.long)
                        # 过滤掉与正样本相同的ID
                        valid_ids = candidate_ids[candidate_ids != pos_tile[mask].repeat(2)[:num_replacements * 2]]
                        # 确保有足够的有效ID
                        while len(valid_ids) < num_replacements:
                            additional_ids = torch.randint(0, mapper.entity_count, 
                                                         (num_replacements - len(valid_ids),), 
                                                         device=device, dtype=torch.long)
                            valid_ids = torch.cat([valid_ids, additional_ids])[:num_replacements]
                        
                        # 替换无效的负样本ID
                        t_neg_ids_chunk[mask] = valid_ids
                
                # 使用自动混合精度进行前向传播
                if torch.cuda.is_available() and MIXED_PRECISION and scaler is not None:
                    try:
                        with torch.cuda.amp.autocast():
                            # 获取头实体、关系和尾实体的嵌入（使用预计算的更新嵌入）
                            h_emb = updated_embeddings[h_ids]
                            r_emb = model.relation_embeddings(r_ids)
                            t_emb = updated_embeddings[t_ids]
                            
                            # 获取负样本尾实体的嵌入
                            t_neg_emb_chunk = updated_embeddings[t_neg_ids_chunk]
                            
                            # 计算正样本得分（只在第一个块计算一次）
                            if neg_start == 0:
                                pos_score = model.score_triple(h_emb, r_emb, t_emb)
                            
                            # 计算当前块负样本得分
                            h_emb_expanded = h_emb.unsqueeze(1).expand(-1, current_neg_size, -1)
                            r_emb_expanded = r_emb.unsqueeze(1).expand(-1, current_neg_size, -1)
                            
                            # 向量化计算当前块的负样本得分
                            neg_scores_chunk = torch.zeros(batch_size, current_neg_size, device=device)
                            for i in range(current_neg_size):
                                neg_scores_chunk[:, i] = model.score_triple(
                                    h_emb_expanded[:, i], r_emb_expanded[:, i], t_neg_emb_chunk[:, i]
                                ).squeeze(1)
                    except AttributeError:
                        with torch.amp.autocast('cuda', enabled=True):
                            # 代码与上面相同，但使用新版API
                            h_emb = updated_embeddings[h_ids]
                            r_emb = model.relation_embeddings(r_ids)
                            t_emb = updated_embeddings[t_ids]
                            
                            t_neg_emb_chunk = updated_embeddings[t_neg_ids_chunk]
                            
                            if neg_start == 0:
                                pos_score = model.score_triple(h_emb, r_emb, t_emb)
                            
                            h_emb_expanded = h_emb.unsqueeze(1).expand(-1, current_neg_size, -1)
                            r_emb_expanded = r_emb.unsqueeze(1).expand(-1, current_neg_size, -1)
                            
                            neg_scores_chunk = torch.zeros(batch_size, current_neg_size, device=device)
                            for i in range(current_neg_size):
                                neg_scores_chunk[:, i] = model.score_triple(
                                    h_emb_expanded[:, i], r_emb_expanded[:, i], t_neg_emb_chunk[:, i]
                                ).squeeze(1)
                else:
                    # 不使用混合精度训练
                    h_emb = updated_embeddings[h_ids]
                    r_emb = model.relation_embeddings(r_ids)
                    t_emb = updated_embeddings[t_ids]
                    
                    t_neg_emb_chunk = updated_embeddings[t_neg_ids_chunk]
                    
                    if neg_start == 0:
                        pos_score = model.score_triple(h_emb, r_emb, t_emb)
                    
                    h_emb_expanded = h_emb.unsqueeze(1).expand(-1, current_neg_size, -1)
                    r_emb_expanded = r_emb.unsqueeze(1).expand(-1, current_neg_size, -1)
                    
                    neg_scores_chunk = torch.zeros(batch_size, current_neg_size, device=device)
                    for i in range(current_neg_size):
                        neg_scores_chunk[:, i] = model.score_triple(
                            h_emb_expanded[:, i], r_emb_expanded[:, i], t_neg_emb_chunk[:, i]
                        ).squeeze(1)
                
                # 保存当前块的负样本得分
                total_neg_scores[:, neg_start:neg_end] = neg_scores_chunk
                
                # 清理当前块的临时变量
                del t_neg_ids_chunk, t_neg_emb_chunk, h_emb_expanded, r_emb_expanded, neg_scores_chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 计算最终损失
            pos_score_expanded = pos_score.expand_as(total_neg_scores)
            individual_loss = torch.relu(total_neg_scores - pos_score_expanded + 1.0)
            loss = torch.mean(individual_loss) * batch_size
            
            # 梯度累积
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            if torch.cuda.is_available() and MIXED_PRECISION and scaler is not None:
                # 混合精度下的反向传播
                scaler.scale(loss).backward()
                
                # 每GRADIENT_ACCUMULATION_STEPS步更新一次参数
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                    # 梯度裁剪以避免梯度爆炸
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True更高效
            else:
                # 常规反向传播
                loss.backward()
                
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # 每处理几个批次后进行内存清理
            if (batch_idx + 1) % 10 == 0:
                del h_emb, r_emb, t_emb, pos_score, total_neg_scores
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                progress = (batch_idx + 1) / len(train_loader) * 100
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f} - Progress: {progress:.1f}%", end='\r')
        
        # 更新学习率
        scheduler.step()
        
        # 彻底清理内存
        del updated_embeddings
        model.clear_caches()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # 打印平均损失和训练速度
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1}/{EPOCHS} - Average Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f} - Time: {epoch_time:.2f}s")

# 评测函数 - 评估RGCN模型在开发集上的表现 (超内存优化版)
@memory_monitor
@release_memory
def evaluate(model, dev_dataset, graph_data, mapper, device, batch_size=32, max_entities_per_batch=1000):
    model.eval()
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_10 = 0
    mrr_score = 0
    total_count = 0
    
    total_dev_triples = len(dev_dataset.triples)
    print(f"\n开始在开发集上评估模型...")
    print(f"将评估 {total_dev_triples} 个三元组")
    
    # 进一步减小实体批次大小以控制显存
    max_entities_per_batch = min(1000, mapper.entity_count)
    
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
            
            # 转换为ID和张量 - 使用更内存高效的方式
            h_ids = torch.tensor([mapper.entity_to_id[h] for h in h_list], device=device)
            r_ids = torch.tensor([mapper.relation_to_id[r] for r in r_list], device=device)
            t_ids = torch.tensor([mapper.entity_to_id[t] for t in t_list], device=device)
            
            # 获取头实体和关系的嵌入
            h_emb = updated_embeddings[h_ids]  # [current_batch_size, embedding_dim]
            r_emb = model.relation_embeddings(r_ids)  # [current_batch_size, embedding_dim]
            
            # 不预分配大的得分矩阵，而是使用滚动计算
            current_ranks = torch.zeros(current_batch_size, device=device, dtype=torch.long)
            
            # 分块计算得分 - 超内存优化实现
            for entity_start in range(0, mapper.entity_count, max_entities_per_batch):
                entity_end = min(entity_start + max_entities_per_batch, mapper.entity_count)
                entity_chunk_size = entity_end - entity_start
                
                # 获取当前块的实体嵌入
                entity_ids = torch.arange(entity_start, entity_end, device=device)
                entity_embeddings = updated_embeddings[entity_ids]  # [entity_chunk_size, embedding_dim]
                
                # 向量化计算当前块的得分 - 使用更小的子批次
                chunk_scores = torch.zeros(current_batch_size, entity_chunk_size, device=device)
                
                # 进一步减小子批次大小
                sub_batch_size = min(16, current_batch_size)
                for sub_start in range(0, current_batch_size, sub_batch_size):
                    sub_end = min(sub_start + sub_batch_size, current_batch_size)
                    
                    # 获取子批次的嵌入
                    sub_h_emb = h_emb[sub_start:sub_end]
                    sub_r_emb = r_emb[sub_start:sub_end]
                    
                    # 不使用广播，而是使用score_triple_batch函数
                    for i in range(entity_chunk_size):
                        sub_chunk_scores = model.score_triple_batch(
                            sub_h_emb,
                            sub_r_emb,
                            entity_embeddings[i].unsqueeze(0).expand(sub_end - sub_start, -1)
                        )
                        chunk_scores[sub_start:sub_end, i] = sub_chunk_scores.squeeze(1)
                
                # 计算当前块中比正确实体得分高的实体数量
                for i in range(current_batch_size):
                    t_idx = t_ids[i].item()
                    # 检查正确实体是否在当前块中
                    if entity_start <= t_idx < entity_end:
                        correct_score = chunk_scores[i, t_idx - entity_start]
                        # 计算当前块中得分高于正确实体的数量（不包括正确实体自己）
                        better_count = (chunk_scores[i] > correct_score).sum().item()
                        current_ranks[i] += better_count
                    else:
                        # 正确实体不在当前块中，计算所有得分高于正确实体的数量
                        correct_entity_emb = updated_embeddings[t_idx].unsqueeze(0)
                        correct_score = model.score_triple(sub_h_emb[0].unsqueeze(0), 
                                                          sub_r_emb[0].unsqueeze(0), 
                                                          correct_entity_emb).item()
                        del correct_entity_emb
                        
                        # 计算当前块中得分高于正确实体的数量
                        better_count = (chunk_scores[i] > correct_score).sum().item()
                        current_ranks[i] += better_count
                
                # 立即清理中间变量，释放显存
                del entity_embeddings, chunk_scores
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 排名从1开始
            ranks = current_ranks + 1
            
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
            
            # 每批次处理完成后立即清理显存
            del h_emb, r_emb, current_ranks, ranks, h_ids, r_ids, t_ids
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print()  # 确保进度条完成后换行
        
        # 清理所有剩余的显存和变量
        del updated_embeddings
        model.clear_caches()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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

# 预测函数 - 超内存优化版本
@memory_monitor
@release_memory
def predict_tail_entities(model, test_dataset, graph_data, mapper, device, max_head_entities=None, batch_size=32, max_entities_per_batch=1000):
    model.eval()
    results = []
    total_test_triples = len(test_dataset.triples)
    
    process_count = total_test_triples if max_head_entities is None else min(max_head_entities, total_test_triples)
    print(f"\n开始预测尾实体...")
    print(f"将处理 {process_count} 个头实体/关系对")
    
    # 进一步减小实体批次大小以控制显存
    max_entities_per_batch = min(1000, mapper.entity_count)  # 限制最大实体数，防止显存溢出
    sub_batch_size = 8  # 更小的子批次大小，进一步减少内存使用
    
    with torch.no_grad():
        # 获取更新后的实体嵌入
        updated_embeddings = model(graph_data)
        
        # 创建批次进行处理
        for batch_start in range(0, process_count, batch_size):
            batch_end = min(batch_start + batch_size, process_count)
            batch_triples = test_dataset.triples[batch_start:batch_end]
            current_batch_size = len(batch_triples)
            
            # 获取批次中的头实体和关系
            h_list = [h for h, _, _ in batch_triples]
            r_list = [r for _, r, _ in batch_triples]
            
            # 转换为ID和张量 - 更内存高效的方式
            h_ids = torch.tensor([mapper.entity_to_id[h] for h in h_list], device=device)
            r_ids = torch.tensor([mapper.relation_to_id[r] for r in r_list], device=device)
            
            # 获取嵌入向量
            h_emb = updated_embeddings[h_ids]  # [batch_size, embedding_dim]
            r_emb = model.relation_embeddings(r_ids)  # [batch_size, embedding_dim]
            
            # 为每个样本存储top-k结果，而不是存储所有得分
            batch_top_k_scores = torch.zeros(current_batch_size, 10, device=device)
            batch_top_k_indices = torch.zeros(current_batch_size, 10, device=device, dtype=torch.long)
            
            # 初始化top-k结果为负无穷
            batch_top_k_scores.fill_(-float('inf'))
            
            # 分块计算得分 - 超内存优化实现
            for entity_start in range(0, mapper.entity_count, max_entities_per_batch):
                entity_end = min(entity_start + max_entities_per_batch, mapper.entity_count)
                entity_chunk_size = entity_end - entity_start
                
                # 获取当前块的实体嵌入
                entity_ids = torch.arange(entity_start, entity_end, device=device)
                entity_embeddings = updated_embeddings[entity_ids]  # [chunk_size, embedding_dim]
                
                # 使用更小的子批次处理
                for sub_batch_start in range(0, current_batch_size, sub_batch_size):
                    sub_batch_end = min(sub_batch_start + sub_batch_size, current_batch_size)
                    sub_batch_size_actual = sub_batch_end - sub_batch_start
                    
                    # 获取子批次的嵌入
                    sub_h_emb = h_emb[sub_batch_start:sub_batch_end]
                    sub_r_emb = r_emb[sub_batch_start:sub_batch_end]
                    
                    # 初始化当前子批次的top-k结果
                    sub_top_k_scores = torch.zeros(sub_batch_size_actual, 10, device=device)
                    sub_top_k_indices = torch.zeros(sub_batch_size_actual, 10, device=device, dtype=torch.long)
                    sub_top_k_scores.fill_(-float('inf'))
                    
                    # 逐实体计算得分，避免大矩阵计算
                    for i in range(entity_chunk_size):
                        # 计算当前实体作为尾实体的得分
                        current_entity_emb = entity_embeddings[i].unsqueeze(0).expand(sub_batch_size_actual, -1)
                        scores = model.score_triple_batch(sub_h_emb, sub_r_emb, current_entity_emb).squeeze(1)
                        
                        # 对每个样本更新top-k结果
                        for j in range(sub_batch_size_actual):
                            current_score = scores[j]
                            current_entity_id = entity_start + i
                            
                            # 更新top-k结果
                            if current_score > sub_top_k_scores[j, -1]:
                                # 找到插入位置
                                insert_pos = (sub_top_k_scores[j] < current_score).sum()
                                # 移动元素并插入新元素
                                sub_top_k_scores[j, insert_pos+1:] = sub_top_k_scores[j, insert_pos:-1]
                                sub_top_k_indices[j, insert_pos+1:] = sub_top_k_indices[j, insert_pos:-1]
                                sub_top_k_scores[j, insert_pos] = current_score
                                sub_top_k_indices[j, insert_pos] = current_entity_id
                        
                        # 清理临时变量
                        del current_entity_emb, scores
                    
                    # 将子批次的top-k结果与全局batch结果合并
                    for j in range(sub_batch_size_actual):
                        global_j = sub_batch_start + j
                        # 合并两个top-k列表
                        merged_scores = torch.cat([batch_top_k_scores[global_j], sub_top_k_scores[j]])
                        merged_indices = torch.cat([batch_top_k_indices[global_j], sub_top_k_indices[j]])
                        
                        # 排序并取top-k
                        sorted_indices = torch.argsort(merged_scores, descending=True)[:10]
                        batch_top_k_scores[global_j] = merged_scores[sorted_indices]
                        batch_top_k_indices[global_j] = merged_indices[sorted_indices]
                        
                        # 清理临时变量
                        del merged_scores, merged_indices, sorted_indices
                
                # 立即清理当前块的变量
                del entity_embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 将预测结果转换为实体标签并存储
            for i in range(current_batch_size):
                h = h_list[i]
                r = r_list[i]
                # 获取top-10实体ID
                top_k_entity_ids = batch_top_k_indices[i].tolist()
                # 转换为实体标签
                top_k_entities = [mapper.id_to_entity[entity_id] for entity_id in top_k_entity_ids]
                results.append((h, r, top_k_entities))
            
            # 显示进度
            progress = min(batch_end, process_count) / process_count * 100
            bar_length = 50
            filled_length = int(bar_length * progress // 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"\r  预测进度: |{bar}| {progress:.1f}% ({min(batch_end, process_count)}/{process_count})", end='')
            
            # 清理当前批次的变量
            del h_emb, r_emb, h_ids, r_ids
            del batch_top_k_scores, batch_top_k_indices
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print()  # 确保进度条完成后换行
        
        # 清理所有变量
        del updated_embeddings
        model.clear_caches()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    return results
# 主函数 - 内存优化版
@memory_monitor
def main():
    # 全局内存优化设置
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 设置CUDA优化选项
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False  # 关闭benchmark以减少内存波动
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = False  # 禁用TF32以减少内存使用
        torch.backends.cudnn.allow_tf32 = False
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 内存监控
    if torch.cuda.is_available():
        print(f"初始GPU内存占用: {torch.cuda.memory_allocated()/1024/1024:.2f} MB")
    
    # 加载数据集 - 使用流式加载减少内存占用
    print("正在加载数据集...")
    train_dataset = StreamingKnowledgeGraphDataset(TRAIN_FILE_PATH, max_lines=MAX_LINES)
    test_dataset = StreamingKnowledgeGraphDataset(TEST_FILE_PATH, is_test=True)
    dev_dataset = StreamingKnowledgeGraphDataset(DEV_FILE_PATH, is_dev=True)  # 加载开发集
    
    print(f"训练数据大小: {len(train_dataset)}")
    print(f"测试数据大小: {len(test_dataset)}")
    print(f"开发数据大小: {len(dev_dataset)}")
    
    # 构建实体和关系映射
    print("正在构建实体和关系映射...")
    mapper = EntityRelationMapper()
    mapper.build_mappings(train_dataset, test_dataset, dev_dataset)
    
    print(f"实体数量: {mapper.entity_count}")
    print(f"关系数量: {mapper.relation_count}")
    
    # 释放数据集原始数据，只保留必要的映射
    train_dataset.clear_raw_data()
    test_dataset.clear_raw_data()
    dev_dataset.clear_raw_data()
    
    # 创建图数据
    print("正在创建图数据结构...")
    graph_data = create_graph_data(train_dataset, mapper)
    
    # 创建RGCN模型 - 低内存模式
    print("正在创建RGCN模型...")
    model = RGCNKG(
        mapper.entity_count, 
        mapper.relation_count, 
        EMBEDDING_DIM, 
        num_bases=30,
        low_memory_mode=True  # 启用低内存模式
    )
    model.to(device)
    
    # 训练模型 - 使用更内存高效的参数
    print("开始训练模型...")
    train_model(model, train_dataset, graph_data, mapper, device)
    
    # 训练完成后清理数据集
    del train_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 在开发集上评估模型
    print("\n训练完成，开始在开发集上评估模型性能...")
    evaluate(model, dev_dataset, graph_data, mapper, device)
    
    # 深度清理训练后的内存
    print("\n深度清理内存...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # 收集跨进程内存
    gc.collect()
    model.clear_caches()
    
    # 预测尾实体 - 使用超内存优化设置
    print("\n开始预测尾实体（超内存优化模式）...")
    predict_tail_entities(
        model, 
        test_dataset, 
        graph_data, 
        mapper, 
        device, 
        MAX_HEAD_ENTITIES,
        batch_size=PREDICT_BATCH_SIZE,  # 使用全局配置的批次大小
        max_entities_per_batch=MAX_ENTITIES_PER_BATCH  # 使用全局配置的实体分块大小
    )
    
    # 最终清理
    print("\n最终清理内存...")
    del model, graph_data, mapper, test_dataset, dev_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    
    print("任务完成！")

if __name__ == "__main__":
    main()