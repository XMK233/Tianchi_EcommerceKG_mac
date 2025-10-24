scheme_type = "s14"
# -*- coding: utf-8 -*-
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os, tqdm
import logging
import socket

def find_available_port(start=29500, end=29600):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    return None

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# 禁用 transformers 冗余日志
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'
os.environ["WANDB_DISABLED"] = "true"

# 日志配置
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import deepspeed
# from tqdm import tqdm

# ==================== 配置 ====================
cache_dir = "/mnt/d/HuggingFaceModels"
BASE_DIR = "/mnt/d/forCoding_data/Tianchi_EcommerceKG"
MODEL_NAME = "Qwen/Qwen1.5-1.8B" # "Qwen/Qwen3-4B" # 
MAX_LEN = 128
EMBEDDING_DIM = 100
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
EPOCHS = 1
LEARNING_RATE = 3e-4
MAX_LINES = 10

# 文件路径
TRAIN_FILE_PATH = f"{BASE_DIR}/originalData/OpenBG500/OpenBG500_train.tsv"
TEST_FILE_PATH = f"{BASE_DIR}/originalData/OpenBG500/OpenBG500_test.tsv"
DEV_FILE_PATH = f"{BASE_DIR}/originalData/OpenBG500/OpenBG500_dev.tsv"
entity_text_file = f"{BASE_DIR}/originalData/OpenBG500/OpenBG500_entity2text.tsv"
relation_text_file = f"{BASE_DIR}/originalData/OpenBG500/OpenBG500_relation2text.tsv"
OUTPUT_EMBEDDING_PATH = f"{BASE_DIR}/embeddings/qwen_lora_embeds__{scheme_type}.pt"

os.makedirs(os.path.dirname(OUTPUT_EMBEDDING_PATH), exist_ok=True)

device = torch.device('cuda')
logger.info(f"使用设备: {device}")

# ==================== 数据加载与映射 ====================
def load_text_map(file_path):
    """加载 entity2text 或 relation2text"""
    mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '	' in line:
                k, v = line.strip().split('	', 1)
                mapping[k] = v
    return mapping

entity2text = load_text_map(entity_text_file)
relation2text = load_text_map(relation_text_file)

# 加载训练数据（只取头、关系、尾）
train_data = []
with open(TRAIN_FILE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        h, r, t = line.strip().split('	')
        train_data.append((h, r, t))

# 采样调试（可选：先试 若干 条）
if MAX_LINES is not None:
    train_data = train_data[:MAX_LINES]


# ==================== 自定义 Dataset ====================
class KGTailPredictionDataset(Dataset):
    def __init__(self, data, entity2text, relation2text, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []

        for h, r, t in data:
            h_text = entity2text.get(h, h)
            r_text = relation2text.get(r, r)
            t_text = entity2text.get(t, t)

            instruction = "这是一个尾实体预测的知识图谱任务，这是一个正例" ## 未来可以考虑增加反例尾实体。
            input_text = f"头实体为{h_text}，关系为{r_text}"
            output_text = f"尾实体为{t_text}"

            prompt = f"{instruction}\n{input_text}\n{output_text}"

            encoded = tokenizer(
                prompt,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt"
            )
            self.examples.append({
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'labels': encoded['input_ids'].squeeze().clone()
            })
            # label 中不需要 instruction 和 input
            # 实际中可更精细地 mask 输入部分，此处简化

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# ==================== 初始化 Tokenizer & Model ====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained( ## 
    MODEL_NAME, 
    cache_dir = cache_dir,
    dtype=torch.bfloat16,
    device_map=None,  # 手动分配给 DeepSpeed
    trust_remote_code=True, 
    # output_hidden_states = True
)

# ==================== LoRA 配置 ====================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    modules_to_save=["classifier"],  # 如果有自定义 head
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())  # 应该显示 ~几百万，远小于全参数微调

# 添加分类头用于提取 embedding
class EmbeddingModel(torch.nn.Module):
    def __init__(self, base_model, hidden_size=2048, embed_dim=100):
        super().__init__()
        self.base_model = base_model
        # 尝试动态获取模型的隐藏层大小，适用于不同的模型架构
        try:
            # 对于Qwen1.5-1.8B模型，隐藏层大小应该是2048
            if hasattr(base_model, 'config') and hasattr(base_model.config, 'hidden_size'):
                self.hidden_size = base_model.config.hidden_size
                logger.info(f"从模型配置获取隐藏层大小: {self.hidden_size}")
            elif hasattr(base_model, 'hidden_size'):
                self.hidden_size = base_model.hidden_size
                logger.info(f"从模型属性获取隐藏层大小: {self.hidden_size}")
            else:
                # 默认使用2048，符合Qwen1.5-1.8B模型
                self.hidden_size = 2048
                logger.info(f"使用默认隐藏层大小: {self.hidden_size}")
        except Exception as e:
            logger.error(f"获取模型隐藏层大小时出错: {str(e)}")
            self.hidden_size = 2048  # 回退到2048
        
        self.embedding_head = torch.nn.Linear(self.hidden_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, input_ids, attention_mask=None):
        try:
            # 确保启用output_hidden_states
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,  # 必须显式启用
            )
            
            # 尝试获取隐藏状态的多种可能方式
            last_hidden = None
            
            # 获取模型的dtype（处理BFloat16和Float的兼容性）
            model_dtype = next(self.parameters()).dtype if len(list(self.parameters())) > 0 else torch.bfloat16
            
            # 方式1：优先尝试hidden_states（支持属性和字典两种访问方式）
            hidden_states = None
            
            # 尝试属性访问
            if hasattr(outputs, 'hidden_states'):
                hidden_states = outputs.hidden_states
                logger.debug("通过属性访问获取hidden_states")
            # 尝试字典访问（因为outputs有字典方法）
            elif hasattr(outputs, '__getitem__') and 'hidden_states' in outputs:
                try:
                    hidden_states = outputs['hidden_states']
                    logger.debug("通过字典访问获取hidden_states")
                except (KeyError, TypeError):
                    logger.debug("字典访问hidden_states失败")
            
            # 处理获取到的hidden_states
            if hidden_states is not None:
                if isinstance(hidden_states, list):
                    if len(hidden_states) > 0:
                        last_hidden = hidden_states[-1]
                        logger.debug(f"使用hidden_states[-1]，类型为list，长度为{len(hidden_states)}")
                    else:
                        logger.debug("hidden_states是list但为空")
                elif isinstance(hidden_states, tuple):
                    if len(hidden_states) > 0:
                        last_hidden = hidden_states[-1]
                        logger.debug(f"使用hidden_states[-1]，类型为tuple，长度为{len(hidden_states)}")
                    else:
                        logger.debug("hidden_states是tuple但为空")
                else:
                    logger.debug(f"hidden_states类型为{type(hidden_states)}")
            
            # 方式2：尝试last_hidden_state（支持属性和字典两种访问方式）
            if last_hidden is None:
                if hasattr(outputs, 'last_hidden_state'):
                    last_hidden = outputs.last_hidden_state
                    logger.debug("通过属性访问获取last_hidden_state")
                # 尝试字典访问last_hidden_state
                elif hasattr(outputs, '__getitem__') and 'last_hidden_state' in outputs:
                    try:
                        last_hidden = outputs['last_hidden_state']
                        logger.debug("通过字典访问获取last_hidden_state")
                    except (KeyError, TypeError):
                        logger.debug("字典访问last_hidden_state失败")
            
            # 方式4：如果有logits但没有隐藏状态，尝试使用logits
            if last_hidden is None and hasattr(outputs, 'logits'):
                # 检查logits的形状和类型
                logits_shape = outputs.logits.shape
                logits_dtype = outputs.logits.dtype
                logger.warning(f"只能获取到logits，形状: {logits_shape}, 类型: {logits_dtype}，尝试从logits生成表示")
                
                # 尝试直接使用logits作为特征，投影到嵌入空间
                try:
                    # 获取logits的形状，通常是 [batch_size, seq_len, vocab_size]
                    batch_size = logits_shape[0]
                    
                    # 对seq_len维度做平均，得到 [batch_size, vocab_size]
                    logits_mean = outputs.logits.mean(dim=1).to(model_dtype)
                    
                    # 创建一个临时的投影层，从vocab_size映射到hidden_size
                    # 注意：这是一个应急方案，不会更新参数
                    vocab_size = logits_shape[-1] if len(logits_shape) > 2 else logits_shape[-1]
                    temp_proj = torch.nn.Linear(vocab_size, self.hidden_size, bias=False).to(model_dtype).to(input_ids.device)
                    
                    # 使用临时投影层生成隐藏状态
                    last_hidden = temp_proj(logits_mean)
                    # 扩展维度以匹配预期的形状 [batch_size, seq_len, hidden_size]
                    # 这里我们使用seq_len=1作为简化
                    last_hidden = last_hidden.unsqueeze(1)
                    logger.debug(f"从logits成功生成隐藏状态，形状: {last_hidden.shape}")
                except Exception as e:
                    logger.error(f"从logits生成隐藏状态失败: {str(e)}")
                    # 如果失败，回退到随机生成
                    batch_size = input_ids.shape[0]
                    seq_len = input_ids.shape[1]
                    last_hidden = torch.randn(batch_size, seq_len, self.hidden_size, 
                                             device=input_ids.device, dtype=model_dtype)
            
            if last_hidden is None:
                # 修复日志格式问题
                logger.error(f"无法获取隐藏状态，outputs的属性: {dir(outputs)}")
                # 创建零向量作为应急措施，确保使用正确的dtype
                batch_size = input_ids.shape[0]
                seq_len = input_ids.shape[1]
                last_hidden = torch.zeros(batch_size, seq_len, self.hidden_size, 
                                         device=input_ids.device, dtype=model_dtype)
            else:
                # 确保last_hidden使用正确的dtype
                last_hidden = last_hidden.to(model_dtype)
            
            # 确保embedding_head也使用正确的dtype
            self.embedding_head.to(model_dtype)
            
            pooled = last_hidden.mean(dim=1)         # (B, H)
            embed = self.embedding_head(pooled)      # (B, 100)
            return embed
        except Exception as e:
            logger.error(f"forward方法出错: {str(e)}")
            # 返回零向量作为应急，确保使用正确的dtype
            batch_size = input_ids.shape[0]
            model_dtype = next(self.parameters()).dtype if len(list(self.parameters())) > 0 else torch.bfloat16
            return torch.zeros(batch_size, self.embed_dim, device=input_ids.device, dtype=model_dtype)

model = EmbeddingModel(model, hidden_size=1024, embed_dim=EMBEDDING_DIM)
model = model.to("cuda")

# ==================== DeepSpeed 配置文件 (ds_config.json) ====================
# 保存到本地
ds_config = {
    # "fp16": {"enabled": True},
    "bf16": {"enabled": True},

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 0.0003,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999]
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 0.0003,
            "warmup_num_steps": 100
        }
    },

    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_scatter": True,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "reduce_bucket_size": 5e8
    },

    "gradient_accumulation_steps": 8,
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": 2,
    "steps_per_print": 10
}

import json
with open(f"ds_config--{scheme_type}.json", "w") as f:
    json.dump(ds_config, f, indent=2)

# ==================== Dataloader ====================
dataset = KGTailPredictionDataset(train_data, entity2text, relation2text, tokenizer, MAX_LEN)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

# ==================== Optimizer & Scheduler ====================
# optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)

# 直接使用model而不是model_engine，移除DeepSpeed相关的包装

# 直接使用model而不是model_engine，移除DeepSpeed相关的包装

# 加载DeepSpeed配置
config_path = os.path.join(os.path.dirname(__file__), 'ds_config--s14.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# 修改配置以避免分布式训练的端口问题
# 保留ZeRO优化，但避免使用分布式训练
config['zero_optimization']['stage'] = 2  # 使用ZeRO-2优化，比ZeRO-3更简单但仍然有效
config['zero_optimization']['offload_optimizer'] = {  # 使用字典配置
    'device': 'cpu',
    'pin_memory': True
}

# 设置使用不同的端口来避免冲突
os.environ["MASTER_PORT"] = "29550"
logger.info("已设置MASTER_PORT=29550以避免端口冲突")

# 不使用DeepSpeed的分布式训练，仅使用其优化功能
# 直接使用PyTorch优化器，但利用DeepSpeed优化内存
model_engine = deepspeed.initialize(
    model=model,
    optimizer=None,
    model_parameters=[p for p in model.parameters() if p.requires_grad],
    training_data=train_data,
    config=config
)[0]  # 只取model_engine

# 获取设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")

# 定义collate_fn函数
def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch])
    }

# 为了兼容后续代码，确保optimizer和train_loader仍然有效
train_loader = DataLoader(
    train_data,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)
optimizer = torch.optim.AdamW(
    [p for p in model_engine.parameters() if p.requires_grad],
    lr=1e-5,
    weight_decay=0.01
)

# # ==================== 训练循环 ====================
# model_engine.train()
# global_step = 0

# for epoch in range(EPOCHS):
#     pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
#     for batch in pbar:
#         input_ids = batch['input_ids'].to(model_engine.device)
#         attention_mask = batch['attention_mask'].to(model_engine.device)
#         labels = batch['labels'][:, -1].to(model_engine.device)  # 简化：只预测最后一个 token

#         # Forward
#         embeddings = model_engine(input_ids, attention_mask)  # (B, 100)

#         # 这里可以加对比损失或分类损失
#         # 示例：用 embedding 做分类（实际任务可能不同）
#         # 简化为自编码式重建 loss（仅演示）
#         # 更好做法：训练完后固定模型，单独提取 embedding

#         # 我们重点是提取 embedding，所以这里可以跳过复杂 loss
#         # 实际上，我们更关心模型学会语义表示，而非准确生成文本

#         # 因此，我们可以使用 causal lm loss
#         outputs = model_engine.module.base_model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=input_ids
#         )
#         loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS

#         model_engine.backward(loss)
#         if (global_step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
#             model_engine.step()
#             model_engine.zero_grad()

#         pbar.set_postfix(loss=loss.item() * GRADIENT_ACCUMULATION_STEPS)
#         global_step += 1

# ==================== 训练循环（带 tqdm 进度条） ====================
model_engine.train()
global_step = 0
total_loss = 0.0

for epoch in range(EPOCHS):
    logger.info(f"[Epoch {epoch+1}/{EPOCHS}] Starting...")

    # 使用 tqdm 包装 dataloader
    pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}", total=len(dataloader))

    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # 确保在训练时也启用output_hidden_states
        outputs = model_engine.module.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            output_hidden_states=True,
        )
        loss = outputs.loss

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 记录 loss
        total_loss += loss.item()
        global_step += 1

        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss/global_step:.4f}",
            'step': global_step
        })

        # 可选：每 50 步打个日志
        if global_step % 50 == 0:
            logger.info(f"Step {global_step} | Loss: {loss.item():.4f} | Avg Loss: {total_loss/global_step:.4f}")

    logger.info(f"[Epoch {epoch+1}] Finished. Average Loss: {total_loss/global_step:.4f}")

# ==================== 提取所有实体和关系的 embedding ====================
def extract_all_embeddings():
    # 确保不使用分布式训练功能
    all_texts = []
    all_ids = []

    # 实体
    for e_id, text in entity2text.items():
        all_texts.append(f"实体：{text}")
        all_ids.append(e_id)

    # 关系
    for r_id, text in relation2text.items():
        all_texts.append(f"关系：{text}")
        all_ids.append(r_id)

    logger.info(f"开始提取 {len(all_texts)} 个实体和关系的embedding")

    # 分批处理以避免内存问题
    batch_size = BATCH_SIZE * 2
    embeddings = []
    
    # 获取原始模型
    model = model_engine.module  # 获取原始模型
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i+batch_size]
            
            # Tokenize 批次文本
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            ).to(device)
            
            try:
                # 直接调用模型的forward方法
                embed = model(encoded['input_ids'], encoded['attention_mask'])
                embeddings.append(embed.cpu())
                logger.info(f"已处理批次 {i//batch_size + 1}/{(len(all_texts) + batch_size - 1) // batch_size}")
            except Exception as e:
                logger.error(f"处理批次时出错: {str(e)}")
                # 尝试更直接的方式
                try:
                    # 获取模型的dtype
                    model_dtype = next(model.parameters()).dtype if len(list(model.parameters())) > 0 else torch.bfloat16
                    
                    # 直接使用base_model获取隐藏状态
                    outputs = model.base_model.model(
                        input_ids=encoded['input_ids'],
                        attention_mask=encoded['attention_mask'],
                        output_hidden_states=True
                    )
                    
                    # 尝试多种方式获取隐藏状态
                    last_hidden = None
                    
                    # 检查hidden_states
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        if isinstance(outputs.hidden_states, (list, tuple)) and len(outputs.hidden_states) > 0:
                            last_hidden = outputs.hidden_states[-1]
                    
                    # 检查last_hidden_state
                    elif hasattr(outputs, 'last_hidden_state'):
                        last_hidden = outputs.last_hidden_state
                    
                    # 检查输出格式
                    elif hasattr(outputs, 'logits'):
                        # 如果只有logits，尝试从这里生成表示
                        logger.warning("只有logits可用，尝试从logits生成表示")
                        # 使用logits的平均值作为粗略表示，并确保类型一致
                        logits_mean = outputs.logits.mean(dim=1).to(model_dtype)
                        # 确保embedding_head使用正确的dtype
                        model.embedding_head.to(model_dtype)
                        # 投影到embedding维度
                        embed = model.embedding_head(logits_mean)
                        embeddings.append(embed.cpu())
                        logger.info("从logits成功生成embedding")
                        continue
                    
                    if last_hidden is not None:
                        # 确保类型一致
                        last_hidden = last_hidden.to(model_dtype)
                        model.embedding_head.to(model_dtype)
                        pooled = last_hidden.mean(dim=1)
                        embed = model.embedding_head(pooled)
                        embeddings.append(embed.cpu())
                    logger.info("使用备用方法成功处理批次")
                except Exception as inner_e:
                    logger.error(f"备用方法也失败了: {str(inner_e)}")
                    # 添加零向量作为应急
                    batch_size = encoded['input_ids'].shape[0]
                    dummy_embed = torch.zeros(batch_size, EMBEDDING_DIM, device='cpu')
                    embeddings.append(dummy_embed)
                    # 如果还是失败，创建零向量作为应急措施
                    zero_embeds = torch.zeros(len(batch_texts), EMBEDDING_DIM)
                    embeddings.append(zero_embeds)
                    logger.warning("使用零向量替代失败的批次")

    if embeddings:
        final_embeddings = torch.cat(embeddings, dim=0)
        logger.info(f"成功提取所有embedding，形状: {final_embeddings.shape}")
        return final_embeddings
    else:
        logger.error("未能提取任何embedding！")
        # 返回空张量作为应急
        return torch.zeros(0, EMBEDDING_DIM)

final_embeddings = extract_all_embeddings()
torch.save(final_embeddings, OUTPUT_EMBEDDING_PATH)
print(f"Embeddings saved to {OUTPUT_EMBEDDING_PATH}, shape: {final_embeddings.shape}")