scheme_type = "s14"
# -*- coding: utf-8 -*-
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os, tqdm
import logging

# 禁用 transformers 冗余日志
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

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
EMBEDDING_DIM = 32
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
    def __init__(self, base_model, hidden_size=1024, embed_dim=100):
        super().__init__()
        self.base_model = base_model
        self.embedding_head = torch.nn.Linear(hidden_size, embed_dim)
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # output_hidden_states=True,  # ⚠️ 必须加！
        )
        # 取 [CLS] 或平均池化
        last_hidden = outputs.hidden_states[-1] # outputs.last_hidden_state  # (B, L, H)
        pooled = last_hidden.mean(dim=1)         # (B, H)
        embed = self.embedding_head(pooled)      # (B, 100)
        return embed

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

# DeepSpeed 初始化
model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    model_parameters=[p for p in model.parameters() if p.requires_grad],
    config=f"ds_config--{scheme_type}.json", #ds_config  # ← 不传 optimizer，由 JSON 定义
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

        # 使用模型生成 embedding（不用于生成文本，而是训练表示）
        outputs = model_engine.module.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            output_hidden_states=True,
        )
        loss = outputs.loss

        # DeepSpeed 处理梯度累积
        model_engine.backward(loss)
        model_engine.step()

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
    all_texts = []

    # 实体
    for e_id, text in entity2text.items():
        all_texts.append(f"实体：{text}")

    # 关系
    for r_id, text in relation2text.items():
        all_texts.append(f"关系：{text}")

    # Tokenize 所有文本
    encoded = tokenizer(
        all_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(model_engine.device)

    dataset = torch.utils.data.TensorDataset(encoded['input_ids'], encoded['attention_mask'])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE*2, shuffle=False)

    embeddings = []
    model_engine.eval()
    with torch.no_grad():
        for input_ids, attention_mask in loader:
            embed = model_engine(input_ids, attention_mask)
            embeddings.append(embed.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings

final_embeddings = extract_all_embeddings()
torch.save(final_embeddings, OUTPUT_EMBEDDING_PATH)
print(f"Embeddings saved to {OUTPUT_EMBEDDING_PATH}, shape: {final_embeddings.shape}")