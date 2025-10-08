import os
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'
os.environ["WANDB_DISABLED"] = "true"

# 在macOS上禁用DeepSpeed重定向警告
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # 设置调试级别

# 配置为全程使用CPU
import torch
try:
    # 强制使用CPU
    has_cuda = False
    has_mps = False
    device = torch.device("cpu")
    print("将全程使用CPU进行训练")
except Exception as e:
    print(f"配置设备时出错: {e}")
    print("将使用CPU进行训练")
    device = torch.device("cpu")
    has_cuda = False
    has_mps = False

import pandas as pd, openai, random, gc
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

model_path = "/Users/minkexiu/Downloads/HuggingFaceModels"
save_dir = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/trained_models/lora-ckpt/final"

# 训练

# 根据CPU环境调整参数
memory_optim_args = {
    "max_seq_length": 256,            # 减少序列长度节省内存
    "gradient_accumulation_steps": 4,  # 适当的梯度累积步数
    "per_device_train_batch_size": 1,  # CPU上使用较小的批量大小
    "gradient_checkpointing": True,    # 启用梯度检查点节省内存
    "optim": "adamw_torch",          # 使用PyTorch原生优化器
    "fp16": False,                    # CPU不支持混合精度训练
    "bf16": False,                    # CPU不支持混合精度训练
    "gradient_checkpointing_kwargs": {"use_reentrant": False}
}

dataset = load_dataset("json", data_files="/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/preprocessedData/df_prompts.jsonl", split="train")

# 确保数据集不为空
print(f"数据集大小: {len(dataset)}")
if len(dataset) == 0:
    raise ValueError("数据集为空，请检查数据文件路径是否正确")
# dataset = dataset[0:10]  # 调试时可以使用小部分数据

# 查看数据集的前几个样本
sample = dataset[0]
print(f"数据集样本示例: {sample}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B", cache_dir=model_path)
tokenizer.pad_token = tokenizer.eos_token

# 修改数据处理函数，确保正确设置标签
def tokenize(examples):
    texts = [
        f"{inst}\n{inp}\n{out}"
        for inst, inp, out in zip(
            examples["instruction"],
            examples["input"],
            examples["output"]
        )
    ]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=memory_optim_args["max_seq_length"],  # 使用优化的序列长度节省内存
        padding="max_length",  # 使用max_length填充而不是动态填充
        return_tensors="pt"  # 提前返回PyTorch张量，减少内存转换开销
    )
    # 正确设置labels，确保梯度能够流动
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# 检查tokenized后的数据集格式
print(f"Tokenized后的数据格式: {list(dataset.features.keys())}")
print(f"Input IDs示例: {dataset[0]['input_ids'][:10]}...")
print(f"Labels示例: {dataset[0]['labels'][:10]}...")

# 配置量化参数（虽然不使用，但保留代码结构）
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
use_quantization = False  # 不使用量化

# 使用CPU加载模型
model_kwargs = {
    "cache_dir": model_path,
    "torch_dtype": torch.float32,  # CPU使用float32
    "device_map": None,  # 不使用device_map
    "trust_remote_code": True
}

if use_quantization:
    model_kwargs["quantization_config"] = quantization_config

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-1.8B",
    **model_kwargs
)

# 配置LoRA参数
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4, lora_alpha=8, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none"  # 确保偏置不参与LoRA
)

model = get_peft_model(model, lora_config)

# 配置训练参数，适应CPU环境
training_args = TrainingArguments(
    output_dir="./lora-ckpt",
    per_device_train_batch_size=memory_optim_args["per_device_train_batch_size"],
    gradient_accumulation_steps=memory_optim_args["gradient_accumulation_steps"],
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=memory_optim_args.get("fp16", False),
    bf16=memory_optim_args.get("bf16", False),
    optim=memory_optim_args["optim"],
    gradient_checkpointing=memory_optim_args["gradient_checkpointing"],
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
    # 添加以下参数解决梯度问题
    gradient_checkpointing_kwargs=memory_optim_args["gradient_checkpointing_kwargs"],
    # 添加性能监控
    logging_steps=10,
    # 启用梯度裁剪防止梯度爆炸
    max_grad_norm=1.0,
    # 禁用DeepSpeed相关功能
    deepspeed=None,
    # 禁用多进程训练
    dataloader_num_workers=0
)

# 自定义Trainer类以确保梯度正确传播并处理额外参数
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 忽略DeepSpeed可能传递的额外参数
        # 确保inputs包含labels，并且labels是可微分的
        if "labels" not in inputs:
            raise ValueError("输入中缺少labels")
        
        # 确保labels是tensor并且在正确的设备上
        if not isinstance(inputs["labels"], torch.Tensor):
            inputs["labels"] = torch.tensor(inputs["labels"]).to(model.device)
        
        # 清理未使用的变量以释放内存
        gc.collect()
        
        outputs = model(**inputs)
        loss = outputs.loss
        
        # 确保loss是标量并且可以微分
        if loss is None:
            # 如果模型没有自动计算loss，手动计算
            logits = outputs.logits
            labels = inputs["labels"]
            # 使用交叉熵损失
            loss_fct = torch.nn.CrossEntropyLoss()
            # 调整logits和labels的形状以匹配CrossEntropyLoss的要求
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # 定期打印训练状态
        if self.state.global_step % 50 == 0:
            print(f"使用CPU训练中，当前步数: {self.state.global_step}")
        
        return (loss, outputs) if return_outputs else loss

# 将模型移至CPU设备（虽然默认已经在CPU上）
model = model.to(device)
print("模型已移至CPU设备")

# 创建自定义Trainer
trainer = CustomTrainer(model=model, args=training_args, train_dataset=dataset)

trainer.train()
# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)
trainer.save_model(save_dir)