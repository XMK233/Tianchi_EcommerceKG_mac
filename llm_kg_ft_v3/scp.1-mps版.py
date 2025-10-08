import os
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['all_proxy'] = 'socks5://127.0.0.1:7890'
os.environ["WANDB_DISABLED"] = "true"

# # 增加内存碎片整理配置
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 尝试自动检测CUDA路径，如果失败则使用CPU
import torch
try:
    # 尝试导入CUDA相关模块
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print("CUDA可用，将使用GPU进行训练")
        # 打印GPU信息
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name}, 显存: {gpu_memory:.2f}GB")
    else:
        print("CUDA不可用，将使用CPU进行训练")
except:
    has_cuda = False
    print("CUDA不可用，将使用CPU进行训练")

import pandas as pd, openai, random, gc
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

model_path = "/Users/minkexiu/Downloads/HuggingFaceModels"
save_dir = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/trained_models/lora-ckpt/final"

# 训练

memory_optim_args = {
    # 根据GPU显存调整参数
    "max_seq_length": 256,            # 减少序列长度节省内存
    "gradient_accumulation_steps": 8,  # 适当的梯度累积步数，平衡GPU利用率和内存
    "per_device_train_batch_size": 2,  # 增加批量大小提高GPU利用率
    "gradient_checkpointing": True,    # 启用梯度检查点节省显存
    "optim": "paged_adamw_32bit",     # 使用分页优化器
    "fp16": has_cuda,                 # 启用混合精度训练
    "gradient_checkpointing_kwargs": {"use_reentrant": False}
}

dataset = load_dataset("json", data_files="/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/preprocessedData/df_prompts.jsonl", split="train")

# 确保数据集不为空
print(f"数据集大小: {len(dataset)}")
if len(dataset) == 0:
    raise ValueError("数据集为空，请检查数据文件路径是否正确")
# dataset = dataset[0:10]


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

# 配置量化参数，适合16GB显存
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # 使用float16而不是bfloat16，对某些显卡更友好
)
use_quantization = False ## 不要做量化了。

# 使用GPU和量化加载模型
model_kwargs = {
    "cache_dir": model_path,
    "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    "device_map": "auto" if torch.cuda.is_available() else None,
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

# 配置训练参数，平衡显存使用和GPU利用率
training_args = TrainingArguments(
    output_dir="./lora-ckpt",
    per_device_train_batch_size=memory_optim_args["per_device_train_batch_size"],
    gradient_accumulation_steps=memory_optim_args["gradient_accumulation_steps"],
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=memory_optim_args["fp16"],
    bf16=False,
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
    max_grad_norm=1.0
)

# 自定义Trainer类以确保梯度正确传播并处理额外参数
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 忽略DeepSpeed可能传递的额外参数，解决'num_items_in_batch'参数错误
        # 确保inputs包含labels，并且labels是可微分的
        if "labels" not in inputs:
            raise ValueError("输入中缺少labels")
        
        # 确保labels是tensor并且在正确的设备上
        if not isinstance(inputs["labels"], torch.Tensor):
            inputs["labels"] = torch.tensor(inputs["labels"]).to(model.device)
        
        # 清理未使用的变量以释放内存
        gc.collect()
        if has_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
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
        
        # # 定期打印GPU内存使用情况
        # if self.state.global_step % 50 == 0 and has_cuda:
        #     allocated = torch.cuda.memory_allocated() / (1024**3)
        #     reserved = torch.cuda.memory_reserved() / (1024**3)
        #     print(f"GPU内存使用: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
        
        return (loss, outputs) if return_outputs else loss

# 在GPU上使用自定义Trainer
trainer = CustomTrainer(model=model, args=training_args, train_dataset=dataset)

trainer.train()
# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)
trainer.save_model(save_dir)