import os
import sys
import re
import random
import numpy as np

sys.path.append("./old_code/OpenKE/")
import openke
from openke.config import Trainer
from old_code.TransE.cover import Tester, TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# 设置随机种子以确保结果可复现
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    
set_seed()

# 数据路径
input_train_path = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_train.tsv"
input_test_path = "/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_test.tsv"
output_path = "./temp_data/"
result_path = "./rst.tsv"

# 创建临时数据目录
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 数据预处理函数
def preprocess_data():
    print("正在预处理数据...")
    
    # 读取训练数据
    with open(input_train_path, 'r', encoding='utf-8') as f:
        train_data = f.readlines()
    
    # 读取测试数据
    with open(input_test_path, 'r', encoding='utf-8') as f:
        test_data = f.readlines()
    
    # 构建实体和关系字典
    ent2id = {}
    id2ent = {}
    ent_num = 0
    rel2id = {}
    id2rel = {}
    rel_num = 0
    
    def update_dict(id_to, to_id, num, item):
        if item not in to_id:
            to_id[item] = num
            id_to[num] = item
            return num + 1
        return num
    
    # 处理训练数据中的实体和关系
    for line in train_data:
        h, r, t = line.strip().split('\t')
        ent_num = update_dict(id2ent, ent2id, ent_num, h)
        ent_num = update_dict(id2ent, ent2id, ent_num, t)
        rel_num = update_dict(id2rel, rel2id, rel_num, r)
    
    # 处理测试数据中的实体和关系
    for line in test_data:
        h, r = line.strip().split('\t')
        ent_num = update_dict(id2ent, ent2id, ent_num, h)
        rel_num = update_dict(id2rel, rel2id, rel_num, r)
    
    # 转换训练数据为OpenKE格式
    train_ids = []
    for line in train_data:
        h, r, t = line.strip().split('\t')
        train_ids.append(f"{ent2id[h]} {ent2id[t]} {rel2id[r]}\n")
    
    # 写入预处理后的文件
    with open(os.path.join(output_path, 'entity2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(ent2id)}\n")
        for ent, idx in ent2id.items():
            f.write(f"{ent}\t{idx}\n")
    
    with open(os.path.join(output_path, 'relation2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(rel2id)}\n")
        for rel, idx in rel2id.items():
            f.write(f"{rel}\t{idx}\n")
    
    with open(os.path.join(output_path, 'train2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(train_ids)}\n")
        f.writelines(train_ids)
    
    # 转换测试数据为OpenKE格式
    test_ids = []
    test_pairs = []  # 保存原始的(头实体, 关系)对用于结果生成
    for line in test_data:
        h, r = line.strip().split('\t')
        test_pairs.append((h, r))
        # 为了让OpenKE能处理，添加一个占位符尾实体
        test_ids.append(f"{ent2id[h]} 0 {rel2id[r]}\n")
    
    with open(os.path.join(output_path, 'test2id.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(test_ids)}\n")
        f.writelines(test_ids)
    
    return ent2id, id2ent, rel2id, test_pairs

# 训练模型
def train_model():
    print("正在训练模型...")
    
    # 数据加载器
    train_dataloader = TrainDataLoader(
        in_path=output_path,
        nbatches=100,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0)
    
    # 定义模型
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=200,
        p_norm=1,
        norm_flag=True)
    
    # 定义损失函数
    model = NegativeSampling(
        model=transe,
        loss=MarginLoss(margin=5.0),
        batch_size=train_dataloader.get_batch_size())
    
    # 训练模型
    trainer = Trainer(model=model, data_loader=train_dataloader, train_times=1000, alpha=1.0, use_gpu=True)
    trainer.run()
    
    # 保存模型
    transe.save_checkpoint('./checkpoints/transe_ckpt')
    
    return transe

# 生成预测结果
def generate_predictions(model, ent2id, id2ent, rel2id, test_pairs):
    print("正在生成预测结果...")
    
    # 加载模型
    model.load_checkpoint('./checkpoints/transe_ckpt')
    
    results = []
    
    # 为每个(头实体, 关系)对预测10个最可能的尾实体
    for h, r in test_pairs:
        h_id = ent2id[h]
        r_id = rel2id[r]
        
        # 获取所有实体的嵌入
        ent_embeddings = model.ent_embeddings.weight.data.cpu().numpy()
        rel_embedding = model.rel_embeddings.weight.data[r_id].cpu().numpy()
        h_embedding = model.ent_embeddings.weight.data[h_id].cpu().numpy()
        
        # 计算TransE的得分：||h + r - t||
        t_embeddings = ent_embeddings
        scores = np.sum(np.abs(h_embedding + rel_embedding - t_embeddings), axis=1)
        
        # 获取得分最高的10个尾实体
        top10_indices = np.argsort(scores)[:10]
        top10_ents = [id2ent[idx] for idx in top10_indices]
        
        # 构建结果行
        result_line = [h, r] + top10_ents
        results.append('\t'.join(result_line) + '\n')
    
    # 写入结果文件
    with open(result_path, 'w', encoding='utf-8') as f:
        f.writelines(results)
    
    print(f"结果已保存到 {result_path}")

# 主函数
def main():
    # 预处理数据
    ent2id, id2ent, rel2id, test_pairs = preprocess_data()
    
    # 训练模型
    model = train_model()
    
    # 生成预测结果
    generate_predictions(model, ent2id, id2ent, rel2id, test_pairs)
    
    print("任务完成！")

if __name__ == "__main__":
    main()