# -*- coding: utf-8 -*-
"""
数据处理脚本：加载数据集、进行负采样、生成训练所需的表格数据
"""
import os
import random
import pandas as pd
from tqdm import tqdm
import numpy as np

class KGPromptGenerator:
    def __init__(self):
        # 数据集路径配置
        self.train_file = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/originalData/OpenBG500/OpenBG500_train.tsv"
        self.entity_text_file = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/originalData/OpenBG500/OpenBG500_entity2text.tsv"
        self.relation_text_file = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/originalData/OpenBG500/OpenBG500_relation2text.tsv"
        self.output_file = "/Users/minkexiu/Downloads/GitHub/Tianchi_EcommerceKG_mac/preprocessedData/df_prompts.jsonl"
        
        # 负采样参数
        self.num_negatives = 3  # 每个正样本对应的负样本数量
        self.max_retries = 100  # 生成负样本的最大尝试次数
        
        # 初始化映射表
        self.entity2text = {}
        self.relation2text = {}
        self.all_entities = set()
        self.all_entities_list = []  # 预先转换为列表以加速随机选择
        
        # 加载映射表
        self._load_mappings()
        
        # 创建输出目录
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
    def _load_mappings(self):
        """加载实体和关系的中文文本映射"""
        # 加载实体文本映射
        print(f"加载实体文本映射: {self.entity_text_file}")
        if os.path.exists(self.entity_text_file):
            with open(self.entity_text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        entity_id, entity_text = parts[0], parts[1]
                        self.entity2text[entity_id] = entity_text
                        self.all_entities.add(entity_id)
            # 预先转换为列表以加速随机选择
            self.all_entities_list = list(self.all_entities)
        else:
            raise FileNotFoundError(f"实体文本映射文件不存在: {self.entity_text_file}")
        
        # 加载关系文本映射
        print(f"加载关系文本映射: {self.relation_text_file}")
        if os.path.exists(self.relation_text_file):
            with open(self.relation_text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        relation_id, relation_text = parts[0], parts[1]
                        self.relation2text[relation_id] = relation_text
        else:
            raise FileNotFoundError(f"关系文本映射文件不存在: {self.relation_text_file}")
        
        print(f"加载完成: {len(self.entity2text)}个实体, {len(self.relation2text)}个关系")
    
    def load_dataset(self, file_path: str) -> list:
        """加载数据集（三元组）"""
        data = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        head, rel, tail = parts[0], parts[1], parts[2]
                        data.append((head, rel, tail))
        else:
            raise FileNotFoundError(f"数据集文件不存在: {file_path}")
        
        print(f"加载数据集{file_path}完成，共{len(data)}个三元组")
        return data
    
    def generate_prompts(self):
        """生成prompt数据并保存为jsonl文件"""
        # 加载训练数据
        train_triples = self.load_dataset(self.train_file)
        
        # 构建所有正样本的集合，用于检查负样本是否存在
        positive_triples = set(train_triples)
        
        # 存储所有结果
        all_results = []
        
        print("正在生成prompt数据...")
        
        # 处理每个三元组
        for head, rel, tail in tqdm(train_triples, desc="生成样本"):
            # 获取实体和关系的文本
            head_text = self.entity2text.get(head, head)
            rel_text = self.relation2text.get(rel, rel)
            tail_text = self.entity2text.get(tail, tail)
            
            # 生成正例prompt
            instruction = "这是一个尾实体预测的知识图谱任务，这是一个正例"
            input_prompt = f"头实体为{head_text}，关系为{rel_text}"
            output_prompt = f"尾实体为{tail_text}"
            all_results.append({
                "instruction": instruction,
                "input": input_prompt,
                "output": output_prompt
            })
            
            # 生成负样本
            neg_tails = set()  # 使用集合避免重复
            attempts = 0
            max_attempts = self.num_negatives * self.max_retries
            
            while len(neg_tails) < self.num_negatives and attempts < max_attempts:
                neg_tail = random.choice(self.all_entities_list)
                if (head, rel, neg_tail) not in positive_triples:
                    neg_tails.add(neg_tail)
                attempts += 1
            
            # 生成负例prompt
            for neg_tail in neg_tails:
                neg_tail_text = self.entity2text.get(neg_tail, neg_tail)
                instruction = "这是一个尾实体预测的知识图谱任务，这是一个反例"
                input_prompt = f"头实体为{head_text}，关系为{rel_text}"
                output_prompt = f"尾实体为{neg_tail_text}"
                all_results.append({
                    "instruction": instruction,
                    "input": input_prompt,
                    "output": output_prompt
                })
        
        # 创建DataFrame
        df = pd.DataFrame(all_results)
        
        # 保存为jsonl文件
        df.to_json(self.output_file, orient='records', lines=True, force_ascii=False)
        print(f"样本数据已保存至: {self.output_file}")
        print(f"生成的样本数据规模: {len(df)}条")
        
        # 显示一些样本
        print("\n生成的样本数据示例:")
        print(df.head().to_string())

if __name__ == "__main__":
    # 设置随机种子以确保结果可复现
    random.seed(42)
    np.random.seed(42)
    
    # 创建数据处理器实例
    generator = KGPromptGenerator()
    
    # 生成样本数据
    generator.generate_prompts()