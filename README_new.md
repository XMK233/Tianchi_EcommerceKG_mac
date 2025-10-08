# 知识图谱嵌入预测工具

本工具基于TransE模型实现知识图谱嵌入，并为给定的(头实体, 关系)二元组预测最可能的尾实体。

## 功能介绍

1. 数据预处理：将用户提供的TSV格式数据转换为OpenKE框架所需的格式
2. 模型训练：使用TransE算法训练知识图谱嵌入模型
3. 结果预测：为每个测试二元组预测10个最可能的尾实体
4. 结果输出：生成符合要求格式的TSV结果文件

## 环境配置

1. 首先安装必要的依赖：

```bash
pip install -r requirements.txt
```

2. 确保已安装PyTorch和CUDA（如有GPU）

## 使用方法

1. 确保您的数据文件路径正确：
   - 训练数据路径：`/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_train.tsv`
   - 测试数据路径：`/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_test.tsv`

2. 运行脚本：

```bash
python kg_prediction.py
```

3. 结果将保存在当前目录下的`rst.tsv`文件中

## 输出格式

结果文件`rst.tsv`的格式如下：

```
头实体id\t关系id\t尾实体1-id\t尾实体2-id\t...\t尾实体10-id\n
例如:

ent_013469\trel_0006\tent_012552\tent_xxxxxx\tent_xxxxxx\tent_xxxxxx\tent_xxxxxx\tent_xxxxxx\tent_xxxxxx\tent_xxxxxx\tent_xxxxxx\tent_xxxxxx
ent_025701\trel_0041\tent_xxxxxx\tent_023834\tent_xxxxxx\tent_xxxxxx\tent_xxxxxx\tent_xxxxxx\tent_xxxxxx\tent_xxxxxx\tent_xxxxxx\tent_xxxxxx
```

## 注意事项

1. 脚本会在当前目录创建`temp_data`和`checkpoints`文件夹用于存储中间数据和模型
2. 训练过程中会使用GPU（如果可用）以加速训练
3. 您可以根据需要调整脚本中的参数，如嵌入维度、训练轮数等
4. 如果数据量很大，训练时间可能会较长