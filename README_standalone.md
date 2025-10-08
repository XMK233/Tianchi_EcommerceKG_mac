# 知识图谱嵌入预测工具（完全独立实现）

本工具是一个完全独立实现的知识图谱嵌入预测系统，不依赖于任何外部的知识图谱库或框架。它从头实现了TransE算法，用于训练知识图谱嵌入模型并预测最可能的尾实体。

## 功能特点

- **完全独立实现**：不依赖于任何现有的知识图谱库或框架
- **从头实现的TransE算法**：基于经典的TransE知识图谱嵌入方法
- **高效的数据处理**：使用PyTorch的数据加载和批处理功能
- **GPU加速**：自动检测并使用GPU（如果可用）以加速训练
- **可配置的超参数**：支持调整嵌入维度、学习率、批次大小等参数

## 实现原理

TransE是一种经典的知识图谱嵌入方法，其核心思想是：
- 将知识图谱中的实体和关系映射到低维向量空间
- 对于有效的三元组 (h, r, t)，期望满足 h + r ≈ t
- 使用L1或L2范数来衡量 h + r 与 t 之间的距离
- 距离越小，表示三元组越可能有效

## 环境配置

1. 安装必要的Python包：

```bash
pip install -r requirements.txt
```

2. 确保您的系统已安装Python 3.6或更高版本

3. 对于GPU加速，确保已安装兼容的CUDA版本（如适用）

## 使用方法

1. 确保您的数据文件在正确的路径下：
   - 训练数据：`/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_train.tsv`
   - 测试数据：`/mnt/d/forCoding_data/Tianchi_EcommerceKG/originalData/OpenBG500/OpenBG500_test.tsv`

2. 运行脚本：

```bash
python new_kg_predictor.py
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

## 超参数配置

您可以在脚本中调整以下超参数以获得更好的性能：

- `EMBEDDING_DIM`：嵌入维度，默认为200
- `LEARNING_RATE`：学习率，默认为0.01
- `MARGIN`：margin损失中的margin值，默认为1.0
- `BATCH_SIZE`：训练批次大小，默认为1024
- `EPOCHS`：训练轮数，默认为200
- `NEGATIVE_SAMPLES`：每个正样本的负样本数量，默认为5

## 注意事项

1. 训练时间取决于数据量大小、嵌入维度和训练轮数
2. 对于大型知识图谱，可能需要调整批次大小和嵌入维度以适应可用内存
3. 您可以调整超参数以获得更好的预测性能
4. 脚本会自动检测并使用GPU（如果可用）