# OpenBG500基线

[OpenBG](https://kg.alibaba.com/)是开放的数字商业知识图谱，是一个使用统一Schema组织、涵盖产品和消费需求的百万级多模态数据集。OpenBG由阿里巴巴藏经阁团队联合提供，开放的目标是利用开放的商业知识发现社会经济的价值，促进数字商务数字经济等领域的交叉学科研究，服务数字经济健康发展的国家战略需求。

[OpenBG Benchmark](https://tianchi.aliyun.com/dataset/dataDetail?dataId=122271)是一个以OpenBG为基础构建的大规模开放数字商业知识图谱评测基准，包含多个子数据集和子任务。欢迎小伙伴打榜[https://tianchi.aliyun.com/dataset/dataDetail?dataId=122271](https://tianchi.aliyun.com/dataset/dataDetail?dataId=122271)。

OpenBG500包含500个关系，从OpenBG中筛选采样得到。

本仓库是阿里天池[电商知识图谱链接预测挑战赛](https://tianchi.aliyun.com/competition/entrance/532033/introduction)的基线代码，运行代码后生成结果提交。

## 环境配置

使用以下代码进行环境配置
```
git clone --recurse-submodules https://github.com/OpenBGBenchmark/OpenBG500_baselines.git
pip install -r requirements.txt
```

## 数据集

请将天池平台上的数据放置在`./data/`，数据目录如下

```shell
data
 |-- OpenBG500
 |    |-- OpenBG500_train.tsv           # 训练数据
 |    |-- OpenBG500_dev.tsv             # 验证数据
 |    |-- OpenBG500_test.tsv            # 需要预测的数据，选手需为每条记录预测10个尾实体
 |    |-- OpenBG500_entity2text.tsv     # 实体对应文本
 |    |-- OpenBG500_relation2text.tsv 	# 关系对应文本
 |    |-- OpenBG500_example_pred.tsv 	# 提交结果示例
```

数据集统计数据如下：
|    Dataset    |    # Ent   | # Rel |   # Train   |  # Dev  | # Test  |
| ------------- | ---------- | ----- | ----------- | ------- | ------- |
|   OpenBG500   | 249,743    |  500  | 1,242,550   | 5,000   |  5,000  |


### 使用数据

#### 数据集格式

* 三元组数据，tsv格式

```shell
# OpenBG500_train.tsv/OpenBG500_dev.tsv
头实体<\t>关系<\t>尾实体<\n>
```

* 实体/关系对应文本数据，tsv格式

```shell
# OpenBG500_entity2text.tsv/OpenBG500_relation2text.tsv
实体（关系）<\t>实体（关系）对应文本<\n>
```

* 评测相关数据格式

```shell
# OpenBG500_test.tsv，选手需要为每行记录补充10格预测的尾实体，提交格式参照OpenBG500_example_pred.tsv
头实体<\t>关系<\n>

# OpenBG500_example_pred.tsv
头实体<\t>关系<\t>尾实体1<\t>尾实体2<\t>...<\t>尾实体10<\n>
```

#### 查看数据集数据

```
$ head -n 3 OpenBG500_train.tsv
ent_135492      rel_0352        ent_015651
ent_020765      rel_0448        ent_214183
ent_106905      rel_0418        ent_121073
```

#### 使用python读取并转换数据集

1. 读取原始数据：
```python
with open('OpenBG500_train.tsv', 'r') as fp:
    data = fp.readlines()
    train = [line.strip('\n').split('\t') for line in data]
    _ = [print(line) for line in train[:2]]
    # ['ent_135492', 'rel_0352', 'ent_015651']
    # ['ent_020765', 'rel_0448', 'ent_214183']
```

2. 获取实体、关系对应文本字典：`ent2text`和`rel2text`
```python
with open('OpenBG500_entity2text.tsv', 'r') as fp:
    data = fp.readlines()
    lines = [line.strip('\n').split('\t') for line in data]
    _ = [print(line) for line in lines[:2]]
    # ['ent_101705', '短袖T恤']
    # ['ent_116070', '套装']

ent2text = {line[0]: line[1] for line in lines}

with open('OpenBG500_relation2text.tsv', 'r') as fp:
    data = fp.readlines()
    lines = [line.strip().split('\t') for line in data]
    _ = [print(line) for line in lines[:2]]
    # ['rel_0418', '细分市场']
    # ['rel_0290', '关联场景']

rel2text = {line[0]: line[1] for line in lines}
```

3. 数据转换成文本：
```python
train = [[ent2text[line[0]],rel2text[line[1]],ent2text[line[2]]] for line in train]
_ = [print(line) for line in train[:2]]
# ['苦荞茶', '外部材质', '苦荞麦']
# ['精品三姐妹硬糕', '口味', '原味硬糕850克【10包40块糕】']
```

## 如何运行基线

### TransE & TransH & TransE & DistMult & ComplEx

模型参考并修改了[OpenKE](https://github.com/thunlp/OpenKE)中的实现。

- 编译C++代码

```shell
    cd 模型目录
    bash scripts/make.sh
```

- 数据预处理

```shell
    bash scripts/prepro.sh
```

- 训练模型并预测结果，结果保存在`./results/result.tsv`


```shell
    bash scripts/train.sh
```

### TuckER

模型参考并修改了[TuckER](https://github.com/ibalazevic/TuckER)中的实现。

- 数据预处理

```shell
    bash scripts/prepro.sh
```

- 训练模型并预测结果，结果保存在`./results/result.tsv`


```shell
    bash scripts/train.sh
```

## 致谢

此代码参考了以下代码：

- [https://github.com/thunlp/OpenKE](https://github.com/thunlp/OpenKE)
- [https://github.com/ibalazevic/TuckER](https://github.com/ibalazevic/TuckER)
