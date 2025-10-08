# 修复说明：张量维度不匹配问题

## 问题描述

在运行脚本时遇到了以下错误：

```
RuntimeError: The size of tensor a (1024) must match the size of tensor b (5) at non-singleton dimension 1
```

这个错误发生在训练模型时计算损失函数的部分。具体来说，是`pos_score`和`neg_score`的维度不匹配导致的。

## 问题原因

在`train_model`函数中，当计算损失时，`pos_score`的形状是`[batch_size]`（例如1024），而`neg_score`的形状是`[batch_size, NEGATIVE_SAMPLES]`（例如1024×5）。这两个张量无法直接进行相减操作。

## 解决方案

我修改了`train_model`函数，在计算损失之前，将`pos_score`扩展为与`neg_score`相同的维度：

```python
# 将pos_score扩展为与neg_score相同的维度
pos_score_expanded = pos_score.unsqueeze(1).expand_as(neg_score)
loss = torch.sum(torch.relu(pos_score_expanded - neg_score + MARGIN))
```

这个修改使用`unsqueeze(1)`在`pos_score`的第1维添加一个维度，然后使用`expand_as(neg_score)`将其扩展为与`neg_score`相同的形状。这样，两个张量就可以正确地进行相减操作了。

## 其他注意事项

1. 修改后的脚本应该能够正常运行，不会再出现维度不匹配的错误。
2. 预测函数的实现是正确的，但对于大型知识图谱可能会比较慢，因为它需要遍历所有实体来计算得分。
3. 如果您的数据集非常大，可能需要考虑进一步优化代码，例如使用向量化操作来加速预测过程。