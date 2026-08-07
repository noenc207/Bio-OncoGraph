# BioOncoGraph v2.0

BioOncoGraph 是一个全面的开源框架，用于构建和分析生物医学知识图谱，特别侧重于肿瘤学。

## 特性
- 可扩展的知识图谱构建
- 高级图神经网络（GNN）功能
- 端到端的训练、评估和预测管道
- 用于肿瘤学命名实体识别和关系提取的预训练模型

## 快速入门
```bash
pip install bio-oncograph
bio-oncograph train --config configs/train.yaml
bio-oncograph evaluate --model-path models/best_model
```

## 文档
有关完整文档，请访问我们的[文档网站](https://bio-oncograph.readthedocs.io/)。

## 许可证
Apache License 2.0。详情请参见 [LICENSE](LICENSE)。
