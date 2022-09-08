# 本教程介绍了使用 TensorFlow 进行时间序列预测。它构建了几种不同风格的模型，包括卷积和循环神经网络（CNN 和 RNN）。

1. 单个时间步长的预测
    1. A single feature.单一特征
    2. All features.全特征
2. 多个时间步长的预测
   1. Single-shot：一次性做出所有预测。
   2. 自回归：一次做一个预测并将输出反馈给模型。

- LSTM_stock_v1:数据分析

- LSTM_stock_v2:single-step-model

- LSTM_stock_v3:multi-step-model

LSTM
https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM