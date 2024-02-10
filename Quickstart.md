# Quickstart 快速开始指南

本指南将帮助您快速开始使用“基于LSTM的综合性股票市场指数预测模型”。

## 环境准备

确保您的Python环境已安装以下库：

- yfinance
- torch
- numpy
- matplotlib
- scikit-learn

您可以通过运行以下命令来安装这些依赖：

```bash
pip install yfinance torch numpy matplotlib scikit-learn
```

## 数据获取和预处理
### 1.导入必要的库
```python
import torch 
import yfinance as yf
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
```
### 2.下载股票数据：

使用yfinance库下载您感兴趣的股票数据。示例中以上证综指为例：
```python
ticker = "000001.SS"
stock = yf.Ticker(ticker)
data = stock.history(period="1d", start="2018-01-01", end="2022-12-31")
```
### 3.数据预处理：

对数据进行归一化处理，并构造时间序列数据：
```python
features = ['Open', 'High', 'Low', 'Close', 'Volume']
target = 'Close'
scaler = MinMaxScaler()
features_data = scaler.fit_transform(data[features])
```
## 模型定义和训练
### 1.定义LSTM模型：

定义一个简单的LSTM模型，您可以根据需要调整模型结构：
```python
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```
### 2.训练模型：
实例化模型、定义损失函数和优化器，然后进行训练。

```python
model = SimpleNet(input_size=5, hidden_size=64, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## 回测与评估
使用2023年的数据对模型进行回测，并计算评估指标，如MAE、MSE、RMSE、R^2和MAPE。

## 结果可视化
绘制实际值与模型预测值的对比图，以直观展示模型的预测效果。











