
# Comprehensive Stock Market Index Forecasting Model Based on LSTM: A Case Study of Dow Jones and SSE Composite Index
# 基于LSTM的综合性股票市场指数预测模型 —— 以道琼斯指数和上证综指为例

## 写在前面

本人是深度学习的初学者，本项目是本人运用所学知识解决实际问题的拙作。由于所学十分有限，代码有混乱或者冗余处还请各位同好多多包涵！
如有任何疑问或您不吝提供优化意见，请打开code文件夹下的contact me文件，获取本人的联系方式！

As a beginner in deep learning, this project is my first attempt to apply the knowledge I've acquired to solve real-world problems. Given my limited expertise, please excuse any confusion or redundancy in the code. 
If you have any questions, or suggestions for modifications and optimizations, please refer to the "contact me" file in the code folder to get in touch with me.

## 项目概述

本项目是一个综合性的股票市场指数预测模型，使用长短期记忆网络（LSTM）对道琼斯指数和上证综指进行预测。项目利用2018年1月1日至2022年12月31日的股市数据，主要参考收盘价进行深度学习模型的训练，并使用2023年的数据进行了回测，并进行参数调优。


在模型评估阶段，我们采用了多种指标来衡量预测性能。对于道琼斯指数，模型展现了MAE为397.18、MSE为231303.40、RMSE为480.94、R^2为0.836和MAPE为0.0117的表现。对于上证综指，模型的MAE为31.57、MSE为1521.68、RMSE为39.01、R^2为0.890和MAPE为0.01。同时，本项目提供了可视化的预测图表，展示了实际走势以及模型预测的指数范围，使得预测结果更加直观。

This project is a comprehensive stock market index forecasting model that employs Long Short-Term Memory networks (LSTM) to predict the Dow Jones Industrial Average and the SSE Composite Index. It leverages stock market data from January 1, 2018, to December 31, 2022, primarily focusing on closing prices for training the deep learning model, and performs backtesting with data from 2023, along with parameter optimization.

In the model evaluation phase, a variety of metrics were used to assess predictive performance. For the Dow Jones index, the model showed an MAE of 397.18, an MSE of 231303.40, an RMSE of 480.94, an R^2 of 0.836, and an MAPE of 0.0117. For the SSE Composite Index, the model achieved an MAE of 31.57, an MSE of 1521.68, an RMSE of 39.01, an R^2 of 0.890, and an MAPE of 0.01.

Additionally, this project provides visual prediction charts, displaying both the actual trends and the model's predicted index range, making the forecast results more intuitive.


## 快速开始
此处仅提供简单说明，如有疑问，请参见`Quickstart.md`文件以获取更详细的关于如何安装依赖项、配置环境和启动预测模型的说明。
1. **安装依赖**
    ```bash
    !pip install yfinance torch numpy matplotlib scikit-learn
    ```

2. **导入必要的库**
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

3. **数据准备和预处理**
    - 使用`yfinance`下载股票数据。
    - 对数据进行归一化处理。
    - 构造时间序列数据。

4. **模型定义和训练**
    - 定义LSTM模型架构。
    - 训练模型，并使用训练集和验证集进行评估。

5. **回测和评估**
    - 使用2023年的数据进行模型回测。
    - 计算并展示模型的性能指标（MAE, MSE, RMSE, R^2, MAPE）。
    - 绘制实际值与预测值的对比图。

## 贡献指南
欢迎通过Pull Requests或Issues来提供新的功能、改进或修复bug。

## 许可证
本项目采用MIT许可证。

