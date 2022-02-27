## MAML：Model-Agnostic Meta-Learning模型的tensorflow实现
---

## 目录
1. [所需环境 Environment](#所需环境)
2. [模型结构 Structure](#模型结构)
3. [注意事项 Attention](#注意事项)
4. [文件下载 Download](#文件下载)
5. [训练步骤 How2train](#训练步骤) 

## 所需环境
Python3.7
tensorflow-gpu>=2.0  
Numpy==1.19.5
CUDA 11.0+
Pandas==1.2.4
Matplotlib==3.2.2

## 模型结构
MAML for classification
![image](https://github.com/JJASMINE22/MAML/blob/master/model_structure/maml/maml.png)

MAML for series prediction
![image](https://github.com/JJASMINE22/MAML/blob/master/model_structure/maml_lstm/maml_lstm.png)

## 注意事项
1. MAML结构适用于小样本模型训练，为避免过学习，模型不应设计过重
2. sub_model的参数务必通过手动更新，否则meta_model无法使用综合误差反向传递
3. 数据路径、训练参数均位于config.py

## 文件下载    
链接：https://pan.baidu.com/s/13T1Qs4NZL8NS4yoxCi-Qyw 
提取码：sets 
下载解压后放置于config.py中设置的路径即可。  

## 训练步骤
1. 分类模型：运行train.py
2. 预测模型：运行train_lstm.py


