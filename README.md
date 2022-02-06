## MAMAL：Model-Agnostic Meta-Learning模型的TF实现
---

1. [所需环境 Environment](#所需环境)
2. [注意事项 Cautions](#注意事项)
 
1.所需环境
numpy==1.19.5
tensorflow-gpu==2.5.1  
tensorflow-datasets==4.4.0  

2.注意事项
MAML仅适用于小样本建模，当模型参数过多，极可能导致过学习，
通过遍历模型的每一层来获取参数属性
tensorflow提供的LSTM层在CUDA加速状态下无法实现MAML, 该BUG与cudnn有关
