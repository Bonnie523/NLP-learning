## 1. word2vec
&emsp;word2vec作为神经概率语言模型的输入，其本身其实是神经概率模型的副产品，是为了通过神经网络学习某个语言模型而产生的中间结果。
具体来说，“某个语言模型”指的是“CBOW”和“Skip-gram”。具体学习过程会用到两个降低复杂度的近似方法——Hierarchical Softmax或Negative Sampling。
两个模型乘以两种方法，一共有四种实现。这些内容就是本文理论部分要详细阐明的全部了。   
![CBOW](./images/CBOW.png)   
