****
1. 文本表示：从one-hot到word2vec。   
1.1 词袋模型：离散、高维、稀疏。   
1.2 分布式表示：连续、低维、稠密。word2vec词向量原理并实践，用来表示文本。   
****    
### 1. 文本表示   
**词袋模型：离散、高维、稀疏。**    
&emsp;离散：无法衡量词向量之间的关系。比如酒店、宾馆、旅社 三者都只在某一个固定的位置为 1 ，所以找不到三者的关系，各种度量(与或非、距离)都不合适，即太稀疏，很难捕捉到文本的含义。   
&emsp;高维：词表维度随着语料库增长膨胀，n-gram 序列随语料库膨胀更快。   
&emsp;稀疏： 数据都没有特征多，数据有 100 条，特征有 1000 个   
![词向量](./images/词向量.png)   
### 2. word2vec词向量原理并实践   
&emsp;&emsp;word2vec作为神经概率语言模型的输入，其本身其实是神经概率模型的副产品，是为了通过神经网络学习某个语言模型而产生的中间结果。
具体来说，“某个语言模型”指的是“CBOW”和“Skip-gram”。具体学习过程会用到两个降低复杂度的近似方法——Hierarchical Softmax或Negative Sampling。
两个模型乘以两种方法，一共有四种实现。   
1. CBOW：CBOW 是 Continuous Bag-of-Words Model 的缩写，是一种根据上下文的词语预测当前词语的出现概率的模型。
2. Skip-gram：Skip-gram只是逆转了CBOW的因果关系而已，即已知当前词语，预测上下文。     

[基于 Hierarchical Softmax 的模型](https://blog.csdn.net/itplus/article/details/37969817)   
&emsp;[文章介绍]word2vec 是 Google 于 2013 年开源推出的一个用于获取 word vector 的工具包，它简单、高效，因此引起了很多人的关注。由于 word2vec 的作者 Tomas Mikolov 在两篇相关的论文 [3,4] 中并没有谈及太多算法细节，因而在一定程度上增加了这个工具包的神秘感。一些按捺不住的人于是选择了通过解剖源代码的方式来一窥究竟，出于好奇，我也成为了他们中的一员。读完代码后，觉得收获颇多，整理成文，给有需要的朋友参考。     
[word2vec原理(二) 基于Hierarchical Softmax的模型-刘建平](http://www.cnblogs.com/pinard/p/7243513.html#!comments)   
写的很详细，特别是需要看评论，会解释你很多困惑   
怎么说呢，文章从昨天开始，认真看完了，朦胧的感知吧。贴了一些好的评论在Word2vec.md里    
