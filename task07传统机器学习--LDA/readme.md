## 1. pLSA、共轭先验分布；LDA主题模型原理  
#### LSA简介  
&emsp;&emsp;该方法和传统向量空间模型(vector space model)一样使用向量来表示词(terms)和文档(documents)，并通过向量间的关系(如夹角)来判断词及文档间的关系；
不同的是，LSA 将词和文档映射到潜在语义空间，从而去除了原始向量空间中的一些“噪音”，提高了信息检索的精确度。    
&emsp;&emsp;如何得到这个低维空间呢，和PCA采用特征值分解的思想类似，作者采用了奇异值分解(Singular Value Decomposition)的方式来求解Latent Semantic Space。    
&emsp;&emsp;下图形象的展示了LSA的过程：   
![LSA](./images/LSA.jpg)  
#### pLSA简介
&emsp;&emsp;一篇文章通常是由多个主题构成的，而每一个主题大概可以用与该主题相关的频率最高的一些词来描述。对于语言学，容易想到的词包括：语法，句子，主语等；对于概率统计，容易想到的词包括：概率，模型，均值等；
以上这种想法由Hofmann于1999年给出的pLSA模型中首先进行了明确的数学化。Hofmann认为一篇文章（Doc）可以由多个主题（Topic）混合而成，
而每个Topic都是词汇上的概率分布，文章中的每个词都是由一个固定的Topic生成的。参考[具体公式推导过程](http://www.cnblogs.com/bentuwuying/p/6219970.html)    
#### 共轭先验分布    
先验分布和后验分布的形式应该是一样的，这样的分布我们一般叫共轭分布  
[推导在这里-刘建平-自然语言处理之LDA](https://www.cnblogs.com/pinard/p/6831308.html)   
概率问题慢慢悟吧还是^_^        
#### LDA主题模型原理   
