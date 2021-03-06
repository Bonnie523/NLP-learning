## 1. pLSA、共轭先验分布；LDA主题模型原理  
#### LSA简介  
&emsp;&emsp;该方法和传统向量空间模型(vector space model)一样使用向量来表示词(terms)和文档(documents)，并通过向量间的关系(如夹角)来判断词及文档间的关系；
不同的是，LSA 将词和文档映射到潜在语义空间，从而去除了原始向量空间中的一些“噪音”，提高了信息检索的精确度。    
&emsp;&emsp;如何得到这个低维空间呢，和PCA采用特征值分解的思想类似，作者采用了奇异值分解(Singular Value Decomposition)的方式来求解Latent Semantic Space。    
&emsp;&emsp;下图形象的展示了LSA的过程：   
![LSA](./LSA.jpg)  
&emsp;&emsp;三个矩阵有非常清楚的物理含义。第一个矩阵 U 中的每一行表示意思相关的一类词，其中的每个非零元素表示这类词中每个词的重要性（或者说相关性），数值越大越相关。最后一个矩阵 V 中的每一列表示同一主题一类文章，其中每个元素表示这类文章中每篇文章的相关性。中间的矩阵 D 则表示类词和文章类之间的相关性。因此，我们只要对关联矩阵 X 进行一次奇异值分解，我们就可以同时完成了近义词分类和文章的分类。（同时得到每类文章和每类词的相关性）。    
#### pLSA简介
&emsp;&emsp;一篇文章通常是由多个主题构成的，而每一个主题大概可以用与该主题相关的频率最高的一些词来描述。对于语言学，容易想到的词包括：语法，句子，主语等；对于概率统计，容易想到的词包括：概率，模型，均值等；
以上这种想法由Hofmann于1999年给出的pLSA模型中首先进行了明确的数学化。Hofmann认为一篇文章（Doc）可以由多个主题（Topic）混合而成。       
而每个Topic都是词汇上的概率分布，文章中的每个词都是由一个固定的Topic生成的。参考[具体公式推导过程](http://www.cnblogs.com/bentuwuying/p/6219970.html)    
#### 共轭先验分布    
先验分布和后验分布的形式应该是一样的，这样的分布我们一般叫共轭分布  
[推导在这里-刘建平-自然语言处理之LDA](https://www.cnblogs.com/pinard/p/6831308.html)   
概率问题慢慢悟吧还是^_^        
#### LDA(Latent Dirichlet allocation)主题模型原理（隐含狄利克雷分布）    

&emsp;&emsp;对于语料库中的每篇文档，LDA定义了如下生成过程（generative process）：   
(1) 对每一篇文档，从主题分布中抽取一个主题    
(2) 从上述被抽到的主题所对应的单词分布中抽取一个单词    
(3) 重复上述过程直至遍历文档中的每一个单词。    
概率看不明白，自己也没完全搞懂，就先不写这部分了，以后学到再说好了，推荐[LDA数学八卦](http://www.52nlp.cn/lda-math-汇总-lda数学八卦)系列，内容详细通俗易懂。   
[通俗理解LDA主题模型](https://cloud.tencent.com/developer/article/1058777)    
## 2、应用场景   
通常LDA用于进行主题模型挖掘，当然也可用于降维。    
* 推荐系统：应用LDA挖掘物品主题，计算主题相似度    
* 情感分析：学习出用户讨论、用户评论中的内容主题   
## 3. LDA优缺点   
LDA算法既可以用来降维，又可以用来分类，但是目前来说，主要还是用于降维。  
LDA算法的主要**优点**有：   
&emsp;&emsp;1）在降维过程中可以使用类别的先验知识经验，而像PCA这样的无监督学习则无法使用类别先验知识。   
&emsp;&emsp;2）LDA在样本分类信息依赖均值而不是方差的时候，比PCA之类的算法较优。   
LDA算法的主要**缺点**有：   
&emsp;&emsp;1）LDA不适合对非高斯分布样本进行降维，PCA也有这个问题。   
&emsp;&emsp;2）LDA降维最多降到类别数k-1的维数，如果我们降维的维度大于k-1，则不能使用LDA。当然目前有一些LDA的进化版算法可以绕过这个问题。   
&emsp;&emsp;3）LDA在样本分类信息依赖方差而不是均值的时候，降维效果不好。   
&emsp;&emsp;4）LDA可能过度拟合数据。    
## 4、LDA 参数学习   
LDA在sklearn中，sklearn.decomposition.LatentDirichletAllocation()   
主要参数：   
```
n_components : int, optional (default=10)
    主题数

doc_topic_prior : float, optional (default=None)
    文档主题先验Dirichlet分布θd的参数α

topic_word_prior : float, optional (default=None)
    主题词先验Dirichlet分布βk的参数η

learning_method : 'batch' | 'online', default='online'
    LDA的求解算法。有 ‘batch’ 和 ‘online’两种选择

learning_decay : float, optional (default=0.7)
   控制"online"算法的学习率，默认是0.7

learning_offset : float, optional (default=10.)
    仅在算法使用"online"时有意义，取值要大于1。用来减小前面训练样本批次对最终模型的影响
    
max_iter : integer, optional (default=10)
    EM算法的最大迭代次数

batch_size : int, optional (default=128)
   仅在算法使用"online"时有意义， 即每次EM算法迭代时使用的文档样本的数量。

evaluate_every : int, optional (default=0)
    多久评估一次perplexity。仅用于`fit`方法。将其设置为0或负数以不评估perplexity
     训练。
     
total_samples : int, optional (default=1e6)
    仅在算法使用"online"时有意义， 即分步训练时每一批文档样本的数量。在使用partial_fit函数时需要。

perp_tol : float, optional (default=1e-1)
    batch的perplexity容忍度。

mean_change_tol : float, optional (default=1e-3)
    即E步更新变分参数的阈值，所有变分参数更新小于阈值则E步结束，转入M步。

max_doc_update_iter : int (default=100)
    即E步更新变分参数的最大迭代次数，如果E步迭代次数达到阈值，则转入M步。

n_jobs : int, optional (default=1)
   在E步中使用的资源数量。 如果为-1，则使用所有CPU。
     ``n_jobs``低于-1，（n_cpus + 1 + n_jobs）被使用。

verbose : int, optional (default=0)
    详细程度。
```
## 5.使用LDA生成主题特征，在之前特征的基础上加入主题特征进行文本分类   
```
from sklearn.decomposition import LatentDirichletAllocation
n_topic = 10
lda = LatentDirichletAllocation(n_topics=n_topic, 
                                max_iter=50,
                                learning_method='batch')
lda.fit(tf) #tf即为Document_word Sparse Matrix     
out...
Topic #0:
中国 发展 问题 工作 经济 国家 表示 进行 合作 建设 政府 国际 加强 部门 规定 社会 记者 要求 相关 会议
Topic #1:
搭配 时尚 设计 风格 组图 性感 黑色 选择 造型 导语 白色 感觉 外套 流行 颜色 款式 气质 色彩 明星 装扮
Topic #2:
功能 像素 采用 拍摄 支持 机身 英寸 镜头 佳能 光学 相机 拥有 索尼 高清 自动 对焦 性能 具有 使用 编辑
Topic #3:
电影 一个 导演 微博 新浪 影片 觉得 表示 中国 记者 现在 观众 娱乐 拍摄 已经 票房 希望 角色 演员 知道
Topic #4:
产品 企业 家具 一个 品牌 市场 消费者 行业 中国 家居 发展 公司 装修 一些 设计 服务 现在 销售 问题 环保
Topic #5:
学生 留学 美国 移民 大学 中国 申请 学校 教育 学习 专业 考试 一个 签证 留学生 英国 孩子 工作 选择 课程
Topic #6:
比赛 球队 篮板 球员 季后赛 时间 赛季 热火 火箭 新浪 进攻 已经 防守 湖人 一个 命中 体育讯 助攻 表现 三分
``` 
[git]()  
[【sklearn】利用sklearn训练LDA主题模型及调参详解](https://blog.csdn.net/TiffanyRabbit/article/details/76445909)   
[NLP实践四：LDA主题模型](https://blog.csdn.net/chen_yiwei/article/details/88370526)
