# AIchallenger_MachineReadingComprehension
AI Challenger 2018 观点型问题阅读理解比赛 8th place solution

****

|Author|[yuhaitao](https://github.com/yuhaitao1994)|[little_white](https://github.com/faverous)|
|---|---|---

[比赛总结]()
****

## 1.比赛成绩
|Model|Accuracy|
|---|---
|baseline|72.36%
|test_A ensemble|76.39%
|best single model|75.13%(dev)
|test_B ensemble|77.33%


## 2.环境配置

|环境/库|版本|
|---|---
|ubuntu|16.04
|Python|>=3.5
|TensorFlow|>=1.6

## 3.baseline

baseline模型借鉴了微软R-Net模型，感谢[HKUST-KnowComp](https://github.com/HKUST-KnowComp/R-Net)的tensorflow实现代码。

与R-Net模型不同的是，我们取消了模型尾部的ptrNet结构，取而代之的是一个单向GRU与softmax层。

### 打开方式

新建file目录，将训练集、验证集、测试集A原始数据移入。

数据预处理

    python config.py --mode prepro

训练

    python config.py --mode train 

评估验证集效果

    python config.py --mode examine_dev

生成测试结果

    python config.py --mode test


## 4.best single model

最好成绩的单模型我们选择加入alternatives语义和feature engineering的方式，基于R-Net改进。

**alternatives语义**：由于观点型问题的某些备选答案是携带语义信息的，所以我们将备选答案也做encoding处理。

**feature engneering**：特征工程，我们使用了tf-idf等方法，将提取的特征向量作为深度模型的另一个输入，只用Linear层进行处理。由于阅读理解任务数据的特性，特征工程这部分工作只有微弱提升，没有公开代码。

### 模型结构
![best single model](/pics/model.png)

## 5.ensemble

最终提交的test_B结果共采用了16个模型进行融合，融合的方式为stacking，在验证集上训练各模型softmax层所占权重。这种方式可能会造成在验证集上的过拟合，但据实际测试，并没有发生此问题。

我们一共使用了三种改进模型，分别基于R-Net、QA-Net和BiDAF。

### ensemble使用方式

训练集成模型的权重
    python ensemble_train.py 
预测test_A的结果
    python ensemble_predict.py
