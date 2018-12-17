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


## 4.best single model

最好成绩的单模型我们选择加入alternatives语义和feature engineering的方式。
alternatives语义：由于观点型问题的某些备选答案是携带语义信息的，所以我们将备选答案也做encoding处理。
feature engneering：特征工程，我们使用了tf-idf等方法，将提取的特征向量作为深度模型的另一个输入，只用Linear层进行处理。由于阅读理解任务数据的特性，特征工程这部分工作只有微弱提升，没有公开代码。

![best single model](/pics/model.png)

## 5.ensemble


