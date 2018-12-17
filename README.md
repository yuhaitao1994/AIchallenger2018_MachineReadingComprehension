# AIchallenger_MachineReadingComprehension
AI Challenger 2018 观点型问题阅读理解比赛 8th place solution

****

|Author|[yuhaitao](https://github.com/yuhaitao1994)|[little_white](https://github.com/faverous)|
|---|---|---

[比赛总结]()
****

## 1.比赛思路



## 2.环境配置

|环境/库|版本|
|---|---
|ubuntu|16.04
|Python|>=3.5
|TensorFlow|>=1.6

## 3.baseline

baseline模型借鉴了微软R-Net模型，感谢[HKUST-KnowComp](https://github.com/HKUST-KnowComp/R-Net)的tensorflow实现代码。


## 4.best single model

最好成绩的单模型我们选择加入alternatives语义和feature engineering的方式。

[](/pics/model.png)

## 5.ensemble


