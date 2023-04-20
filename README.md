# MaiMemo Simulator

## 介绍

MaiMemo Simulator 是一个基于 [DHP](https://www.maimemo.com/paper/) 记忆模型的复习模拟器，可以用于评估不同复习策略的效果。

## 使用方法

### 1. 安装

```bash
pip install -r requirements.txt
```

### 2. 运行

```bash
python simulator.py
```

### 3. 结果

运行后会在当前 `./plot/` 下生成 `pdf` 文件，对比了不同复习策略的复习效果。

`./simulation/` 下会生成 `csv` 文件，记录了模拟结束后的记忆状态和复习历史。

## 项目结构

`./simulator.py` 是模拟器的主程序，可以在其中配置复习策略。

`./environment.py` 是模拟器的环境，根据模拟的复习反馈和记忆模型，更新记忆状态。

`./parameters.csv` 中存放的是 DHP 记忆模型的参数，可以根据需要进行修改。

`./policy/` 中存放的是 [SSP-MMC](https://github.com/maimemo/SSP-MMC) 算法生成的最优复习策略。

`./plot/` 中存放的是模拟器的策略对比结果。

`./simulation/` 中存放的是模拟器的复习过程记录。

## 对比策略

### 1. 墨墨记忆算法

[墨墨记忆算法](https://www.maimemo.com/paper/) 是基于 DHP 记忆模型的复习算法，其源码可以在 [SSP-MMC](https://github.com/maimemo/SSP-MMC) 中找到。

### 2. 列表复习周期

列表复习周期，又名艾宾浩斯复习周期（[实际上并非艾宾浩斯提出](https://www.zhihu.com/question/19798259/answer/2125871191)），是一种简单的复习策略，其复习周期为：1 天、2 天、4 天、7 天和 15 天。

### 3. SM-2

[SM-2](https://zhuanlan.zhihu.com/p/97887756) 是首个运行于计算机上的开源复习算法，由波兰的 Piotr Wozniak 博士于 20 世纪 80 年代发明，并在他的软件 SuperMemo 中首次应用。

### 4. Anki

[Anki](https://apps.ankiweb.net/) 是一款开源的间隔重复软件，其复习算法是 SM-2 的变种。