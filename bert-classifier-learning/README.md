# 文本分类器演进实验

这个目录用于逐步理解文本分类器如何从人工特征发展到 BERT。
每一步解决相同的情感二分类问题，只替换特征提取方式。

## 学习路线

| 步骤 | 特征提取方式 | 分类器 | 状态 |
|---|---|---|---|
| 01 | 人工设计特征 | 逻辑回归 `Wx + b` | 已实现 |
| 02 | Bag of Words | 逻辑回归 `Wx + b` | 已实现 |
| 03 | TF-IDF | 逻辑回归 `Wx + b` | 已实现 |
| 04 | N-gram | 逻辑回归 `Wx + b` | 已实现 |
| 05 | Word2Vec | 逻辑回归 `Wx + b` | 待实现 |
| 06 | RNN / LSTM | 线性分类层 | 待实现 |
| 07 | 冻结的 BERT | 线性分类层 | 待实现 |
| 08 | 微调 BERT | 线性分类层 | 待实现 |

## 第 1 步：人工特征

本目录统一使用 [uv](https://docs.astral.sh/uv/) 管理 Python 和依赖。
首次运行时，`uv` 会按照 `pyproject.toml` 和 `uv.lock` 自动创建环境。

运行：

```bash
uv sync
uv run python 01_manual_features.py
```

也可以分类自己输入的句子：

```bash
uv run python 01_manual_features.py "这个产品很好，我很满意"
```

脚本会展示：

1. 人如何规定文本中的哪些信息算作特征；
2. 一句话如何被转换成数字向量 `x`；
3. 逻辑回归如何从标签中学习 `W` 和 `b`；
4. `Wx + b` 如何变成分类概率；
5. 人工特征为什么容易被否定句等表达方式骗过。

这里实现的是二分类逻辑回归，而不是预测连续值的线性回归。两者都有
`Wx + b`，但逻辑回归还会通过 sigmoid 将分数转换成分类概率。

## 第 2 步：Bag of Words

运行：

```bash
uv run python 02_bag_of_words.py
```

测试自己的句子：

```bash
uv run python 02_bag_of_words.py "这个产品不好"
```

这个实验继续使用第 1 步的训练数据和逻辑回归，只把人工定义的 6 个特征
替换成自动生成的词频向量。可以重点比较“好/不好”和“差/不差”的输出。
不带自定义句子运行时，脚本还会使用 `evaluation_data.py` 中完全没有参与训练
的 32 条留出数据测试泛化能力，并按表达类型展示准确率和误判。

## 准确率对比图

使用相同训练集、测试集和逻辑回归，只比较两种特征提取器：

```bash
uv run python compare_accuracy.py
```

运行后会生成 `accuracy_comparison_4_methods.png`。

## 第 3 步：TF-IDF

运行：

```bash
uv run python 03_tfidf.py
```

测试自己的句子：

```bash
uv run python 03_tfidf.py "这个产品不差，但是不怎么好"
```

TF-IDF 的维度仍等于词表大小，但向量中的值从“出现次数”变成
`句内词频 TF × 稀有程度 IDF`，让常见词贡献更小、稀有词贡献更大。

## 第 4 步：N-gram

运行：

```bash
uv run python 04_ngram.py
```

测试自己的句子：

```bash
uv run python 04_ngram.py "这次购物让我不满意"
```

这个实验使用 `1-gram + 2-gram` 的 TF-IDF。它既保留单个词，也把连续两个
词作为新特征，因此可以为“满意”和“不 + 满意”学习不同权重。
