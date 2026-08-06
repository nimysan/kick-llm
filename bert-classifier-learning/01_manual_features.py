#!/usr/bin/env python3
"""第 1 个实验：人工特征 + 从零实现的二分类逻辑回归。"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence

import numpy as np


POSITIVE_WORDS = ("好", "满意", "喜欢", "优秀", "棒", "快", "值得", "推荐", "惊喜")
NEGATIVE_WORDS = ("差", "垃圾", "失望", "糟糕", "慢", "讨厌", "后悔", "坏", "难用")

FEATURE_NAMES = (
    "正面词数量",
    "负面词数量",
    "感叹号数量",
    "是否有正面词",
    "是否有负面词",
    "句子长度/20",
)

# 1 表示正面，0 表示负面。这些标签 y 是人工提供的正确答案。
TRAIN_DATA = (
    ("这个产品很好", 1),
    ("质量优秀，非常喜欢", 1),
    ("物流很快，我很满意", 1),
    ("效果真棒", 1),
    ("值得购买", 1),
    ("非常推荐这个产品", 1),
    ("使用体验很好", 1),
    ("速度快，质量也好", 1),
    ("收到以后很惊喜", 1),
    ("我喜欢，值得推荐", 1),
    ("这个产品太垃圾了", 0),
    ("质量很差", 0),
    ("物流太慢，非常失望", 0),
    ("体验非常糟糕", 0),
    ("真的很难用", 0),
    ("我很后悔购买", 0),
    ("效果差，不推荐", 0),
    ("速度慢，质量也差", 0),
    ("这个设计让我讨厌", 0),
    ("用起来很坏", 0),
)


def count_keywords(text: str, keywords: Sequence[str]) -> int:
    """统计人工词典中的词在句子里出现了多少次。"""
    return sum(text.count(word) for word in keywords)


def extract_features(text: str) -> np.ndarray:
    """由人规定规则，将任意文本转换成固定的 6 维向量 x。"""
    positive_count = count_keywords(text, POSITIVE_WORDS)
    negative_count = count_keywords(text, NEGATIVE_WORDS)
    exclamation_count = text.count("!") + text.count("！")

    return np.array(
        [
            positive_count,
            negative_count,
            exclamation_count,
            float(positive_count > 0),
            float(negative_count > 0),
            len(text) / 20.0,
        ],
        dtype=np.float64,
    )


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    """把任意实数分数压缩到 0~1，得到属于正面的概率。"""
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def train_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    learning_rate: float = 0.15,
    epochs: int = 4_000,
) -> tuple[np.ndarray, float]:
    """用梯度下降学习 W 和 b；人工特征提取规则在训练中不会变化。"""
    sample_count, feature_count = features.shape
    weights = np.zeros(feature_count, dtype=np.float64)
    bias = 0.0

    for _ in range(epochs):
        logits = features @ weights + bias
        probabilities = sigmoid(logits)
        errors = probabilities - labels

        weights -= learning_rate * (features.T @ errors) / sample_count
        bias -= learning_rate * float(errors.mean())

    return weights, bias


def predict(text: str, weights: np.ndarray, bias: float) -> tuple[np.ndarray, float, float]:
    features = extract_features(text)
    logit = float(features @ weights + bias)
    probability = float(sigmoid(logit))
    return features, logit, probability


def print_prediction(
    text: str,
    weights: np.ndarray,
    bias: float,
    expected: str | None = None,
) -> None:
    features, logit, probability = predict(text, weights, bias)
    label = "正面" if probability >= 0.5 else "负面"
    expected_text = f"；真实含义={expected}" if expected else ""

    print(f'\n文本: "{text}"')
    print(f"x = {np.round(features, 3).tolist()}")
    print(f"Wx+b = {logit:.4f}")
    print(f"sigmoid(Wx+b) = P(正面) = {probability:.4f}")
    print(f"预测={label}{expected_text}")


def main() -> None:
    train_features = np.stack([extract_features(text) for text, _ in TRAIN_DATA])
    train_labels = np.array([label for _, label in TRAIN_DATA], dtype=np.float64)
    weights, bias = train_logistic_regression(train_features, train_labels)

    train_probabilities = sigmoid(train_features @ weights + bias)
    train_predictions = (train_probabilities >= 0.5).astype(np.float64)
    accuracy = float((train_predictions == train_labels).mean())

    print("=" * 64)
    print("实验 01：人工特征 + 逻辑回归")
    print("=" * 64)
    print(f"训练样本: {len(TRAIN_DATA)} 条")
    print(f"人工特征: {len(FEATURE_NAMES)} 个")
    print(f"训练准确率: {accuracy:.1%}")

    print("\n人工定义的 x 各维含义：")
    for index, name in enumerate(FEATURE_NAMES):
        print(f"  x[{index}] = {name}")

    print("\n训练学到的 W 和 b：")
    for name, weight in zip(FEATURE_NAMES, weights):
        print(f"  {name:<12} W = {weight:+.4f}")
    print(f"  {'偏置':<12} b = {bias:+.4f}")

    if len(sys.argv) > 1:
        print_prediction(" ".join(sys.argv[1:]), weights, bias)
        return

    print("\n普通样本：")
    print_prediction("物流很快，产品也很好！", weights, bias, "正面")
    print_prediction("质量很差，体验糟糕", weights, bias, "负面")

    print("\n人工特征的局限（重点观察）：")
    print_prediction("我不满意", weights, bias, "负面")
    print_prediction("这东西一点也不差", weights, bias, "正面")

    print("\n结论：")
    print("  标签 y 训练了 W 和 b，但没有训练人工特征规则。")
    print("  模型只看到了人提取的 6 个数字，完全没有看到原始文字。")
    print("  “不满意”和“不差”需要人为增加否定规则，否则容易判断错误。")


if __name__ == "__main__":
    main()
