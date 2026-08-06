#!/usr/bin/env python3
"""第 1 个实验：人工特征 + 从零实现的二分类逻辑回归。"""

from __future__ import annotations

import sys

import numpy as np

from classifier import accuracy, sigmoid, train_logistic_regression
from manual_features import FEATURE_NAMES, extract_features
from training_data import TRAIN_DATA


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

    train_accuracy = accuracy(train_features, train_labels, weights, bias)

    print("=" * 64)
    print("实验 01：人工特征 + 逻辑回归")
    print("=" * 64)
    print(f"训练样本: {len(TRAIN_DATA)} 条")
    print(f"人工特征: {len(FEATURE_NAMES)} 个")
    print(f"训练准确率: {train_accuracy:.1%}")

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
