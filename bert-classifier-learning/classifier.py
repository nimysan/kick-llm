"""各阶段共用的二分类逻辑回归，确保实验只替换特征提取器。"""

from __future__ import annotations

import numpy as np


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
    """使用梯度下降学习 W 和 b。"""
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


def accuracy(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, bias: float) -> float:
    probabilities = sigmoid(features @ weights + bias)
    predictions = (probabilities >= 0.5).astype(np.float64)
    return float((predictions == labels).mean())
