#!/usr/bin/env python3
"""第 3 个实验：TF-IDF 特征 + 同一个二分类逻辑回归。"""

from __future__ import annotations

import sys

import numpy as np

from bag_of_words_features import tokenize
from classifier import accuracy, sigmoid, train_logistic_regression
from evaluation_data import TEST_DATA
from tfidf_features import TfidfVectorizer
from training_data import TRAIN_DATA


def print_prediction(
    text: str,
    vectorizer: TfidfVectorizer,
    weights: np.ndarray,
    bias: float,
) -> None:
    vector = vectorizer.transform_one(text)
    logit = float(vector @ weights + bias)
    probability = float(sigmoid(logit))
    label = "正面" if probability >= 0.5 else "负面"

    print(f'\n文本: "{text}"')
    print(f"分词: {tokenize(text)}")
    print("非零 TF-IDF 特征:")
    for index, value in enumerate(vector):
        if value:
            word = vectorizer.vocabulary[index]
            print(f'  x[{index:>2}] = "{word}"  TF-IDF={value:.4f}  IDF={vectorizer.idf[index]:.4f}')
    print(f"Wx+b = {logit:.4f}")
    print(f"P(正面) = {probability:.4f}")
    print(f"预测={label}")


def evaluate_test_set(
    vectorizer: TfidfVectorizer,
    weights: np.ndarray,
    bias: float,
) -> float:
    texts = [text for text, _, _ in TEST_DATA]
    labels = np.array([label for _, label, _ in TEST_DATA], dtype=np.float64)
    features = vectorizer.transform(texts)
    probabilities = np.asarray(sigmoid(features @ weights + bias))
    predictions = (probabilities >= 0.5).astype(np.float64)
    score = float((predictions == labels).mean())

    print("\n留出测试集:")
    print(f"  总体准确率: {score:.1%}")
    for group in sorted({group for _, _, group in TEST_DATA}):
        indices = [index for index, item in enumerate(TEST_DATA) if item[2] == group]
        correct = int((predictions[indices] == labels[indices]).sum())
        print(f"  {group:<8}: {correct / len(indices):.1%} ({correct}/{len(indices)})")
    return score


def main() -> None:
    train_texts = [text for text, _ in TRAIN_DATA]
    train_labels = np.array([label for _, label in TRAIN_DATA], dtype=np.float64)

    vectorizer = TfidfVectorizer()
    train_features = vectorizer.fit_transform(train_texts)
    weights, bias = train_logistic_regression(train_features, train_labels)
    train_score = accuracy(train_features, train_labels, weights, bias)

    print("=" * 64)
    print("实验 03：TF-IDF + 逻辑回归")
    print("=" * 64)
    print(f"训练样本: {len(TRAIN_DATA)} 条")
    print(f"词表大小/向量维度: {len(vectorizer.vocabulary)}")
    print(f"训练准确率: {train_score:.1%}")

    idf_order = np.argsort(vectorizer.idf)
    print("\n最低 IDF（训练集中更常见，权重被降低）:")
    for index in idf_order[:5]:
        print(f"  {vectorizer.vocabulary[index]:<8} IDF={vectorizer.idf[index]:.4f}")
    print("最高 IDF（训练集中更稀有，权重被提高）:")
    for index in idf_order[-5:]:
        print(f"  {vectorizer.vocabulary[index]:<8} IDF={vectorizer.idf[index]:.4f}")

    if len(sys.argv) > 1:
        print_prediction(" ".join(sys.argv[1:]), vectorizer, weights, bias)
        return

    print_prediction("这个产品不差 非常不差 不怎么好", vectorizer, weights, bias)
    evaluate_test_set(vectorizer, weights, bias)


if __name__ == "__main__":
    main()
