#!/usr/bin/env python3
"""第 5 个实验：TF-IDF + LSA/SVD 稠密特征 + 同一个逻辑回归。"""

from __future__ import annotations

import sys

import numpy as np

from bag_of_words_features import tokenize
from classifier import accuracy, sigmoid, train_logistic_regression
from evaluation_data import TEST_DATA
from lsa_features import LsaVectorizer
from training_data import TRAIN_DATA


def print_latent_components(vectorizer: LsaVectorizer, count: int = 3) -> None:
    """显示潜在维度由哪些原始词维度共同组成。"""
    vocabulary = vectorizer.tfidf.vocabulary
    print("\n前几个潜在维度中绝对贡献较大的词:")
    for component_index, component in enumerate(vectorizer.components[:count]):
        word_indices = np.argsort(np.abs(component))[-6:][::-1]
        words = ", ".join(
            f"{vocabulary[index]}({component[index]:+.2f})"
            for index in word_indices
        )
        print(f"  LSA[{component_index}] = {words}")


def print_prediction(
    text: str,
    vectorizer: LsaVectorizer,
    weights: np.ndarray,
    bias: float,
) -> None:
    tfidf_vector = vectorizer.tfidf.transform_one(text)
    lsa_vector = vectorizer.transform_one(text)
    logit = float(lsa_vector @ weights + bias)
    probability = float(sigmoid(logit))
    label = "正面" if probability >= 0.5 else "负面"

    nonzero_tfidf = {
        vectorizer.tfidf.vocabulary[index]: round(float(value), 4)
        for index, value in enumerate(tfidf_vector)
        if value
    }

    print(f'\n文本: "{text}"')
    print(f"分词: {tokenize(text)}")
    print(f"原始 TF-IDF.shape = {tfidf_vector.shape}")
    print(f"原始非零 TF-IDF = {nonzero_tfidf}")
    print(f"压缩后 LSA.shape = {lsa_vector.shape}")
    print(f"完整 LSA 向量 = {np.round(lsa_vector, 4).tolist()}")
    print(f"Wx+b = {logit:.4f}")
    print(f"P(正面) = {probability:.4f}")
    print(f"预测={label}")


def evaluate_test_set(
    vectorizer: LsaVectorizer,
    weights: np.ndarray,
    bias: float,
) -> float:
    texts = [text for text, _, _ in TEST_DATA]
    labels = np.array([label for _, label, _ in TEST_DATA], dtype=np.float64)
    features = vectorizer.transform(texts)
    score = accuracy(features, labels, weights, bias)

    print("\n留出测试集:")
    print(f"  总体准确率: {score:.1%}")
    for group in sorted({group for _, _, group in TEST_DATA}):
        indices = [index for index, item in enumerate(TEST_DATA) if item[2] == group]
        group_score = accuracy(features[indices], labels[indices], weights, bias)
        correct = round(group_score * len(indices))
        print(f"  {group:<8}: {group_score:.1%} ({correct}/{len(indices)})")
    return score


def main() -> None:
    train_texts = [text for text, _ in TRAIN_DATA]
    train_labels = np.array([label for _, label in TRAIN_DATA], dtype=np.float64)

    vectorizer = LsaVectorizer(n_components=10)
    train_features = vectorizer.fit_transform(train_texts)
    weights, bias = train_logistic_regression(train_features, train_labels)
    train_score = accuracy(train_features, train_labels, weights, bias)

    print("=" * 64)
    print("实验 05：TF-IDF + LSA/SVD + 逻辑回归")
    print("=" * 64)
    print(f"训练样本: {len(TRAIN_DATA)} 条")
    print(f"原始 TF-IDF X.shape = ({len(TRAIN_DATA)}, {len(vectorizer.tfidf.vocabulary)})")
    print(f"压缩后 LSA X.shape = {train_features.shape}")
    print(f"SVD components.shape = {vectorizer.components.shape}")
    print(f"前10个奇异值 = {np.round(vectorizer.singular_values[:10], 4).tolist()}")
    print(f"保留矩阵能量比例 = {vectorizer.explained_energy_ratio:.1%}")
    print(f"训练准确率 = {train_score:.1%}")

    print_latent_components(vectorizer)

    if len(sys.argv) > 1:
        print_prediction(" ".join(sys.argv[1:]), vectorizer, weights, bias)
        return

    print_prediction("这个产品不怎么好", vectorizer, weights, bias)
    evaluate_test_set(vectorizer, weights, bias)
    print("\nLSA 没有使用标签学习特征，也没有反向传播。")
    print("它只用 SVD 将多个相关的 TF-IDF 词维度组合成更少的潜在维度。")


if __name__ == "__main__":
    main()
