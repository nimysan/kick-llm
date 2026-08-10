#!/usr/bin/env python3
"""第 6 个实验：共现矩阵 + PPMI + SVD 词向量 + 逻辑回归。"""

from __future__ import annotations

import sys

import numpy as np

from bag_of_words_features import tokenize
from classifier import accuracy, sigmoid, train_logistic_regression
from evaluation_data import TEST_DATA
from ppmi_features import PpmiSvdVectorizer
from training_data import TRAIN_DATA


def print_word_contexts(
    vectorizer: PpmiSvdVectorizer,
    word: str,
    count: int = 6,
) -> None:
    index = vectorizer.word_to_index[word]
    context_indices = np.argsort(vectorizer.ppmi[index])[-count:][::-1]
    print(f'\n“{word}”的最高 PPMI 上下文:')
    for context_index in context_indices:
        value = vectorizer.ppmi[index, context_index]
        if value:
            print(f"  {vectorizer.vocabulary[context_index]:<8} PPMI={value:.4f}")


def print_prediction(
    text: str,
    vectorizer: PpmiSvdVectorizer,
    weights: np.ndarray,
    bias: float,
) -> None:
    tokens = tokenize(text)
    known_tokens = [token for token in tokens if token in vectorizer.word_to_index]
    unknown_tokens = [token for token in tokens if token not in vectorizer.word_to_index]
    vector = vectorizer.transform_one(text)
    logit = float(vector @ weights + bias)
    probability = float(sigmoid(logit))
    label = "正面" if probability >= 0.5 else "负面"

    print(f'\n文本: "{text}"')
    print(f"分词: {tokens}")
    print(f"参与平均的已知词: {known_tokens}")
    print(f"忽略的未知词: {unknown_tokens}")
    print(f"句子向量 x.shape = {vector.shape}")
    print(f"完整句子向量 x = {np.round(vector, 4).tolist()}")
    print(f"Wx+b = {logit:.4f}")
    print(f"P(正面) = {probability:.4f}")
    print(f"预测={label}")


def evaluate_test_set(
    vectorizer: PpmiSvdVectorizer,
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

    vectorizer = PpmiSvdVectorizer(n_components=10, window_size=2)
    train_features = vectorizer.fit_transform(train_texts)
    weights, bias = train_logistic_regression(train_features, train_labels)
    train_score = accuracy(train_features, train_labels, weights, bias)

    print("=" * 64)
    print("实验 06：共现矩阵 + PPMI + SVD + 逻辑回归")
    print("=" * 64)
    print(f"训练样本: {len(TRAIN_DATA)} 条")
    print(f"上下文窗口: 左右各 {vectorizer.window_size} 个词")
    print(f"词表大小: {len(vectorizer.vocabulary)}")
    print(f"共现矩阵.shape = {vectorizer.cooccurrence.shape}")
    print(f"共现矩阵非零元素 = {np.count_nonzero(vectorizer.cooccurrence)}")
    print(f"PPMI矩阵.shape = {vectorizer.ppmi.shape}")
    print(f"PPMI矩阵非零元素 = {np.count_nonzero(vectorizer.ppmi)}")
    print(f"每个词的SVD向量.shape = {vectorizer.word_embeddings.shape}")
    print(f"句子训练矩阵 X.shape = {train_features.shape}")
    print(f"保留PPMI矩阵能量比例 = {vectorizer.explained_energy_ratio:.1%}")
    print(f"训练准确率 = {train_score:.1%}")

    print_word_contexts(vectorizer, "满意")

    if len(sys.argv) > 1:
        print_prediction(" ".join(sys.argv[1:]), vectorizer, weights, bias)
        return

    print_prediction("这个产品不怎么好", vectorizer, weights, bias)
    evaluate_test_set(vectorizer, weights, bias)
    print("\nPPMI和SVD不使用情感标签，也没有反向传播。")
    print("它先从局部词语共现中得到词向量，再平均成句子特征。")


if __name__ == "__main__":
    main()
