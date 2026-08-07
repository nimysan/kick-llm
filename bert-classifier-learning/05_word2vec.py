#!/usr/bin/env python3
"""第 5 个实验：Word2Vec 平均句向量 + 同一个逻辑回归。"""

from __future__ import annotations

import sys

import numpy as np

from bag_of_words_features import tokenize
from classifier import accuracy, sigmoid, train_logistic_regression
from evaluation_data import TEST_DATA
from training_data import TRAIN_DATA
from word2vec_features import Word2VecVectorizer


def print_prediction(
    text: str,
    vectorizer: Word2VecVectorizer,
    weights: np.ndarray,
    bias: float,
) -> None:
    tokens = tokenize(text)
    known_tokens = [token for token in tokens if token in vectorizer.vocabulary]
    unknown_tokens = [token for token in tokens if token not in vectorizer.vocabulary]
    vector = vectorizer.transform_one(text)
    logit = float(vector @ weights + bias)
    probability = float(sigmoid(logit))
    label = "正面" if probability >= 0.5 else "负面"

    print(f'\n文本: "{text}"')
    print(f"分词: {tokens}")
    print(f"参与平均的已知词: {known_tokens}")
    print(f"忽略的未知词: {unknown_tokens}")
    print(f"x.shape = {vector.shape}")
    print(f"完整 50 维句向量 x: {np.round(vector, 4).tolist()}")
    print(f"Wx+b = {logit:.4f}")
    print(f"P(正面) = {probability:.4f}")
    print(f"预测={label}")


def evaluate_test_set(
    vectorizer: Word2VecVectorizer,
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


def evaluate_seed_stability(train_texts: list[str], train_labels: np.ndarray) -> None:
    """展示极小语料下 Word2Vec 对随机初始化的敏感程度。"""
    test_texts = [text for text, _, _ in TEST_DATA]
    test_labels = np.array([label for _, label, _ in TEST_DATA], dtype=np.float64)
    seeds = (1, 7, 21, 42, 100)
    scores: list[float] = []

    print("\n不同随机种子的测试准确率:")
    for seed in seeds:
        vectorizer = Word2VecVectorizer(vector_size=100, seed=seed)
        train_features = vectorizer.fit_transform(train_texts)
        test_features = vectorizer.transform(test_texts)
        weights, bias = train_logistic_regression(train_features, train_labels)
        score = accuracy(test_features, test_labels, weights, bias)
        scores.append(score)
        print(f"  seed={seed:>3}: {score:.1%}")
    print(
        f"  平均={np.mean(scores):.1%} "
        f"范围={np.min(scores):.1%}~{np.max(scores):.1%}"
    )


def main() -> None:
    train_texts = [text for text, _ in TRAIN_DATA]
    train_labels = np.array([label for _, label in TRAIN_DATA], dtype=np.float64)

    print("正在仅使用 26 条训练文本学习 Word2Vec 特征...")
    vectorizer = Word2VecVectorizer(
        vector_size=100,
        seed=42,
        show_tokenized_texts=True,
    )
    train_features = vectorizer.fit_transform(train_texts)
    weights, bias = train_logistic_regression(train_features, train_labels)
    train_score = accuracy(train_features, train_labels, weights, bias)

    print("=" * 64)
    print("实验 05：Word2Vec 平均句向量 + 逻辑回归")
    print("=" * 64)
    print(f"Word2Vec 无标签训练语料: {len(train_texts)} 句")
    print(f"Word2Vec 词表: {len(vectorizer.vocabulary)} 个词")
    print(f"每个词/每句话的向量维度: {vectorizer.vector_size}")
    print(f"X_train.shape = {train_features.shape}")
    print(f"训练准确率: {train_score:.1%}")

    if vectorizer.model is not None:
        print("\n小语料学到的“好”最相近词（结果可能不可靠）:")
        for word, similarity in vectorizer.model.wv.most_similar("好", topn=5):
            print(f"  {word:<8} cosine={similarity:+.4f}")

    if len(sys.argv) > 1:
        print_prediction(" ".join(sys.argv[1:]), vectorizer, weights, bias)
        return

    print_prediction("这个产品十分出色", vectorizer, weights, bias)
    evaluate_test_set(vectorizer, weights, bias)
    evaluate_seed_stability(train_texts, train_labels)
    print("\n注意：本实验只用 26 句话学习词向量，结果主要用于理解流程。")
    print("工业实践通常加载在百万或十亿词语料上预训练的 Word2Vec。")


if __name__ == "__main__":
    main()
