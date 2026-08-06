#!/usr/bin/env python3
"""第 4 个实验：unigram + bigram TF-IDF + 同一个逻辑回归。"""

from __future__ import annotations

import sys

import numpy as np

from bag_of_words_features import tokenize
from classifier import accuracy, sigmoid, train_logistic_regression
from evaluation_data import TEST_DATA
from ngram_features import NgramTfidfVectorizer, make_ngrams
from training_data import TRAIN_DATA


def print_prediction(
    text: str,
    vectorizer: NgramTfidfVectorizer,
    weights: np.ndarray,
    bias: float,
) -> None:
    tokens = tokenize(text)
    generated_features = make_ngrams(tokens)
    vector = vectorizer.transform_one(text)
    logit = float(vector @ weights + bias)
    probability = float(sigmoid(logit))
    label = "正面" if probability >= 0.5 else "负面"

    print(f'\n文本: "{text}"')
    print(f"分词: {tokens}")
    print(f"生成的 1-gram/2-gram: {generated_features}")
    print(f"x.shape = {vector.shape}")
    print(f"完整 x（{len(vector)} 维）:")
    print(f"  {np.round(vector, 4).tolist()}")
    print(f"非零元素: {np.count_nonzero(vector)}/{len(vector)}")
    print("训练词表中存在的非零特征:")
    for index, value in enumerate(vector):
        if value:
            print(
                f"  x[{index:>3}] = {vectorizer.vocabulary[index]:<20} "
                f"TF-IDF={value:.4f}  W={weights[index]:+.4f}"
            )
    unknown = sorted(set(generated_features) - set(vectorizer.feature_to_index))
    if unknown:
        print(f"训练词表中不存在，已忽略: {unknown}")
    print(f"Wx+b = {logit:.4f}")
    print(f"P(正面) = {probability:.4f}")
    print(f"预测={label}")


def evaluate_test_set(
    vectorizer: NgramTfidfVectorizer,
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

    vectorizer = NgramTfidfVectorizer()
    train_features = vectorizer.fit_transform(train_texts)
    weights, bias = train_logistic_regression(train_features, train_labels)
    train_score = accuracy(train_features, train_labels, weights, bias)

    unigram_count = sum(feature.startswith("1::") for feature in vectorizer.vocabulary)
    bigram_count = sum(feature.startswith("2::") for feature in vectorizer.vocabulary)

    print("=" * 64)
    print("实验 04：Unigram + Bigram TF-IDF + 逻辑回归")
    print("=" * 64)
    print(f"训练样本: {len(TRAIN_DATA)} 条")
    print(f"1-gram 特征: {unigram_count}")
    print(f"2-gram 特征: {bigram_count}")
    print(f"总向量维度: {len(vectorizer.vocabulary)}")
    print(f"训练准确率: {train_score:.1%}")

    bigram_indices = [
        index
        for index, feature in enumerate(vectorizer.vocabulary)
        if feature.startswith("2::")
    ]
    negative_bigrams = sorted(bigram_indices, key=lambda index: weights[index])[:5]
    positive_bigrams = sorted(
        bigram_indices, key=lambda index: weights[index], reverse=True
    )[:5]
    print("\n负权重最高的 2-gram:")
    for index in negative_bigrams:
        print(f"  {vectorizer.vocabulary[index]:<24} W={weights[index]:+.4f}")
    print("正权重最高的 2-gram:")
    for index in positive_bigrams:
        print(f"  {vectorizer.vocabulary[index]:<24} W={weights[index]:+.4f}")

    if len(sys.argv) > 1:
        print_prediction(" ".join(sys.argv[1:]), vectorizer, weights, bias)
        return

    print_prediction("这次购物让我不满意", vectorizer, weights, bias)
    print_prediction("这次购物让我很满意", vectorizer, weights, bias)
    evaluate_test_set(vectorizer, weights, bias)


if __name__ == "__main__":
    main()
