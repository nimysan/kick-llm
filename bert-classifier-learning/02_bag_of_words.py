#!/usr/bin/env python3
"""第 2 个实验：Bag of Words 特征 + 同一个二分类逻辑回归。"""

from __future__ import annotations

import sys

import numpy as np

from bag_of_words_features import BagOfWordsVectorizer, nonzero_features, tokenize
from classifier import accuracy, sigmoid, train_logistic_regression
from evaluation_data import TEST_DATA
from training_data import TRAIN_DATA


def print_prediction(
    text: str,
    vectorizer: BagOfWordsVectorizer,
    weights: np.ndarray,
    bias: float,
    expected: str | None = None,
) -> None:
    tokens = tokenize(text)
    vector = vectorizer.transform_one(text)
    logit = float(vector @ weights + bias)
    probability = float(sigmoid(logit))
    label = "正面" if probability >= 0.5 else "负面"
    expected_text = f"；真实含义={expected}" if expected else ""
    unknown_tokens = sorted(set(tokens) - set(vectorizer.word_to_index))

    print(f'\n文本: "{text}"')
    print(f"分词: {tokens}")
    print(f"完整词频向量 x（{len(vector)} 维）: {vector.astype(int).tolist()}")
    print(f"非零词频 x: {nonzero_features(vector, vectorizer.vocabulary)}")
    print("非零位置:")
    for index, count in enumerate(vector):
        if count:
            print(f'  x[{index:>2}] = "{vectorizer.vocabulary[index]}" 出现 {int(count)} 次')
    if unknown_tokens:
        print(f"训练词表中不存在，已忽略: {unknown_tokens}")
    print(f"Wx+b = {logit:.4f}")
    print(f"sigmoid(Wx+b) = P(正面) = {probability:.4f}")
    print(f"预测={label}{expected_text}")


def evaluate_unseen_data(
    vectorizer: BagOfWordsVectorizer,
    weights: np.ndarray,
    bias: float,
) -> None:
    """在从未参与训练的数据上评估泛化能力。"""
    texts = [text for text, _, _ in TEST_DATA]
    labels = np.array([label for _, label, _ in TEST_DATA], dtype=np.float64)
    features = vectorizer.transform(texts)
    probabilities = np.asarray(sigmoid(features @ weights + bias))
    predictions = (probabilities >= 0.5).astype(np.float64)

    print("\n" + "=" * 64)
    print("留出测试集：这些句子没有参与训练")
    print("=" * 64)
    print(f"测试样本: {len(TEST_DATA)} 条")
    print(f"总体准确率: {(predictions == labels).mean():.1%}")

    groups = sorted({group for _, _, group in TEST_DATA})
    for group in groups:
        indices = [index for index, item in enumerate(TEST_DATA) if item[2] == group]
        group_accuracy = (predictions[indices] == labels[indices]).mean()
        print(f"  {group:<8}: {group_accuracy:.1%} ({int((predictions[indices] == labels[indices]).sum())}/{len(indices)})")

    print("\n误判明细：")
    error_count = 0
    for index, (text, label, group) in enumerate(TEST_DATA):
        if predictions[index] == label:
            continue
        error_count += 1
        expected = "正面" if label == 1 else "负面"
        predicted = "正面" if predictions[index] == 1 else "负面"
        tokens = tokenize(text)
        unknown = sorted(set(tokens) - set(vectorizer.word_to_index))
        print(
            f'  [{group}] "{text}" '
            f"真实={expected} 预测={predicted} "
            f"P(正面)={probabilities[index]:.3f}"
        )
        if unknown:
            print(f"    被忽略的未知词: {unknown}")
    if error_count == 0:
        print("  无")


def main() -> None:
    texts = [text for text, _ in TRAIN_DATA]
    labels = np.array([label for _, label in TRAIN_DATA], dtype=np.float64)

    vectorizer = BagOfWordsVectorizer()
    train_features = vectorizer.fit_transform(texts)
    weights, bias = train_logistic_regression(train_features, labels)
    train_accuracy = accuracy(train_features, labels, weights, bias)

    print("=" * 64)
    print("实验 02：Bag of Words + 逻辑回归")
    print("=" * 64)
    print(f"训练样本: {len(TRAIN_DATA)} 条")
    print(f"词表大小/向量维度: {len(vectorizer.vocabulary)}")
    print(f"训练准确率: {train_accuracy:.1%}")
    print(f"词表: {vectorizer.vocabulary}")

    ranked_indices = np.argsort(weights)
    negative_indices = ranked_indices[:5]
    positive_indices = ranked_indices[-5:][::-1]
    print("\n权重最负的词:")
    for index in negative_indices:
        print(f"  {vectorizer.vocabulary[index]:<8} W = {weights[index]:+.4f}")
    print("权重最正的词:")
    for index in positive_indices:
        print(f"  {vectorizer.vocabulary[index]:<8} W = {weights[index]:+.4f}")
    print(f"偏置 b = {bias:+.4f}")

    if len(sys.argv) > 1:
        print_prediction(" ".join(sys.argv[1:]), vectorizer, weights, bias)
        return

    print("\n与人工关键词特征对比：")
    print_prediction("这个产品很好", vectorizer, weights, bias, "正面")
    print_prediction("这个产品不好", vectorizer, weights, bias, "负面")
    print_prediction("这个东西不差 不差 非常不差", vectorizer, weights, bias, "正面")
    print_prediction("这个东西很差", vectorizer, weights, bias, "负面")

    print("\nBag of Words 的新局限：")
    print_prediction("服务一点也不慢", vectorizer, weights, bias, "正面")
    print("\n结论：")
    print("  分类器仍然是同一个 Wx+b；改变的是文本到 x 的转换方式。")
    print("  词表由训练文本自动生成，不再需要人手工列出正负面词。")
    print("  但它不理解词义；没在训练中见过的词会被直接忽略。")
    evaluate_unseen_data(vectorizer, weights, bias)


if __name__ == "__main__":
    main()
