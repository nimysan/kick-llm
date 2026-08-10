#!/usr/bin/env python3
"""在同一训练集和留出测试集上比较方法 1 与方法 2。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bag_of_words_features import BagOfWordsVectorizer
from classifier import accuracy, train_logistic_regression
from evaluation_data import TEST_DATA
from lsa_features import LsaVectorizer
from manual_features import extract_features
from ngram_features import NgramTfidfVectorizer
from ppmi_features import PpmiSvdVectorizer
from tfidf_features import TfidfVectorizer
from training_data import TRAIN_DATA
from word2vec_features import Word2VecVectorizer


OUTPUT_PATH = Path(__file__).with_name("accuracy_comparison_learning_path.png")


def train_and_evaluate(features_train: np.ndarray, features_test: np.ndarray) -> tuple[float, float]:
    train_labels = np.array([label for _, label in TRAIN_DATA], dtype=np.float64)
    test_labels = np.array([label for _, label, _ in TEST_DATA], dtype=np.float64)
    weights, bias = train_logistic_regression(features_train, train_labels)
    return (
        accuracy(features_train, train_labels, weights, bias),
        accuracy(features_test, test_labels, weights, bias),
    )


def main() -> None:
    train_texts = [text for text, _ in TRAIN_DATA]
    test_texts = [text for text, _, _ in TEST_DATA]

    manual_train = np.stack([extract_features(text) for text in train_texts])
    manual_test = np.stack([extract_features(text) for text in test_texts])
    manual_scores = train_and_evaluate(manual_train, manual_test)

    vectorizer = BagOfWordsVectorizer()
    bow_train = vectorizer.fit_transform(train_texts)
    bow_test = vectorizer.transform(test_texts)
    bow_scores = train_and_evaluate(bow_train, bow_test)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_train = tfidf_vectorizer.fit_transform(train_texts)
    tfidf_test = tfidf_vectorizer.transform(test_texts)
    tfidf_scores = train_and_evaluate(tfidf_train, tfidf_test)

    ngram_vectorizer = NgramTfidfVectorizer()
    ngram_train = ngram_vectorizer.fit_transform(train_texts)
    ngram_test = ngram_vectorizer.transform(test_texts)
    ngram_scores = train_and_evaluate(ngram_train, ngram_test)

    lsa_vectorizer = LsaVectorizer(n_components=10)
    lsa_train = lsa_vectorizer.fit_transform(train_texts)
    lsa_test = lsa_vectorizer.transform(test_texts)
    lsa_scores = train_and_evaluate(lsa_train, lsa_test)

    ppmi_vectorizer = PpmiSvdVectorizer(n_components=10, window_size=2)
    ppmi_train = ppmi_vectorizer.fit_transform(train_texts)
    ppmi_test = ppmi_vectorizer.transform(test_texts)
    ppmi_scores = train_and_evaluate(ppmi_train, ppmi_test)

    word2vec_vectorizer = Word2VecVectorizer(vector_size=50, seed=42)
    word2vec_train = word2vec_vectorizer.fit_transform(train_texts)
    word2vec_test = word2vec_vectorizer.transform(test_texts)
    word2vec_scores = train_and_evaluate(word2vec_train, word2vec_test)

    method_names = [
        "Method 1\nManual Features",
        "Method 2\nBag of Words",
        "Method 3\nTF-IDF",
        "Method 4\n1-2 gram TF-IDF",
        "Method 5\nLSA (10D)",
        "Method 6\nPPMI + SVD",
        "Method 7\nWord2Vec (seed=42)",
    ]
    train_scores = np.array(
        [
            manual_scores[0],
            bow_scores[0],
            tfidf_scores[0],
            ngram_scores[0],
            lsa_scores[0],
            ppmi_scores[0],
            word2vec_scores[0],
        ]
    )
    test_scores = np.array(
        [
            manual_scores[1],
            bow_scores[1],
            tfidf_scores[1],
            ngram_scores[1],
            lsa_scores[1],
            ppmi_scores[1],
            word2vec_scores[1],
        ]
    )
    positions = np.arange(len(method_names))
    width = 0.34

    fig, ax = plt.subplots(figsize=(19, 5.5))
    train_bars = ax.bar(
        positions - width / 2,
        train_scores,
        width,
        label="Training set (26)",
        color="#2A6FBB",
    )
    test_bars = ax.bar(
        positions + width / 2,
        test_scores,
        width,
        label="Held-out test set (32)",
        color="#E07A3F",
    )

    ax.set_title("Text Classification Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(positions, method_names)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.2), [f"{value:.0%}" for value in np.arange(0, 1.01, 0.2)])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    for bars in (train_bars, test_bars):
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in bars.datavalues], padding=4)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180)
    plt.close(fig)

    print("相同数据、相同逻辑回归，只替换特征提取器：")
    print(f"方法 1 人工特征     训练准确率={manual_scores[0]:.1%}  测试准确率={manual_scores[1]:.1%}")
    print(f"方法 2 Bag of Words 训练准确率={bow_scores[0]:.1%}  测试准确率={bow_scores[1]:.1%}")
    print(f"方法 3 TF-IDF      训练准确率={tfidf_scores[0]:.1%}  测试准确率={tfidf_scores[1]:.1%}")
    print(f"方法 4 N-gram      训练准确率={ngram_scores[0]:.1%}  测试准确率={ngram_scores[1]:.1%}")
    print(f"方法 5 LSA         训练准确率={lsa_scores[0]:.1%}  测试准确率={lsa_scores[1]:.1%}")
    print(f"方法 6 PPMI + SVD  训练准确率={ppmi_scores[0]:.1%}  测试准确率={ppmi_scores[1]:.1%}")
    print(f"方法 7 Word2Vec    训练准确率={word2vec_scores[0]:.1%}  测试准确率={word2vec_scores[1]:.1%}")
    print(f"图表已保存: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
