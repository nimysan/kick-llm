"""方法 4 使用的 unigram + bigram TF-IDF 特征提取器。"""

from collections import Counter

import numpy as np

from bag_of_words_features import tokenize


def make_ngrams(tokens: list[str]) -> list[str]:
    """同时生成单词特征和连续双词组合特征。"""
    unigrams = [f"1::{token}" for token in tokens]
    bigrams = [
        f"2::{tokens[index]} + {tokens[index + 1]}"
        for index in range(len(tokens) - 1)
    ]
    return unigrams + bigrams


class NgramTfidfVectorizer:
    """使用 unigram + bigram 的 TF-IDF 向量器。"""

    def __init__(self) -> None:
        self.vocabulary: list[str] = []
        self.feature_to_index: dict[str, int] = {}
        self.idf = np.array([], dtype=np.float64)

    def fit(self, texts: list[str]) -> None:
        document_features = [make_ngrams(tokenize(text)) for text in texts]
        self.vocabulary = sorted(
            {feature for features in document_features for feature in features}
        )
        self.feature_to_index = {
            feature: index for index, feature in enumerate(self.vocabulary)
        }

        document_frequency = np.zeros(len(self.vocabulary), dtype=np.float64)
        for features in document_features:
            for feature in set(features):
                document_frequency[self.feature_to_index[feature]] += 1

        document_count = len(texts)
        self.idf = np.log((document_count + 1) / (document_frequency + 1)) + 1

    def transform_one(self, text: str) -> np.ndarray:
        features = make_ngrams(tokenize(text))
        counts = Counter(features)
        vector = np.zeros(len(self.vocabulary), dtype=np.float64)
        if not features:
            return vector

        for feature, count in counts.items():
            index = self.feature_to_index.get(feature)
            if index is not None:
                term_frequency = count / len(features)
                vector[index] = term_frequency * self.idf[index]
        return vector

    def transform(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.transform_one(text) for text in texts])

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)
