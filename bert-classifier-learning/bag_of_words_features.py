"""方法 2 使用的 Bag of Words 特征提取器。"""

import logging
import warnings
from collections import Counter

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    import jieba


jieba.setLogLevel(logging.ERROR)


def tokenize(text: str) -> list[str]:
    """中文分词，并删除空白与纯标点 token。"""
    return [
        token.strip().lower()
        for token in jieba.lcut(text)
        if token.strip() and any(character.isalnum() for character in token)
    ]


class BagOfWordsVectorizer:
    """一个最小可读的词袋向量器：每一维代表词表中的一个词。"""

    def __init__(self) -> None:
        self.vocabulary: list[str] = []
        self.word_to_index: dict[str, int] = {}

    def fit(self, texts: list[str]) -> None:
        self.vocabulary = sorted({token for text in texts for token in tokenize(text)})
        self.word_to_index = {word: index for index, word in enumerate(self.vocabulary)}

    def transform_one(self, text: str) -> np.ndarray:
        vector = np.zeros(len(self.vocabulary), dtype=np.float64)
        for token, count in Counter(tokenize(text)).items():
            index = self.word_to_index.get(token)
            if index is not None:
                vector[index] = count
        return vector

    def transform(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.transform_one(text) for text in texts])

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)


def nonzero_features(vector: np.ndarray, vocabulary: list[str]) -> dict[str, int]:
    """把高维稀疏向量显示为更容易阅读的 {词: 次数}。"""
    return {
        vocabulary[index]: int(value)
        for index, value in enumerate(vector)
        if value != 0
    }
