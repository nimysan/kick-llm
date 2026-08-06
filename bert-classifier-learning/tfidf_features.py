"""方法 3 使用的 TF-IDF 特征提取器。"""

from collections import Counter

import numpy as np

from bag_of_words_features import tokenize


class TfidfVectorizer:
    """从零实现的 unigram TF-IDF，便于观察每一步计算。"""

    def __init__(self) -> None:
        self.vocabulary: list[str] = []
        self.word_to_index: dict[str, int] = {}
        self.idf = np.array([], dtype=np.float64)

    def fit(self, texts: list[str]) -> None:
        tokenized_texts = [tokenize(text) for text in texts]
        self.vocabulary = sorted({token for tokens in tokenized_texts for token in tokens})
        self.word_to_index = {word: index for index, word in enumerate(self.vocabulary)}

        document_frequency = np.zeros(len(self.vocabulary), dtype=np.float64)
        for tokens in tokenized_texts:
            for token in set(tokens):
                document_frequency[self.word_to_index[token]] += 1

        document_count = len(texts)
        self.idf = np.log((document_count + 1) / (document_frequency + 1)) + 1

    def transform_one(self, text: str) -> np.ndarray:
        tokens = tokenize(text)
        counts = Counter(tokens)
        vector = np.zeros(len(self.vocabulary), dtype=np.float64)
        if not tokens:
            return vector

        for token, count in counts.items():
            index = self.word_to_index.get(token)
            if index is not None:
                term_frequency = count / len(tokens)
                vector[index] = term_frequency * self.idf[index]
        return vector

    def transform(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.transform_one(text) for text in texts])

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)
