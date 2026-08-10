"""方法 6：共现统计 + PPMI + SVD 词向量。"""

import numpy as np

from bag_of_words_features import tokenize


class PpmiSvdVectorizer:
    """从训练文本的局部共现关系中生成低维词向量。"""

    def __init__(self, n_components: int = 10, window_size: int = 2) -> None:
        self.n_components = n_components
        self.window_size = window_size
        self.vocabulary: list[str] = []
        self.word_to_index: dict[str, int] = {}
        self.cooccurrence = np.empty((0, 0), dtype=np.float64)
        self.ppmi = np.empty((0, 0), dtype=np.float64)
        self.word_embeddings = np.empty((0, 0), dtype=np.float64)
        self.singular_values = np.array([], dtype=np.float64)
        self.explained_energy_ratio = 0.0

    def fit(self, texts: list[str]) -> None:
        tokenized_texts = [tokenize(text) for text in texts]
        self.vocabulary = sorted(
            {token for tokens in tokenized_texts for token in tokens}
        )
        self.word_to_index = {
            word: index for index, word in enumerate(self.vocabulary)
        }

        vocabulary_size = len(self.vocabulary)
        self.cooccurrence = np.zeros(
            (vocabulary_size, vocabulary_size),
            dtype=np.float64,
        )

        for tokens in tokenized_texts:
            for target_position, target_word in enumerate(tokens):
                left = max(0, target_position - self.window_size)
                right = min(len(tokens), target_position + self.window_size + 1)
                target_index = self.word_to_index[target_word]

                for context_position in range(left, right):
                    if context_position == target_position:
                        continue
                    context_word = tokens[context_position]
                    context_index = self.word_to_index[context_word]
                    self.cooccurrence[target_index, context_index] += 1

        self.ppmi = self._calculate_ppmi(self.cooccurrence)
        left_vectors, singular_values, _ = np.linalg.svd(
            self.ppmi,
            full_matrices=False,
        )

        component_count = min(self.n_components, vocabulary_size)
        self.singular_values = singular_values
        self.word_embeddings = (
            left_vectors[:, :component_count]
            * np.sqrt(singular_values[:component_count])
        )

        total_energy = float(np.sum(singular_values**2))
        retained_energy = float(np.sum(singular_values[:component_count] ** 2))
        self.explained_energy_ratio = retained_energy / total_energy

    @staticmethod
    def _calculate_ppmi(cooccurrence: np.ndarray) -> np.ndarray:
        total = float(cooccurrence.sum())
        row_totals = cooccurrence.sum(axis=1, keepdims=True)
        column_totals = cooccurrence.sum(axis=0, keepdims=True)
        expected_denominator = row_totals @ column_totals

        ppmi = np.zeros_like(cooccurrence)
        observed = cooccurrence > 0
        ratio = (
            cooccurrence[observed]
            * total
            / expected_denominator[observed]
        )
        ppmi[observed] = np.maximum(np.log2(ratio), 0.0)
        return ppmi

    def transform_one(self, text: str) -> np.ndarray:
        if not self.word_embeddings.size:
            raise RuntimeError("必须先使用训练文本调用 fit()")

        vectors = [
            self.word_embeddings[self.word_to_index[token]]
            for token in tokenize(text)
            if token in self.word_to_index
        ]
        if not vectors:
            return np.zeros(self.word_embeddings.shape[1], dtype=np.float64)

        sentence_vector = np.mean(vectors, axis=0)
        norm = np.linalg.norm(sentence_vector)
        return sentence_vector / norm if norm else sentence_vector

    def transform(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.transform_one(text) for text in texts])

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)
