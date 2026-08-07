"""方法 5 使用的 Word2Vec 平均句向量特征提取器。"""

import numpy as np
from gensim.models import Word2Vec

from bag_of_words_features import tokenize


class Word2VecVectorizer:
    """在训练文本上学习词向量，再取已知词向量的平均值。"""

    def __init__(
        self,
        vector_size: int = 50,
        seed: int = 42,
        show_tokenized_texts: bool = False,
    ) -> None:
        self.vector_size = vector_size
        self.seed = seed
        self.show_tokenized_texts = show_tokenized_texts
        self.model: Word2Vec | None = None

    def fit(self, texts: list[str]) -> None:
        tokenized_texts = [tokenize(text) for text in texts]
        if self.show_tokenized_texts:
            print("\ntokenized_texts（传给 Word2Vec 的 26 条分词数据）:")
            for index, tokens in enumerate(tokenized_texts):
                print(f"  [{index:>2}] {tokens}")

        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=3,
            min_count=1,
            workers=1,
            sg=1,
            epochs=500,
            seed=self.seed,
        )

    def transform_one(self, text: str) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("必须先使用训练文本调用 fit()")

        vectors = [
            self.model.wv[token]
            for token in tokenize(text)
            if token in self.model.wv
        ]
        if not vectors:
            return np.zeros(self.vector_size, dtype=np.float64)

        sentence_vector = np.mean(vectors, axis=0).astype(np.float64)
        norm = np.linalg.norm(sentence_vector)
        return sentence_vector / norm if norm else sentence_vector

    def transform(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.transform_one(text) for text in texts])

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    @property
    def vocabulary(self) -> list[str]:
        if self.model is None:
            return []
        return list(self.model.wv.index_to_key)
