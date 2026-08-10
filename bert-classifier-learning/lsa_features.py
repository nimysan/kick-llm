"""方法 5：使用 SVD 将 TF-IDF 句子向量压缩为 LSA 稠密特征。"""

import numpy as np

from tfidf_features import TfidfVectorizer


class LsaVectorizer:
    """先生成 TF-IDF，再用训练集学到的 SVD 方向压缩维度。"""

    def __init__(self, n_components: int = 10) -> None:
        self.n_components = n_components
        self.tfidf = TfidfVectorizer()
        self.components = np.empty((0, 0), dtype=np.float64)
        self.singular_values = np.array([], dtype=np.float64)
        self.explained_energy_ratio = 0.0

    def fit(self, texts: list[str]) -> None:
        tfidf_matrix = self.tfidf.fit_transform(texts)
        _, singular_values, right_vectors = np.linalg.svd(
            tfidf_matrix,
            full_matrices=False,
        )

        max_components = min(tfidf_matrix.shape)
        component_count = min(self.n_components, max_components)
        self.components = right_vectors[:component_count]
        self.singular_values = singular_values

        total_energy = float(np.sum(singular_values**2))
        retained_energy = float(np.sum(singular_values[:component_count] ** 2))
        self.explained_energy_ratio = retained_energy / total_energy

    def transform_one(self, text: str) -> np.ndarray:
        if not self.components.size:
            raise RuntimeError("必须先使用训练文本调用 fit()")
        tfidf_vector = self.tfidf.transform_one(text)
        return tfidf_vector @ self.components.T

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self.components.size:
            raise RuntimeError("必须先使用训练文本调用 fit()")
        tfidf_matrix = self.tfidf.transform(texts)
        return tfidf_matrix @ self.components.T

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)
