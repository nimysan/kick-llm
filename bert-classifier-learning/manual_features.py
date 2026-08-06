"""方法 1 使用的人工特征提取器。"""

from collections.abc import Sequence

import numpy as np


POSITIVE_WORDS = ("好", "满意", "喜欢", "优秀", "棒", "快", "值得", "推荐", "惊喜")
NEGATIVE_WORDS = ("差", "垃圾", "失望", "糟糕", "慢", "讨厌", "后悔", "坏", "难用")

FEATURE_NAMES = (
    "正面词数量",
    "负面词数量",
    "感叹号数量",
    "是否有正面词",
    "是否有负面词",
    "句子长度/20",
)


def count_keywords(text: str, keywords: Sequence[str]) -> int:
    """统计人工词典中的词在句子里出现了多少次。"""
    return sum(text.count(word) for word in keywords)


def extract_features(text: str) -> np.ndarray:
    """由人规定规则，将任意文本转换成固定的 6 维向量 x。"""
    positive_count = count_keywords(text, POSITIVE_WORDS)
    negative_count = count_keywords(text, NEGATIVE_WORDS)
    exclamation_count = text.count("!") + text.count("！")

    return np.array(
        [
            positive_count,
            negative_count,
            exclamation_count,
            float(positive_count > 0),
            float(negative_count > 0),
            len(text) / 20.0,
        ],
        dtype=np.float64,
    )
