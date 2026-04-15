from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_PATTERN.findall(text):
        token = raw.lower()
        tokens.append(token)
        if any("\u4e00" <= char <= "\u9fff" for char in token):
            tokens.extend(char for char in token if "\u4e00" <= char <= "\u9fff")
    return tokens


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


@dataclass(slots=True)
class LocalHashEmbeddingProvider:
    dimensions: int = 96

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = tokenize(text)
        if not tokens:
            return vector

        counts = Counter(tokens)
        for token, weight in counts.items():
            index = hash(token) % self.dimensions
            sign = 1.0 if hash(f"{token}:sign") % 2 == 0 else -1.0
            vector[index] += sign * float(weight)

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
