"""Near-duplicate detection for instruction-tuning datasets."""

from .dedup import DuplicateGroups, find_near_duplicates, jaccard, shingles
from .synthetic import Corpus, make_corpus

__version__ = "0.1.0"

__all__ = [
    "Corpus",
    "DuplicateGroups",
    "find_near_duplicates",
    "jaccard",
    "make_corpus",
    "shingles",
]
