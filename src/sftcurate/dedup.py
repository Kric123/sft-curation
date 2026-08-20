"""Near-duplicate detection for instruction-tuning data.

Exact-match dedup catches almost nothing. Fine-tuning sets are full of pairs
that differ by a name, a date, or a rephrased opening while teaching the model
the same thing, and those inflate the apparent size of a dataset without adding
signal.

This uses character n-gram shingles and Jaccard similarity, with MinHash to keep
the comparison from being quadratic. No embeddings and no model, so it runs
anywhere and costs nothing, at the price of catching paraphrase only when the
surface forms overlap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["DuplicateGroups", "shingles", "jaccard", "find_near_duplicates"]

_MERSENNE = (1 << 61) - 1


def shingles(text: str, k: int = 5) -> frozenset[str]:
    """Character k-grams of ``text``, lowercased with whitespace collapsed.

    Character shingles rather than word shingles because they degrade gracefully
    on typos and morphology, which word-level matching does not.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    normalised = " ".join(text.lower().split())
    if len(normalised) < k:
        return frozenset([normalised]) if normalised else frozenset()
    return frozenset(normalised[i : i + k] for i in range(len(normalised) - k + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Exact Jaccard similarity between two shingle sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _minhash_signatures(sets: list[frozenset[str]], n_perm: int, seed: int) -> np.ndarray:
    """One signature row per document, ``n_perm`` wide.

    Each column is an independent hash permutation; the entry is the minimum
    hash over that document's shingles. The share of matching columns estimates
    Jaccard, which is what makes the banding step below sound.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _MERSENNE, size=n_perm, dtype=np.int64)
    b = rng.integers(0, _MERSENNE, size=n_perm, dtype=np.int64)

    sig = np.full((len(sets), n_perm), np.iinfo(np.int64).max, dtype=np.int64)
    for i, sh in enumerate(sets):
        if not sh:
            continue
        base = np.array([hash(s) & _MERSENNE for s in sh], dtype=np.int64)
        hashed = (np.outer(base, a) + b) % _MERSENNE
        sig[i] = hashed.min(axis=0)
    return sig


@dataclass(frozen=True)
class DuplicateGroups:
    """Clusters of near-identical examples."""

    groups: tuple[tuple[int, ...], ...]
    threshold: float
    n_examples: int

    @property
    def n_duplicates(self) -> int:
        """How many examples would be dropped, keeping one per group."""
        return sum(len(g) - 1 for g in self.groups)

    @property
    def duplicate_rate(self) -> float:
        return self.n_duplicates / self.n_examples if self.n_examples else 0.0

    def survivors(self) -> list[int]:
        """Indices to keep: everything outside a group, plus the first of each."""
        drop = {i for g in self.groups for i in g[1:]}
        return [i for i in range(self.n_examples) if i not in drop]


def find_near_duplicates(
    texts: Sequence[str],
    threshold: float = 0.80,
    k: int = 5,
    n_perm: int = 128,
    bands: int = 32,
    seed: int = 0,
) -> DuplicateGroups:
    """Group examples whose Jaccard similarity is at least ``threshold``.

    MinHash banding proposes candidate pairs, then every candidate is confirmed
    with exact Jaccard. The banding step is a recall optimisation and can miss
    pairs; the confirmation step means it never reports a pair that is not
    genuinely above the threshold.

    Args:
        texts: the examples, already rendered to a single string each.
        threshold: minimum Jaccard to call two examples duplicates.
        k: shingle size.
        n_perm: MinHash permutations. More is slower and more accurate.
        bands: candidate-generation bands. Must divide ``n_perm``. Fewer bands
            means fewer candidates and lower recall.
        seed: for reproducibility.

    Returns:
        A DuplicateGroups. Groups are transitive: A~B and B~C puts all three
        together even when A and C fall below the threshold.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if n_perm % bands:
        raise ValueError(f"bands ({bands}) must divide n_perm ({n_perm})")
    n = len(texts)
    if n == 0:
        raise ValueError("need at least one example")

    sets = [shingles(t, k) for t in texts]
    sig = _minhash_signatures(sets, n_perm, seed)
    rows = n_perm // bands

    candidates: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = {}
        chunk = sig[:, band * rows : (band + 1) * rows]
        for i in range(n):
            buckets.setdefault(chunk[i].tobytes(), []).append(i)
        for members in buckets.values():
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    candidates.add((members[x], members[y]))

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in candidates:
        if jaccard(sets[i], sets[j]) >= threshold:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    groups = tuple(tuple(sorted(v)) for v in clusters.values() if len(v) > 1)
    return DuplicateGroups(
        groups=tuple(sorted(groups)), threshold=threshold, n_examples=n
    )
