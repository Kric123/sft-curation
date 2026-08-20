"""Datasets whose duplicate and noise rates are known by construction.

Building fixtures for a deduplicator is harder than it looks. A generator that
cycles a few templates produces a corpus that is already full of duplicates
before any are injected, and the detector then looks broken when it is the data
that is wrong. The generator here samples from vocabularies large enough that
collisions are vanishingly unlikely, and asserts uniqueness before returning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Corpus", "make_corpus"]

_VERBS = ["summarise", "translate", "classify", "extract", "rewrite", "compare",
          "outline", "critique", "expand", "simplify", "annotate", "rank"]
_OBJECTS = ["the quarterly filing", "a support transcript", "this patch note",
            "the incident report", "a lease agreement", "the lab protocol",
            "a shipping manifest", "the onboarding guide", "a grant abstract",
            "this pull request", "the risk register", "a clinical summary"]
_QUALIFIERS = ["for a non-technical reader", "in under fifty words", "as bullet points",
               "preserving every figure", "without any hedging", "in the original register",
               "flagging anything ambiguous", "with citations to the source",
               "grouped by severity", "keeping the original ordering"]


@dataclass(frozen=True)
class Corpus:
    """Examples with a known duplicate rate and known low-quality rate."""

    texts: tuple[str, ...]
    duplicate_of: dict[int, int] = field(default_factory=dict)
    low_quality: frozenset[int] = frozenset()

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def n_duplicates(self) -> int:
        return len(self.duplicate_of)

    @property
    def duplicate_rate(self) -> float:
        return self.n_duplicates / len(self.texts) if self.texts else 0.0


def _unique_base(n: int, rng: np.random.Generator) -> list[str]:
    """Distinct instructions sampled from a space far larger than ``n``."""
    space = len(_VERBS) * len(_OBJECTS) * len(_QUALIFIERS)
    if n > space // 4:
        raise ValueError(
            f"asked for {n} unique examples from a space of {space}; "
            "collisions would make the corpus duplicate-laden before injection"
        )
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < n:
        text = "%s %s %s." % (
            _VERBS[rng.integers(len(_VERBS))],
            _OBJECTS[rng.integers(len(_OBJECTS))],
            _QUALIFIERS[rng.integers(len(_QUALIFIERS))],
        )
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def make_corpus(
    n_unique: int = 200,
    duplicate_rate: float = 0.15,
    near_duplicate_share: float = 0.5,
    low_quality_rate: float = 0.0,
    seed: int = 0,
) -> Corpus:
    """Build a corpus with an exactly known duplicate count.

    Args:
        n_unique: distinct examples before injection.
        duplicate_rate: duplicates as a share of the *final* corpus.
        near_duplicate_share: of the injected duplicates, how many are
            paraphrases rather than exact copies.
        low_quality_rate: share of the final corpus that is degenerate --
            empty, truncated, or a single repeated token.
        seed: for reproducibility.

    Returns:
        A Corpus. ``duplicate_of`` maps each injected duplicate to the index it
        was copied from, so detection can be scored exactly.
    """
    if not 0.0 <= duplicate_rate < 1.0:
        raise ValueError("duplicate_rate must be in [0, 1)")
    if not 0.0 <= near_duplicate_share <= 1.0:
        raise ValueError("near_duplicate_share must be in [0, 1]")
    if not 0.0 <= low_quality_rate < 1.0:
        raise ValueError("low_quality_rate must be in [0, 1)")

    rng = np.random.default_rng(seed)
    texts = _unique_base(n_unique, rng)

    n_dup = int(round(n_unique * duplicate_rate / max(1e-9, 1.0 - duplicate_rate)))
    duplicate_of: dict[int, int] = {}
    for _ in range(n_dup):
        src = int(rng.integers(n_unique))
        text = texts[src]
        if rng.random() < near_duplicate_share:
            # A paraphrase that keeps most surface form, as real near-dupes do.
            text = text.replace("the ", "that ", 1).replace(".", " please.")
        duplicate_of[len(texts)] = src
        texts.append(text)

    low_quality: set[int] = set()
    n_bad = int(round(len(texts) * low_quality_rate))
    for _ in range(n_bad):
        idx = int(rng.integers(len(texts)))
        if idx in duplicate_of or idx in low_quality:
            continue
        texts[idx] = rng.choice(["", "...", "ok", "x " * 40]).strip()
        low_quality.add(idx)

    return Corpus(tuple(texts), duplicate_of, frozenset(low_quality))
