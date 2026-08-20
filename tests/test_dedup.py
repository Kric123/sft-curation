"""Tests for near-duplicate detection.

Read test_threshold_defines_what_a_duplicate_is first. It explains why this file
does not assert a recall figure: recall is only meaningful against a definition
of "duplicate", and the threshold *is* that definition.
"""

import pytest

from sftcurate.dedup import find_near_duplicates, jaccard, shingles
from sftcurate.synthetic import make_corpus


# --- similarity primitives ---------------------------------------------------

def test_jaccard_bounds():
    a = shingles("the quick brown fox")
    assert jaccard(a, a) == 1.0
    assert jaccard(a, shingles("zzzzzzzzzzzzzzzz")) == 0.0
    assert jaccard(frozenset(), frozenset()) == 1.0
    assert jaccard(a, frozenset()) == 0.0


def test_shingles_normalise_case_and_whitespace():
    assert shingles("Hello   World") == shingles("hello world")


def test_shingles_handle_text_shorter_than_k():
    assert shingles("ab", k=5) == frozenset({"ab"})
    assert shingles("", k=5) == frozenset()


def test_shingle_count_is_predictable():
    text = "abcdefg"
    assert len(shingles(text, k=3)) == len(text) - 3 + 1


# --- detection ---------------------------------------------------------------

def test_exact_duplicates_are_always_found():
    """Jaccard 1.0 clears every threshold, so these can never be missed."""
    texts = ["summarise the filing.", "translate the memo.", "summarise the filing."]
    res = find_near_duplicates(texts, threshold=0.99)
    assert res.n_duplicates == 1
    assert res.groups == ((0, 2),)


def test_survivors_keep_one_per_group():
    texts = ["a b c d e f g", "a b c d e f g", "a b c d e f g", "totally other content"]
    res = find_near_duplicates(texts, threshold=0.95)
    assert sorted(res.survivors()) == [0, 3]


def test_grouping_is_transitive():
    """A~B and B~C puts all three together even if A and C fall below threshold."""
    texts = ["aaaaaaaaaabbbbbbbbbb", "aaaaaaaaaabbbbbbbbbc", "aaaaaaaaaabbbbbbbbcc"]
    res = find_near_duplicates(texts, threshold=0.70)
    assert len(res.groups) == 1
    assert res.groups[0] == (0, 1, 2)


def test_unrelated_text_is_not_grouped():
    texts = [
        "summarise the quarterly filing for a non-technical reader.",
        "rank a shipping manifest grouped by severity.",
        "critique the lab protocol without any hedging.",
    ]
    assert find_near_duplicates(texts, threshold=0.80).groups == ()


def test_fewer_flagged_as_threshold_rises():
    c = make_corpus(n_unique=200, duplicate_rate=0.15, seed=1)
    counts = [
        find_near_duplicates(c.texts, threshold=t).n_duplicates
        for t in (0.60, 0.70, 0.80, 0.90, 0.99)
    ]
    assert counts == sorted(counts, reverse=True)


def test_threshold_defines_what_a_duplicate_is():
    """Why no recall assertion appears in this file.

    The generator injects paraphrases whose Jaccard against their source has a
    median around 0.69. At threshold 0.80 the detector excludes most of them --
    correctly, because they are not 80% similar. Calling that a recall failure
    would be measuring against a different definition of duplicate than the one
    the caller asked for.

    What is assertable: everything flagged really is above the threshold.
    """
    c = make_corpus(n_unique=200, duplicate_rate=0.20, seed=2)
    res = find_near_duplicates(c.texts, threshold=0.85)
    sets = [shingles(t) for t in c.texts]
    for group in res.groups:
        head = group[0]
        assert any(jaccard(sets[head], sets[other]) >= 0.85 for other in group[1:]), (
            "every group must contain a pair genuinely above the threshold"
        )


def test_precision_is_high_at_a_strict_threshold():
    c = make_corpus(n_unique=250, duplicate_rate=0.15, seed=1)
    res = find_near_duplicates(c.texts, threshold=0.90)
    flagged = {i for g in res.groups for i in g[1:]}
    truth = set(c.duplicate_of)
    assert flagged, "strict threshold should still catch the exact copies"
    assert len(flagged & truth) / len(flagged) >= 0.95


def test_no_duplicates_injected_means_few_flagged():
    c = make_corpus(n_unique=250, duplicate_rate=0.0, seed=3)
    res = find_near_duplicates(c.texts, threshold=0.90)
    assert res.duplicate_rate < 0.05


# --- validation --------------------------------------------------------------

def test_validation():
    with pytest.raises(ValueError, match=r"threshold must be in \(0, 1\]"):
        find_near_duplicates(["a"], threshold=0.0)
    with pytest.raises(ValueError, match="must divide"):
        find_near_duplicates(["a"], n_perm=128, bands=30)
    with pytest.raises(ValueError, match="at least one example"):
        find_near_duplicates([])
    with pytest.raises(ValueError, match="k must be >= 1"):
        shingles("abc", k=0)


def test_corpus_generator_refuses_a_collision_prone_request():
    """The fixture bug this generator exists to prevent."""
    with pytest.raises(ValueError, match="collisions would make the corpus"):
        make_corpus(n_unique=10_000)


def test_corpus_reports_its_own_duplicate_rate():
    c = make_corpus(n_unique=200, duplicate_rate=0.20, seed=4)
    assert c.duplicate_rate == pytest.approx(0.20, abs=0.02)
    assert c.n_duplicates == len(c.duplicate_of)
