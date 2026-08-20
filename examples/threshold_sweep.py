"""What threshold to pick, and why the question has no single answer.

Run:  python examples/threshold_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sftcurate import find_near_duplicates, jaccard, make_corpus, shingles  # noqa: E402


def main() -> None:
    corpus = make_corpus(n_unique=250, duplicate_rate=0.15, near_duplicate_share=0.5, seed=1)
    truth = set(corpus.duplicate_of)

    near = [i for i in truth if corpus.texts[i] != corpus.texts[corpus.duplicate_of[i]]]
    sims = sorted(
        jaccard(shingles(corpus.texts[i]), shingles(corpus.texts[corpus.duplicate_of[i]]))
        for i in near
    )
    print("\nHow similar are the injected paraphrases?")
    print("  min %.2f   median %.2f   max %.2f   (n=%d)"
          % (sims[0], sims[len(sims) // 2], sims[-1], len(sims)))

    print("\nThreshold sweep on %d examples with %d injected duplicates\n" % (len(corpus), len(truth)))
    print("  %-11s %-9s %-9s %-11s %s" % ("threshold", "flagged", "recall", "precision", "note"))
    print("  " + "-" * 72)
    for th in (0.60, 0.70, 0.75, 0.80, 0.85, 0.90):
        res = find_near_duplicates(corpus.texts, threshold=th)
        flagged = {i for g in res.groups for i in g[1:]}
        hit = len(flagged & truth)
        note = ""
        if th <= 0.70:
            note = "groups unrelated rows sharing a template"
        if th >= 0.85:
            note = "excludes paraphrases below the bar"
        print("  %-11.2f %-9d %-9.2f %-11.2f %s"
              % (th, len(flagged), hit / len(truth),
                 hit / len(flagged) if flagged else 1.0, note))

    print()
    print("=" * 74)
    print(
        "The paraphrases sit around 0.69 similarity. At 0.80 the detector\n"
        "excludes most of them -- correctly, because they are not 80% similar.\n"
        "That shows up as 'recall 0.70' only because the fixture calls them\n"
        "duplicates and the threshold does not.\n\n"
        "The threshold is the definition. Pick it from this curve against what\n"
        "a duplicate means for your data, then report which one you used."
    )
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
