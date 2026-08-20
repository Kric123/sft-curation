# sft-curation

Near-duplicate detection for instruction-tuning datasets — and an honest account of what the similarity threshold actually decides.

Exact-match dedup catches almost nothing. Fine-tuning sets are full of pairs that differ by a name, a date, or a rephrased opening while teaching the model the same thing. Those inflate the apparent size of a dataset without adding signal.

```bash
pip install -e ".[dev]"
pytest                                  # 15 tests
python examples/threshold_sweep.py
```

## The finding

**The threshold is not a tuning knob. It is the definition of "duplicate."**

Injected paraphrases in the test corpus sit at a median Jaccard of **0.69** against their source. Here is what happens as the threshold moves:

| Threshold | Flagged | Recall | Precision | |
|---|---|---|---|---|
| 0.60 | 186 | 1.00 | 0.24 | groups unrelated rows sharing a template |
| 0.70 | 168 | 0.80 | 0.21 | |
| 0.75 | 109 | 0.73 | 0.29 | |
| **0.80** | **40** | **0.70** | **0.78** | |
| 0.85 | 26 | 0.59 | **1.00** | excludes paraphrases below the bar |
| 0.90 | 25 | 0.57 | 1.00 | |

That "recall 0.70" at threshold 0.80 is **not a detector failure.** The paraphrases are 69% similar; the caller asked for 80%. The detector correctly excluded them. Recall only means something against a definition of duplicate, and the threshold *is* that definition — so measuring recall against a fixture that disagrees with the threshold is a category error.

**What is assertable, and what the tests check:** everything flagged is genuinely above the threshold. Precision at 0.90 is ≥0.95. Nothing beyond that is a property of the algorithm rather than of your definition.

Pick the threshold from this curve against what a duplicate means for *your* data, then report which one you used.

## How it works

Character n-gram shingles, Jaccard similarity, MinHash banding to avoid comparing every pair. No embeddings and no model, so it runs anywhere and costs nothing.

Banding proposes candidate pairs; **every candidate is then confirmed with exact Jaccard.** Banding is a recall optimisation and can miss pairs, but it can never cause a pair below the threshold to be reported.

Character shingles rather than word shingles, because they degrade gracefully on typos and morphology where word matching does not.

```python
from sftcurate import find_near_duplicates

result = find_near_duplicates(texts, threshold=0.80)
result.duplicate_rate     # share of the corpus that is redundant
result.groups             # clusters of near-identical examples
result.survivors()        # indices to keep: one per group, plus everything unique
```

Grouping is **transitive**: if A~B and B~C, all three cluster together even when A and C fall below the threshold. That is usually what you want for dedup and occasionally is not — a long chain of small edits can merge two genuinely different examples.

## On building fixtures for a detector

The generator in `synthetic.py` exists because the first one was wrong, and wrong in a way that made the detector look broken.

It built examples by cycling small lists — 5 names, 3 days, 3 offices, 4 topics — which repeats every 60 rows. Across 200 rows that produced roughly **140 unintended exact duplicates before a single one was injected.** The detector flagged 248 of 260 examples and looked hopeless. It was correct; the fixture was not.

`make_corpus` now samples from a space at least 4× the requested size and **raises rather than returning a collision-prone corpus**:

```python
make_corpus(n_unique=10_000)
# ValueError: asked for 10000 unique examples from a space of 1440;
# collisions would make the corpus duplicate-laden before injection
```

That check is itself a test — `test_corpus_generator_refuses_a_collision_prone_request`.

**The general lesson:** when you build a tool that detects a problem, your fixtures need more care than the tool does. A fixture that accidentally contains the thing you are detecting will make a correct implementation look broken, and the instinct is to "fix" the implementation.

## Tests

15 tests. The ones carrying the argument:

- `test_threshold_defines_what_a_duplicate_is` — explains why no recall figure is asserted, and checks the claim that *is* assertable: every group contains a pair genuinely above the threshold
- `test_exact_duplicates_are_always_found` — Jaccard 1.0 clears every threshold, so these can never be missed
- `test_grouping_is_transitive` — A~B~C clusters together even when A~C is below threshold
- `test_fewer_flagged_as_threshold_rises` — an invariant; a detector ignoring its threshold would pass everything else
- `test_corpus_generator_refuses_a_collision_prone_request` — the fixture bug, encoded so it cannot recur

## Limitations

- **Surface-form only.** Two examples teaching the same thing in entirely different words will not be caught. Embedding-based dedup catches those and costs a model; this deliberately does not.
- **Quality filtering is not implemented.** `make_corpus` can inject degenerate examples (empty, truncated, single repeated token) and the `low_quality` labels exist, but no filter consumes them yet. That is the obvious next module.
- **No downstream impact measurement.** This tells you what is redundant, not whether removing it improves the fine-tune. Answering that needs a train-and-evaluate loop, and until then dedup is a hypothesis, not a result.
- **Transitive grouping can over-merge.** A chain of small edits can link two examples that share nothing directly.
- **`hash()` is used for shingles**, so signatures are not stable across Python processes unless `PYTHONHASHSEED` is fixed. Results within a run are deterministic; across runs they are not.

## How this was built

Written with AI assistance (Claude), reviewed line by line before release. The part worth scrutinising is the recall discussion. The first instinct on seeing recall 0.70 was that the detector needed fixing; the correct reading was that the fixture and the threshold disagreed about what a duplicate is. Getting that backwards would have meant tuning the threshold until the number looked good, which is exactly the failure this kind of tool exists to prevent.

## License

MIT
