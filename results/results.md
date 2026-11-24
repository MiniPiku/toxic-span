# Results

F1 = official SemEval-2021 Task 5 metric (mean per-post character F1). Precision / Recall = micro-averaged over all characters. Latency for local models is amortised batch inference on this machine's CPU; for Claude it is the measured API round-trip of the original (uncached) call.

## Full test set (2000 posts)

| Model | F1 | Precision | Recall | Avg latency per post | Posts evaluated |
|---|---|---|---|---|---|
| Transformer (distilroberta-base) | 0.6633 | 0.5302 | 0.7719 | 25.86 ms | 2000 |
| Lexicon baseline | 0.6198 | 0.4828 | 0.7262 | 0.03 ms | 2000 |

## Same-subset comparison: first 300 test posts

| Model | F1 | Precision | Recall | Avg latency per post | Posts evaluated |
|---|---|---|---|---|---|
| Transformer (distilroberta-base) | 0.7275 | 0.5610 | 0.8753 | 25.86 ms | 300 |
| Lexicon baseline | 0.6773 | 0.5582 | 0.7924 | 0.03 ms | 300 |

## Not evaluated

- **claude_0shot**: no ANTHROPIC_API_KEY and no complete cache for the subset (run `make llm`)
- **claude_6shot**: no ANTHROPIC_API_KEY and no complete cache for the subset (run `make llm`)
