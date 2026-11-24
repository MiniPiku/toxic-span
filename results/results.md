# Results

F1 = official SemEval-2021 Task 5 metric (mean per-post character F1). Precision / Recall = micro-averaged over all characters. Latency for local models is amortised batch inference on this machine's CPU; for the LLM it is the measured API round-trip of the original (uncached) call.

## Full test set (2000 posts)

| Model | F1 | Precision | Recall | Avg latency per post | Posts evaluated |
|---|---|---|---|---|---|
| Transformer (distilroberta-base) | 0.6633 | 0.5302 | 0.7719 | 31.07 ms | 2000 |
| Lexicon baseline | 0.6198 | 0.4828 | 0.7262 | 0.03 ms | 2000 |

## Same-subset comparison: first 300 test posts

| Model | F1 | Precision | Recall | Avg latency per post | Posts evaluated |
|---|---|---|---|---|---|
| Transformer (distilroberta-base) | 0.7275 | 0.5610 | 0.8753 | 31.07 ms | 300 |
| Lexicon baseline | 0.6773 | 0.5582 | 0.7924 | 0.03 ms | 300 |
| Gemini gemini-3.5-flash-lite (6-shot) | 0.6329 | 0.4885 | 0.7724 | 999.03 ms | 300 |
| Gemini gemini-3.5-flash-lite (zero-shot) | 0.5890 | 0.3825 | 0.8011 | 911.27 ms | 300 |
