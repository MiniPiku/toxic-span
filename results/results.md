# Results

F1 = official SemEval-2021 Task 5 metric (mean per-post character F1). Precision / Recall = micro-averaged over all characters. Latency for local models is amortised batch inference on this machine's CPU; for the LLM it is the measured API round-trip of the original (uncached) call.

## Full test set (2000 posts)

| Model | F1 | Precision | Recall | Avg latency per post | Posts evaluated |
|---|---|---|---|---|---|
| Transformer (distilroberta-base) | 0.6633 | 0.5302 | 0.7719 | 24.44 ms | 2000 |
| Lexicon baseline | 0.6198 | 0.4828 | 0.7262 | 0.03 ms | 2000 |

## Same-subset comparison: first 300 test posts

| Model | F1 | Precision | Recall | Avg latency per post | Posts evaluated |
|---|---|---|---|---|---|
| Transformer (distilroberta-base) | 0.7275 | 0.5610 | 0.8753 | 24.44 ms | 300 |
| Lexicon baseline | 0.6773 | 0.5582 | 0.7924 | 0.03 ms | 300 |
| Gemini gemini-3.5-flash-lite (6-shot) [partial: 184/300 posts] | 0.6555 | 0.5489 | 0.7714 | 1.08 s | 184 |
| Gemini gemini-3.5-flash-lite (zero-shot) | 0.5890 | 0.3825 | 0.8011 | 911.27 ms | 300 |

Rows marked *partial* did not finish (free-tier daily request quota); their F1 is on fewer posts, so compare them only in the next table, where every model is scored on the same posts.

## All models on the first 184 test posts (covered by every LLM run)

| Model | F1 | Precision | Recall | Avg latency per post | Posts evaluated |
|---|---|---|---|---|---|
| Transformer (distilroberta-base) | 0.7773 | 0.6187 | 0.9327 | 24.44 ms | 184 |
| Lexicon baseline | 0.7228 | 0.6176 | 0.8254 | 0.03 ms | 184 |
| Gemini gemini-3.5-flash-lite (6-shot) [partial: 184/300 posts] | 0.6555 | 0.5489 | 0.7714 | 1.08 s | 184 |
| Gemini gemini-3.5-flash-lite (zero-shot) | 0.6333 | 0.4296 | 0.8425 | 911.27 ms | 184 |
