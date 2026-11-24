# Error analysis

Best model on the full test set: **Transformer (distilroberta-base)** (F1 = 0.6633, n = 2000).

Offensive words are masked (first letter kept, vowels replaced by `*`). A word counts as *marked* when at least half of its characters are marked.

## Top 10 false positives (word marked by the model, not in gold) - 1588 occurrences in total

Function words are excluded from both top-10 tables.

| Word (masked) | Count | Toxic ratio in train |
|---|---|---|
| st*p*d | 100 | 0.86 |
| ign*r*nt | 51 | 0.70 |
| id**t | 40 | 0.91 |
| d*mb | 28 | 0.75 |
| f**l | 28 | 0.75 |
| r*c*st | 25 | 0.44 |
| m*r*n | 25 | 0.85 |
| id**ts | 23 | 0.92 |
| p*th*t*c | 23 | 0.75 |
| cr*p | 22 | 0.72 |

## Top 10 false negatives (gold word the model missed) - 576 occurrences in total

| Word (masked) | Count | Occurrences in train |
|---|---|---|
| weapons | 9 | 30 |
| chemical | 6 | 5 |
| g*rb*g* | 5 | 113 |
| assad | 5 | 6 |
| l*s*r | 4 | 170 |
| people | 3 | 1120 |
| trump | 3 | 1417 |
| back | 3 | 276 |
| wh*t* | 3 | 329 |
| l**r | 3 | 162 |

Function words (e.g. *and*, *the*, *are*) are left out of both tables: they account for 186 missed occurrences (32.3% of false negatives), almost all inside multi-word gold phrases.

## F1 by text length and by presence of toxic spans (full test set)

| Group | Posts | Lexicon baseline | Transformer (distilroberta-base) |
|---|---|---|---|
| Length < 100 chars | 780 | 0.730 | 0.759 |
| Length 100-199 | 583 | 0.623 | 0.657 |
| Length 200-399 | 430 | 0.541 | 0.601 |
| Length >= 400 | 207 | 0.357 | 0.450 |
| Posts with toxic spans | 1606 | 0.758 | 0.819 |
| Posts with no toxic spans | 394 | 0.058 | 0.030 |

## Observations

1. **The metric punishes any prediction on clean posts.** 394 of 2000 test posts (19.7%) have no gold span; the model flagged something in 382 of them (97.0%), and each of those scores F1 = 0. Many such posts are offensive overall but annotators did not tie the toxicity to specific words (context-dependent toxicity).
2. **False positives are mostly genuinely rude words that were not annotated in that post.** 65.2% of false-positive word occurrences are words that are marked toxic in at least 30% of their training occurrences (e.g. st*p*d, ign*r*nt, id**t). Whether such a word is 'the toxic span' depends on context and annotator judgement (e.g. self-directed or quoted use, sarcasm), which a word list cannot see and a model trained on ~8k posts learns only partly.
3. **Many missed words are harmless out of context; fewer are rare or obfuscated.** 10.6% of missed word occurrences appear fewer than 3 times in training and 17 of 1850 gold spans (0.9%) contain obfuscation (asterisks or symbols inside a word). Several of the most-missed words are neutral on their own (weapons, chemical, assad, people): they are toxic only as part of a threat or accusation, which needs context rather than a word list.
4. **Boundary disagreement is common.** 594 posts (29.7%) get partial credit (0 < F1 < 1); 43 gold spans (2.3%) are phrases of 4+ words where annotators marked a whole insult or threat, while models mark only the strongest word.
5. **Longer posts are harder.** F1 by length bucket for the best model: < 100 chars: 0.759, 100-199: 0.657, 200-399: 0.601, >= 400: 0.450. Long posts give more chances for both extra and missed words; posts over 256 tokens are also truncated for the transformer.
