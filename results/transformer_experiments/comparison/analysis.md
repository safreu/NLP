# Dual-decoder Transformer experiment analysis

## Scope and evidence

E1-E9 are completed controlled CPU runs on the same WikiLarge split: 2,000 training pairs, 200
validation pairs, 191 test pairs, five epochs, batch size 8, and seed 42. E0 is an existing
historical result and was not retrained. Reported differences are observations under this single
seed and small-data setting; they do not establish effects outside it.

## Attention corrections and the controlled baseline

E1 corrects per-head scaling and combines causal with target-padding masking. Relative to the
historical E0, E1 increased SARI from 20.6740 to 20.7426 and BERTScore F1 from 0.8239 to 0.8316,
while test loss increased slightly from 7.5161 to 7.5497. Because E0 also differs in embedding size
and head count, this comparison does not isolate the correction causally. E1 is the appropriate
baseline for the new architecture.

## Effect of the reconstruction decoder and weight

Among E2-E4, lambda 1.00 achieved the strongest SARI (20.9681), BERTScore F1 (0.8154), readability
grade (9.1391), and validation simplification loss (6.8546). The balanced rank rule therefore
selected lambda 1.00. The raw weighted validation and test totals increase with lambda by
construction and are not evidence that stronger reconstruction inherently trains worse.

Compared with E1, none of E2-E4 improved BERTScore, ROUGE-L, token F1, or readability. The largest
spaCy entity-preservation estimate in the weight sweep was 0.0423 at lambda 0.25 versus 0.0317 for
E1, but the absolute rates are very low. Thus the evidence does not support a broad claim that the
reconstruction decoder improved information preservation. At most, it produced a small entity
retention change while sacrificing semantic and readability metrics.

## SwiGLU ablation

E5 should be compared with E4 because both use lambda 1.00, learned positions, and eight heads.
SwiGLU increased parameters from 18,811,284 to 19,600,788 and training time from 127.6 to 139.0
seconds. It improved BERTScore F1 (0.8238 versus 0.8154), ROUGE-L (0.1503 versus 0.1416), token F1
(0.1610 versus 0.1515), and FK grade (8.42 versus 9.14), but reduced SARI (20.7668 versus 20.9681).
Observed evidence therefore favors SwiGLU for semantic overlap and readability, not for SARI or
efficiency.

## RoPE ablation

E6 replaced learned positions in E4 with RoPE. It attained the highest SARI in the experiment
(21.0348) and a lower total test loss, but BERTScore F1 fell to 0.8075, ROUGE-L to 0.1293, token F1
to 0.1366, and FK grade worsened sharply to 17.41. It also took 150.2 seconds, the slowest training
run. RoPE was not selected because the isolated SARI improvement conflicted with semantic overlap
and readability.

## Attention-head ablation

At embedding size 256, E7, E4, and E8 use head dimensions 64, 32, and 16 respectively. Four heads
gave SARI 20.9468, BERTScore F1 0.8264, and FK grade 7.86. Sixteen heads gave lower SARI (20.7258)
but higher BERTScore F1 (0.8299), ROUGE-L (0.1542), BLEU (0.0057), token F1 (0.1668), entity
preservation (0.0423), and easier output (FK grade 5.64). Sixteen heads won the available-metric
rank sum by one point over four heads. This result demonstrates a trade-off rather than monotonic
benefit from more heads.

## Final combined model

The selected E9 configuration is lambda 1.00, ReLU, learned positions, 16 heads, and embedding size
256. E9 was retrained from scratch and exactly reproduced E8's deterministic metrics, as expected.
Relative to E1, E9 improved ROUGE-L (0.1542 versus 0.1518), BLEU (0.0057 versus 0.0032), token F1
(0.1668 versus 0.1659), and entity preservation (0.0423 versus 0.0317). E1 remained better on SARI
(20.7426 versus 20.7258), BERTScore F1 (0.8316 versus 0.8299), and FK grade (4.28 versus 5.64), and
used 39% fewer parameters (11.40M versus 18.79M).

Consequently, E9 is the best tested *dual-decoder* configuration under the predeclared balanced
rule, but it did not improve consistently over the corrected baseline. A report should present E9
as the architectural ablation winner and E1 as the stronger efficiency/readability comparator.

## Qualitative observations

The fixed first 20 test examples were used for the consolidated comparison, avoiding success-only
cherry-picking. The files include long sentences, dates/days, lexical changes, compression, and
deletion. Most outputs are generic, repetitive, hallucinated, or excessively deleted. Many
reconstructions also fail to recover source content. These errors are consistent with training a
large word-level vocabulary on only 2,000 pairs and using greedy decoding.

The test representation contains no numeric tokens, so number preservation is `NA`, not 1.0. The
text is lowercased, which also weakens named-entity recognition; entity preservation uses spaCy NER
and must be interpreted as a diagnostic estimate rather than a definitive factuality score.

## Limitations and next experiments

- One random seed gives no uncertainty estimate.
- The controlled data size is too small for reliable open-vocabulary generation.
- Whitespace tokenization causes sparsity and generic-output collapse.
- Greedy decoding was not compared with beam search or constrained decoding.
- The reconstruction decoder adds roughly 7.4 million parameters.
- WikiLarge normalization prevents number-preservation evaluation on this test representation.

A server follow-up should retain E1 and E9, use multiple seeds and a subword tokenizer, then run a
larger-data experiment. That follow-up should not replace the controlled ablations reported here.
