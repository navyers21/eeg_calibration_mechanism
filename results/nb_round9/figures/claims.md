| Kind | Claim | Mean (points) | 95% CI | Improved | Wilcoxon p | Evidence | Limitation |
|---|---|---|---|---|---|---|---|
| calibration | 500-image calibration (rigid) beats the zero-label pipeline, s4R4, batch CSLS (locked, seed 2026) | +3.89 | +2.40 to +5.40 | 9/9 | 0.0039 | round 7 confirm, 5 draws | same 9 people as development; one dataset |
| calibration | same, each query scored alone | +2.42 | +0.71 to +4.27 | 7/9 | 0.0391 | round 7 confirm | as above |
| calibration | same, single-presentation queries (s4R1) | +0.58 | -0.22 to +1.33 | 6/9 | 0.25 | round 7 confirm | small or no gain: the consumer-like single-view case is not solved |
| calibration | same-day calibration | +4.08 | +2.63 to +5.58 | 9/9 | 0.0039 | round 7 confirm | as above |
| calibration | fine-tuning + rigid beats fine-tuning alone at 4000 images | +2.39 | +1.26 to +3.54 | 7/9 | 0.0156 | round 7 combo | 8,000 labeled presentations; 1 draw |
| calibration | fine-tuning + rigid vs rigid alone at 500 images | +0.04 | -0.76 to +0.83 | 4/9 | 0.8438 | round 7 combo | no gain: fine-tuning does not help at small budgets |
| gallery | predicted population gallery beats the spectrum-matched image-only gallery (zero labels, scored alone) | +1.65 | +0.29 to +3.18 | 6/9 | 0.1289 | round 8 gallery test | image-only controls cover whitening, spectrum and rank, not every possible metric |
| gallery | predicted population gallery beats the same ridge with pairs shuffled within concept | +2.99 | +1.74 to +4.41 | 9/9 | 0.0039 | round 8 gallery test | as above |
| gallery | measured source templates beat predicted ones (headroom; privileged) | +9.94 | +7.54 to +12.50 | 9/9 | 0.0039 | round 8 gallery test | requires earlier people to have seen every candidate |
| gallery | retrieval-trained template predictor vs ridge, rigid k = 500 | -1.23 | -3.28 to +0.96 | 3/9 | 0.25 | round 8 gallery test | trained on source people and held concepts only |
| calibration | calibrated cross-subject decoder beats a decoder trained from scratch on the same 500 images | +9.02 | +6.58 to +11.46 | 9/9 | 0.0039 | round 8 scratch control | scratch recipe tuned only by epochs (person 1) |
