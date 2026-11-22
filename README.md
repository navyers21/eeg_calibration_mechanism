# Cross-subject and within-person checkpoints (THINGS-EEG2, posterior17)

Exported by `scripts/export_ckpts.py` on the main branch; scores and calibration images are in `manifest.json`.

```python
from ckpt_loader import load_stage            # torch + numpy only
m = load_stage('.', person=3, stage='rot500')
z = m(x)                                       # x: (n, 17, 250), our preprocessing; z: (n, 512)
top1 = (F.normalize(z, dim=-1) @ m.gallery_test.T).argmax(1)
```

| stage | what it is | scored against |
|---|---|---|
| f0 | repaug EEGProject trained on the other 9 people (the decoder every calibration round uses) | image gallery |
| zero | f0 + label-free layer-1 offset from 500 calibration images' EEG | population gallery |
| rot100 | offset + output rotation shrunk toward identity, 100 labeled images (~40 s) | population gallery |
| rot500 | same with 500 images (~3 min) | population gallery |
| within | trained on the person's own pool (nb_train recipe, used in the interp experiments) | image gallery |
| within_rep | f0's exact recipe and optimizer steps on the person's own pool, if present | image gallery |

All stages share layer names (encoder.model[0..2], eeg_projector, out), so hooks line up. In calibrated stages the
layer-1 offset is folded into encoder.model[0].bias, so activations from layer 1 on are the calibrated ones; the
rotation is the `out` layer (identity elsewhere). The rotation only acts after the projector, so anything measured
before `out` sees the offset but not the rotation.

Calibration is fit on session 3 (2 presentations per image); session-4 queries and the 80-presentation benchmark are
untouched. Weights are fp16. Seed 2025 for every cross-subject stage. People 2-10 are the evaluation cohort; person 1
was used for tuning choices in earlier rounds.

To rebuild the gallery for other images: population gallery = normalize((feats - pop_ridge.xm) @ pop_ridge.B + pop_ridge.ym),
image gallery = normalize(img_projector(feats)), with feats = NeuroBridge's 1024-d augmented CLIP features. Both are
stored in f0.pt (and within.pt has its own img_projector).
