# Wider-model experiment

**Status: running; no final accuracy comparison yet.** Both wider configurations completed a full startup epoch. The queue runs three seeds (42, 43, 44) for each width, including a freshly trained 1x control. New references are excluded from this capacity-only experiment.

| Width | Scalar / vector / tensor channels | Parameters | Startup peak GPU allocation |
| --- | --- | ---: | ---: |
| 1x control | 64 / 32 / 16 | 354,545 | Pending |
| 2x | 128 / 64 / 32 | 1,356,513 | 2,426 MB |
| 4x | 256 / 128 / 64 | 5,288,873 | 4,801 MB |

These are channel multipliers, not parameter multipliers. Blocks remain 4, cutoff 5 A, radial basis 32, attention heads 4, and activation SwiGLU. The 3,156 training and 145 validation frames are unchanged and hashed in the [protocol](../reports/width_scaling/protocol.json).

The schedule retains 200 planned epochs, learning rate 0.001 with warmup/cosine decay, loss weights 1/10/1 for energy/forces/stress, and early stopping after at least 100 epochs with patience 50. Validation force MAE selects checkpoints to isolate width from selection changes. Energy and force MAE/RMSE are recorded together; improved force MAE alone will not justify promoting a model.

Original optimizer batches and their structure weighting are preserved. Each batch is processed in microbatches of at most 256 atoms / 16,000 edges, except indivisible single structures. No structure is dropped to fit a microbatch. A gradient test checks equivalence with unequal atom counts, force derivatives and missing stress labels. This path rejects auxiliary losses because they use a different normalization. Floating-point reduction order can differ from the original baseline, so fresh controls use the same microbatch implementation.

Startup epochs resume with the original 200-epoch learning-rate schedule; they are not separate fits. Final metrics require completion. No test set has been used for selecting these runs and the public baseline has not been replaced.

## Run or resume

```powershell
.venv/Scripts/python.exe -u -m scripts.width_scaling
```

Only start one queue at a time. Checkpoints are saved each epoch under `runs/width_scaling/`. `reports/width_scaling/run.log` contains live epoch output; `active.json` identifies the current trial. Generate a current snapshot with:

```powershell
.venv/Scripts/python.exe -m scripts.width_scaling --report
```

[Progress snapshot](../reports/width_scaling/progress.json) | [Model-capacity rationale](model_capacity.md) | [Replacement alloy references](alloy_references.md)
