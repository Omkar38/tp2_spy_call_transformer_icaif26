# TP2 Transformer Ablation Colab Commands

These commands implement Professor Polak's training checklist without changing
the main `03_train_mv_alibi_transformer.py` result. Every ablation writes to:

```text
TP2MVTransformer/ablation_runs/<tag>/
```

Run independent commands in separate Colab sessions if you want parallel runs.
Use GPU runtime.

## Setup

```python
!git clone -b fresh-files-2026-05-12 https://github.com/Omkar38/tp2_spy_call_transformer_icaif26.git
%cd tp2_spy_call_transformer_icaif26
!pip install -q -r requirements.txt
```

If the repo is already cloned:

```python
%cd /content/tp2_spy_call_transformer_icaif26
!git pull
```

## 1. Best Validation Loss Checkpoint

This is the first priority because professor explicitly asked to restore the
best validation-loss checkpoint.

```python
!python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py \
  --tag best_loss_lr3e4 \
  --checkpoint-metric val_loss \
  --scheduler warmup_cosine \
  --scheduler-metric val_loss \
  --lr 3e-4 \
  --clip-norm 1.0
```

## 2. Learning-Rate Sweep

Run these in parallel sessions if possible.

```python
!python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py --tag lr_3e5_loss --checkpoint-metric val_loss --scheduler warmup_cosine --lr 3e-5 --clip-norm 1.0
!python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py --tag lr_1e4_loss --checkpoint-metric val_loss --scheduler warmup_cosine --lr 1e-4 --clip-norm 1.0
!python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py --tag lr_3e4_loss --checkpoint-metric val_loss --scheduler warmup_cosine --lr 3e-4 --clip-norm 1.0
!python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py --tag lr_1e3_loss --checkpoint-metric val_loss --scheduler warmup_cosine --lr 1e-3 --clip-norm 1.0
```

## 3. Capacity Ablation

Tests whether the current transformer is too flexible relative to stable signal.

```python
!python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py --tag cap_h32_heads2 --checkpoint-metric val_loss --hidden 32 --heads 2 --lr 3e-4 --clip-norm 1.0
!python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py --tag cap_h64_heads4 --checkpoint-metric val_loss --hidden 64 --heads 4 --lr 3e-4 --clip-norm 1.0
!python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py --tag cap_h128_heads4 --checkpoint-metric val_loss --hidden 128 --heads 4 --lr 3e-4 --clip-norm 1.0
```

## 4. Regularization Ablation

Tests dropout and weight decay.

```python
!python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py --tag reg_dropout_low --checkpoint-metric val_loss --input-dropout 0.05 --head-dropout 0.10 --lr 3e-4 --clip-norm 1.0
!python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py --tag reg_dropout_high --checkpoint-metric val_loss --input-dropout 0.20 --head-dropout 0.30 --lr 3e-4 --clip-norm 1.0
!python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py --tag reg_wd_1e3 --checkpoint-metric val_loss --weight-decay 1e-3 --lr 3e-4 --clip-norm 1.0
```

## 5. History-Length Ablation

Tests whether long windows cause noisy historical attention.

```python
!python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py --tag hist16_loss --checkpoint-metric val_loss --max-history 16 --lr 3e-4 --clip-norm 1.0
!python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py --tag hist32_loss --checkpoint-metric val_loss --max-history 32 --lr 3e-4 --clip-norm 1.0
!python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py --tag hist64_loss --checkpoint-metric val_loss --max-history 64 --lr 3e-4 --clip-norm 1.0
```

## 6. Distance-Bias Ablation

Tests whether MV-ALiBi distance terms help or hurt.

```python
!python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py --tag no_distance_bias --checkpoint-metric val_loss --no-distance-bias --lr 3e-4 --clip-norm 1.0
!python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py --tag no_strategy_bonus --checkpoint-metric val_loss --no-strategy-bonus --lr 3e-4 --clip-norm 1.0
```

## 7. Random-Seed Check

Tests whether results are stable or seed-sensitive.

```python
!python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py --tag seed7_loss --checkpoint-metric val_loss --seed 7 --lr 3e-4 --clip-norm 1.0
!python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py --tag seed42_loss --checkpoint-metric val_loss --seed 42 --lr 3e-4 --clip-norm 1.0
!python TP2MVTransformer/09_train_mv_alibi_transformer_ablation.py --tag seed123_loss --checkpoint-metric val_loss --seed 123 --lr 3e-4 --clip-norm 1.0
```

## 8. Drift and Calibration Audit

Run after the main transformer or after copying predictions back into the repo.

```python
!python TP2MVTransformer/10_transformer_drift_calibration_audit.py
```

## 9. Summarize All Completed Ablations

```python
!python TP2MVTransformer/11_summarize_transformer_ablation_runs.py
```

## 10. Download Results

```python
!zip -r tp2_transformer_ablation_outputs.zip TP2MVTransformer/ablation_runs
from google.colab import files
files.download("tp2_transformer_ablation_outputs.zip")
```

