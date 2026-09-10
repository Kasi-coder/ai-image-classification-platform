# Experiment Notes

The executable experiment is implemented in `scripts/train.py`.

Recommended experiments:
1. CPU baseline: 10k train / 2k validation / 2k test, 3 epochs.
2. Stronger run: full dataset, 5–10 epochs.
3. Compare ResNet18 vs MobileNetV3-Small.
4. Compare frozen-backbone and partial fine-tuning.
5. Tune learning rate (`1e-3`, `3e-4`) and batch size (`32`, `64`).

Record the final results in `reports/model_comparison.json`.
