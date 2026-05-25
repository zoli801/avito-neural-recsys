# Kaggle Run Notes

The local notebook at `/Users/dmitrii/Downloads/Новая папка/main.ipynb` points to this attached Kaggle dataset:

```text
/kaggle/input/datasets/nikitakuznetsof/avito-ml-cup-2026
```

Inside a Kaggle notebook with that dataset and GPU enabled:

```bash
git clone https://github.com/zoli801/avito-neural-recsys.git
cd avito-neural-recsys
python scripts/eda.py
EPOCHS=6 BATCH_SIZE=1024 MAX_TRAIN_POSITIVES=2000000 python train.py
CANDIDATE_POOL_SIZE=4096 python predict.py --out /kaggle/working/submission.csv
```

This configuration is intended to finish in roughly 40 minutes on a Kaggle GPU session. Increase `EPOCHS`, `MAX_TRAIN_POSITIVES`, or `CANDIDATE_POOL_SIZE` if the session has more time. The default path resolver automatically uses the Kaggle dataset path above, so `DATA_DIR` is optional there.
