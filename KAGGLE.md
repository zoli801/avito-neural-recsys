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
EPOCHS=8 BATCH_SIZE=1024 MAX_TRAIN_POSITIVES=3000000 python train.py
python predict.py --out /kaggle/working/submission.csv
```

For a one-hour training run, increase `EPOCHS` or `MAX_TRAIN_POSITIVES` depending on the GPU session speed. The default path resolver automatically uses the Kaggle dataset path above, so `DATA_DIR` is optional there.
