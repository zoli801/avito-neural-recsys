# Kaggle Run Notes

The local notebook at `/Users/dmitrii/Downloads/Новая папка/main.ipynb` points to this attached Kaggle dataset:

```text
/kaggle/input/datasets/nikitakuznetsof/avito-ml-cup-2026
```

The standalone notebook used for the neural-only run is:

```text
notebooks/avito_neural_solution_submit.ipynb
```

Inside a Kaggle notebook with that dataset and GPU enabled, the script-based run is:

```bash
git clone https://github.com/zoli801/avito-neural-recsys.git
cd avito-neural-recsys
python scripts/eda.py
EPOCHS=6 BATCH_SIZE=1024 MAX_TRAIN_POSITIVES=2000000 python train.py
CANDIDATE_POOL_SIZE=4096 python predict.py --out submission.csv
```

Or one command:

```bash
bash scripts/run_kaggle_40m.sh
```

The default path resolver automatically uses the Kaggle dataset path above, so `DATA_DIR` is optional there.
