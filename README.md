# Group K-Fold Gradient Boosting Alpha

LightGBM/CatBoost ensemble (Huber loss) for short-horizon return prediction on 661k rows, and a validation study showing that a shuffled-KFold R² of 0.055 collapses to about 0.002 once folds respect the dataset's 70 groups.

## Context

- Event: Kaggle code competition "Short-Horizon Return Prediction Challenge by iRage" (iRage AlgoArena).
- Dates: 25 Mar 2026 to 12 Apr 2026.
- Solo work.
- Constraints from the competition: 30-minute CPU runtime, no internet, row-wise feature engineering only (no cross-sample operations on test).

## Problem

Predict a short-horizon return target from 445 anonymised features.

- Train: 661,574 rows. Test: 410,139 rows.
- A `CV_GROUP` column partitions the training data into 70 groups.
- The target is heavy-tailed (kurtosis 48.09).
- Offline metric: R².

## Approach

1. **Validation first.** The first pipeline (v4) used `KFold(5, shuffle=True)`. Rows from the same group then sit in both train and validation folds, so the score measures within-group fit rather than generalisation to unseen groups. From v5 onward all validation uses `GroupKFold(5)` on `CV_GROUP` (70 groups into 5 folds).
2. **Fat tails.** The target is clipped at P2/P98 (kurtosis falls from 48.09 to 1.52 after clipping, v6 output), and the tree models use Huber objectives.
3. **Features.** About 471 row-wise features (momentum accelerations and divergences, absolute momentum, price features, feature-family aggregates, polynomial terms), taking the frame to 916-926 columns. Z-scoring uses training statistics only.
4. **Feature selection.** Top 300 by LightGBM gain (v8), Pearson correlation (v10), top 200 by LightGBM gain (v11).
5. **Models.** LightGBM Huber (`num_leaves=15`), a shallow LightGBM (`num_leaves=7`) and CatBoost (`Huber:delta=0.05`). Ensemble weights are fitted with Nelder-Mead on out-of-fold (OOF) predictions.
6. **Runtime budget.** The Kaggle notebooks have hard time guards per model so the full pipeline finishes inside the 30-minute limit.

## Results

OOF R² as printed in the saved notebook outputs:

| Version | Folder | Validation | OOF R² (optimised weights) |
|---|---|---|---|
| v4 | `algo1` | shuffled KFold(5) | +0.055276 |
| v5 | `algo3` | GroupKFold(5), 70 groups | +0.000536 |
| v6 | `algo4` | GroupKFold(5) | +0.001476 |
| v6 (variant) | `algo5` | GroupKFold(5) | +0.001266 |
| v7 | `algo6` | GroupKFold(5) | +0.001776 |
| v8 | `algo7` | GroupKFold(5) | +0.001956 |
| v9 | `algo8` | GroupKFold(5) | +0.001830 |
| v10 | `algo9` | GroupKFold(5) | +0.001241 |
| v11 | `algo10` | GroupKFold(5) | +0.001999 |
| v12 | `algo11` | GroupKFold(5) | +0.001873 |

Main finding: the 0.055 shuffled-KFold estimate fell to about 0.002 under group-aware validation. The shuffled estimate was leaking group-level information. v4 also differs from later versions in features and model settings, so the drop is best read as the honest CV *exposing* an optimistic estimate rather than a controlled one-variable ablation.

An OOF R² of about 0.002 is small in absolute terms but is the realistic order of magnitude for short-horizon return prediction across unseen groups.

Public and private leaderboard scores were not recorded. The final Kaggle notebooks (`algo12_final`, `algo13`-`algo15`) are saved without outputs.

## Repository layout

```
algo1/          v4: shuffled-KFold baseline (LightGBM/XGBoost), code_try_quant.py, feature importances
algo2/          v5 draft (no outputs)
algo3/          v5: first GroupKFold run
algo4/, algo5/  v6: target clipping + Huber
algo6/          v7: expanded row-wise feature engineering
algo7/          v8: best GroupKFold OOF among the early versions (0.001956)
algo8/ - algo11/  v9-v12: CatBoost tuning, Pearson vs gain selection, best OOF 0.001999 (v11)
algo12_final/   v8 Kaggle submission notebook (30-min budget)
algo13/ - algo15/ v9-v11 Kaggle submission notebooks with time guards
algo*/output*/feature_importance.csv   per-version gain importances
```

## How to run

1. Accept the competition rules on Kaggle and download `train.parquet` and `test.parquet` for "Short-Horizon Return Prediction Challenge by iRage". The data is not included here.
2. Install the dependencies: `pip install pandas numpy pyarrow lightgbm catboost xgboost scikit-learn scipy jupyter`.
3. For the Kaggle notebooks (`algo12_final`, `algo13`-`algo15`), attach the competition dataset; they read from `/kaggle/input/competitions/short-horizon-return-prediction-challenge-by-i-rage/` and write `submission.csv`.
4. For the other notebooks (`algo1`-`algo11`), point `DATA_DIR` in the config cell at the downloaded files and run all cells. Full runs take tens of minutes on CPU.

## Notes and limitations

- Ensemble weights are optimised on the same OOF predictions used to report the final R², so the reported figures are mildly optimistic.
- Feature selection is done once, on a single LightGBM fold, before the CV loop rather than nested inside it. This leaks a little selection information into the CV scores.
- Which notebook was finally submitted is not recorded: `algo12_final` (v8) was labelled final, but `algo15` (v11) is newer.
- Competition data, submission files and the local virtual environment are excluded.
