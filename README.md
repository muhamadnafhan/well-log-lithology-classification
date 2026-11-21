# Lithology Classification from Well Logs (FORCE 2020, North Sea)

Predicting rock type at every depth from wireline well logs, using gradient boosting and validating **by well** (not by random rows) on 118 wells from the Norwegian Sea.

> Personal learning project. Results are my own evaluation and are **not comparable** to the official FORCE 2020 competition leaderboard (different metric and split).

## Interactive demos

- **[Results dashboard](https://muhamadnafhan.github.io/well-log-lithology-classification/dashboard.html)**: class imbalance, experiment log, score stability, data quality. Source: [`docs/dashboard.html`](docs/dashboard.html)
- **[Well viewer](https://muhamadnafhan.github.io/well-log-lithology-classification/well_viewer.html)**: log curves next to true vs predicted lithology, with mismatches highlighted, for 6 randomly chosen held-out wells. Source: [`docs/well_viewer.html`](docs/well_viewer.html)

Both are single static HTML files (no server, no external libraries).

## Why this problem

Core samples are expensive and rare; well logs exist in almost every well. Interpreting rock type from log curves is traditionally manual, slow and somewhat subjective. This project asks: how far can a simple, well-validated ML pipeline get on real, messy, imbalanced log data?

## Data

- **FORCE 2020 well log and lithofacies dataset**, 118 wells, Norwegian Sea. Lithofacies labels were interpreted by experienced geoscientists.
- Source: Bormann, Aursand, Dilib, Manral, Dischington (2020), Zenodo, DOI [10.5281/zenodo.4351156](https://doi.org/10.5281/zenodo.4351156). License CC BY 4.0. Original well logs come from the Norwegian government (NOLD 2.0).
- 2,337,639 rows in total, 1,431,383 of them labelled (all 118 wells have labels).
- The raw data is **not** included in this repo. See the notebook for the download code.

### The challenge: heavily imbalanced classes

| Class | Share of labelled rows |
|---|---|
| Shale | 61.3% |
| Sandstone | 14.5% |
| Sandstone/Shale | 12.6% |
| Limestone | 4.9% |
| Marl | 2.9% |
| Tuff, Halite, Chalk | ~1% each |
| Coal, Dolomite, Anhydrite, Basement | < 0.35% each (Basement: 141 rows) |

Predicting "Shale" everywhere already gives ~61% accuracy, so **accuracy is misleading here**. The main metric is **macro-F1**, which weights every class equally. The "always Shale" baseline scores a macro-F1 of 0.063.

## Method

**Features**
- 9 log curves: GR, RHOB, NPHI, DTC, PEF, RDEP, RMED, SP, CALI (resistivities log10-transformed).
- Context features per well: rolling mean and std (windows of 5 and 25 samples, roughly 0.75 m and 4 m) and a 5-sample difference, for GR, RHOB, NPHI and DTC.
- Coordinates and depth were deliberately **not** used, so the model cannot memorise well locations.
- Missing values are left as NaN. Missingness is per well (some wells lack whole curves), so mean imputation would be misleading. The model handles NaN natively.

**Model**: scikit-learn `HistGradientBoostingClassifier`, trained on every 6th row (neighbouring samples are nearly identical), with sample weights proportional to (1 / class frequency)^0.75.

**Validation**: grouped by well (`GroupKFold`, 5 folds across 118 wells), plus repeated hold-outs of 20% of wells never seen in training.

## Results

### Experiment log (5-fold grouped CV)

| # | Setup | Macro-F1 | Accuracy |
|---|---|---|---|
| 0 | Always predict Shale | 0.063 | ~0.61 |
| 1 | 9 raw curves, `class_weight=balanced` | 0.337 | 0.57 |
| 2 | + rolling features (windows 25, 100), balanced | 0.369 | 0.62 |
| 3 | Windows 25, 100, softer weights (power 0.5) | 0.387 | 0.70 |
| 4 | Windows 5, 25, power 0.5 | 0.399 | 0.69 |
| **5** | **Windows 5, 25, power 0.75 (final)** | **0.415** | 0.66 |
| 6 | Windows 5, 25, power 1.0 | 0.407 | 0.61 |
| 7 | Final + drop low-confidence labels from training | 0.403 | 0.66 |

Note: run 1 used slightly different training settings (subsampling and iterations) than runs 2-7, so comparisons are approximate. Each row is a single run with no variance estimate, and configurations were selected on the same CV, so the best CV score is somewhat optimistic.

### Held-out wells (more honest estimate)

Five random hold-outs, each leaving out 20% of wells:

**Macro-F1 = 0.390 ± 0.032** (range 0.335 to 0.434). The number of classes present in the test wells varied from 9 to 12, which adds noise to macro-F1. An earlier single hold-out (different seed, heavier training settings) scored 0.304, lower than all five runs above.

Per-class F1 on that single hold-out (23 wells):

| Class | F1 |
|---|---|
| Shale | 0.77 |
| Sandstone | 0.59 |
| Limestone | 0.52 |
| Chalk | 0.35 |
| Halite | 0.32 |
| Sandstone/Shale | 0.26 |
| Coal | 0.18 |
| Marl | 0.17 |
| Tuff | 0.15 |
| Anhydrite | 0.03 |
| Dolomite | 0.01 |

## Findings

1. **Accuracy hides the real picture.** Class weighting lowered accuracy but improved rare-class recognition. Macro-F1 and per-class F1 are the right lens.
2. **Context helps, but window size matters.** Rolling features improved most classes, yet wide windows (about 15 m) blurred thin beds: Halite F1 was 0.53 with windows (25, 100) versus 0.79 with (5, 25) in the same setup.
3. **Class weighting is a trade-off.** Stronger weights raised macro-F1 up to power 0.75, then hurt accuracy and gave no further gain.
4. **Filtering by label confidence backfired.** About 98% of Tuff labels carry the lowest confidence level (3), so filtering them out removed nearly all Tuff examples and its F1 collapsed to about 0.
5. **Large classes are stable, rare classes are not.** Shale, Sandstone and Limestone score consistently across splits; Anhydrite and Halite swing widely depending on which wells land in the test set.

6. **Performance varies a lot from well to well.** On six randomly chosen held-out wells (shown in the well viewer), per-sample accuracy ranged from 53% to 84%. Two of the six had sparse RHOB and NPHI coverage, but they landed at opposite ends (53% and 84%), so curve completeness alone does not explain the gap. Six wells is too few to say more. These accuracies are computed on data downsampled to every 10th sample and are not comparable to the macro-F1 figures above.

## Limitations

- Dolomite and Basement are essentially not learned (very few examples).
- "Sandstone/Shale" is a mixed class and is inherently ambiguous.
- Results depend on which wells are held out; treat single numbers with caution.
- The lithology code-to-name mapping follows the competition documentation; please verify against the official repository.
- Lithology is not hydrocarbon detection. It identifies rock type, which helps locate potential reservoir (sandstone, carbonate) and seal (shale) intervals.

## Possible next steps

- Compare LightGBM or XGBoost and tune hyperparameters with nested validation.
- Sequence models (1D CNN or LSTM along depth).
- Merge or hierarchically model rare classes.
- Predict porosity or missing curves as a follow-up project.
- Build a Streamlit dashboard to browse logs and predictions per well.

## Reproduce

Open the notebook in Google Colab and run all cells. It downloads the data from Zenodo, parses the LAS files with `lasio`, builds features and runs the validation.

## Acknowledgements

Data: FORCE 2020 / Norwegian Offshore Directorate (NOLD 2.0) via Zenodo (CC BY 4.0).
