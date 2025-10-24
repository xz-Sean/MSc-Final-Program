# Copula-VAE Hybrid Synthesis for Small-Sample Geoscience Data

## Due to potential restrictions, real and synthetic data are not displayed in this repo.

> Synthetic tabular data generation for blasting–fragmentation under **small-n, moderate-d, multi-site heterogeneity**.
> Pipeline: **Copula (PIT→AIC)** ⟶ **TVAE variants (Vanilla / β-annealed / GMM prior)** ⟶ **Evaluation (TSTR / MMD / FAED / multi-seed)**.

---

## Table of Contents

* [Overview](#overview)
* [Key Findings](#key-findings)
* [Environment & Installation](#environment--installation)
* [Repository Structure](#repository-structure)
* [Data & Preprocessing](#data--preprocessing)
* [End-to-End Reproducibility](#end-to-end-reproducibility)
* [Copula Fitting](#copula-fitting)
* [TVAE Variants](#tvae-variants)
* [Evaluation (Utility / Fidelity / Robustness)](#evaluation-utility--fidelity--robustness)
* [Results Summary](#results-summary)
* [Limitations](#limitations)
* [Future Work](#future-work)
* [Cite This Work](#cite-this-work)
* [Acknowledgements & License](#acknowledgements--license)

---

## Overview

This project studies **tabular synthetic data** generation for blasting–fragmentation when data are scarce and heterogeneous across sites. We unify 8 public sources into a **12-feature schema**, and use 3 fully matching sources to create an **FI pool** with **n = 262** rows:

* `hudaverdi_2010_full` (110), `hudaverdi_2012` (62), `kulatilake_2010` (90).

We compare **Copulas** (Gaussian, Student-t, Clayton) with **TVAE variants**:

* **Vanilla TVAE** (standard normal prior)
* **Annealed TVAE** (β-annealing schedule)
* **TVAE-GMM** (Gaussian Mixture prior)

Evaluation uses:

* **Utility**: **TSTR / TRTS** with fixed-hyperparam **XGBoost** on **Fragmentation Index** (FI)
* **Fidelity**: **MMD** (RBF multi-kernel), **FAED** (Fréchet Autoencoder Distance)
* **Robustness**: **multi-seed** sampling + downstream retraining (mean ± std)

All **preprocessing and model fitting use training split only**; held-out test is never used for tuning.

---

## Key Findings

* **Copula (train-only AIC/obs)**: **Clayton** best; **Gaussian** fragile at small-n (high parameter count, tail-independent).
* **TVAE**:

  * **Vanilla** weak under small-n; **copula-only** or **mixed** training **degrades** utility.
  * **β-annealing** gives a large gain over vanilla.
  * **TVAE-GMM (real-only)** is **best overall** (see numbers below).
* **Purpose-first**: Choose generator/metrics by goal (development/stress test vs distributional mimicry vs coverage of rare regimes).

---

## Environment & Installation

**Conda**

```bash
# from repo root
conda env create -f notebook/environment.yml -n irp-synth
conda activate irp-synth
cd notebook
jupyter lab   # or: jupyter notebook
```

> GPU users: install the PyTorch build matching your CUDA if needed; CPU-only works out of the box.

**Pip**

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip wheel
pip install pandas numpy scikit-learn xgboost scipy matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Repository Structure

```
.
├─ README.md
├─ deliverables/
│  ├─ README.md
│  ├─ xz2821-project-plan.pdf
│  └─ xz2821-final-report.pdf
├─ logbook/
│  ├─ README.md
│  └─ logbook.md
├─ title/                         # (title-page assets/templates, if any)
└─ notebook/
   ├─ environment.yml            # reproducible conda environment
   ├─ Data_Processing.ipynb      # data cleaning & unification; creates train/test
   ├─ Copula.ipynb               # PIT → copula fitting → AIC/obs → sampling
   ├─ VAE.ipynb                  # TVAE (Vanilla/Anneal/GMM) training & evaluation
   ├─ Constraint.ipynb           # (optional) physical/rule-based constraint checks
   ├─ figure/                    # generated figures
   └─ data/
      ├─ mult_var_blast.xlsx     # original consolidated workbook
      ├─ clean_data/             # per-source cleaned tables
      ├─ data_stat_report/       # descriptive statistics / missingness reports
      ├─ merged_data/
      │  ├─ fi_pool.xlsx         # 3-source merge (n=262)
      │  ├─ train_set.csv        # fixed 80/20 training split
      │  └─ test_set.csv         # fixed 80/20 test split
      ├─ Copula_data/
      │  ├─ synthetic_clayton_split.csv   # Clayton-copula synthetic table (e.g., 5k rows)
      │  └─ copula_comparison.csv         # AIC/obs comparison and logs (if saved)
      ├─ synthetic_data/         # TVAE samples (by variant/recipe/seed)
      ├─ TVAE_hyperparameters/   # Optuna or grid-search results
      └─ Clayton_qc_reports/     # copula quality-check plots/reports

```

> **Note**: Paths in notebooks assume the tree above. Adjust if you relocate files.

---

## Data & Preprocessing

* **Golden-12 schema** (unified names/units):

  * `S/B`, `H/B`, `B/D`, `T/B`, `powder_factor (kg·m^-3)`, `youngs_modulus_gpa`,
    `fragment_median_m (X50)`, `fragmentation_index (FI)`, plus identifiers (dropped in modeling).
* **Core modeling set**: **6 fully covered variables**: `FI` (target), `X50`, `PF`, `H/B`, `B/D`, `T/B` (0% missingness across sources).
* **Split**: fixed **80/20** train/test saved as CSV; **standardisation** (z-score) fitted **on train only** for TVAE; XGBoost uses **physical scales**.

**Run preprocessing**
Use `notebooks/Data_Processing.ipynb` to reproduce cleaning and verify the split files exist under `data/merged_data/`.

---

## End-to-End Reproducibility

1. **Preprocess** ⟶ ensure `train_set.csv` & `test_set.csv` exist.
2. **Copula.ipynb** ⟶ fit Gaussian / t / Clayton (PIT→AIC/obs), sample **Clayton** to `Copula_data/synthetic_clayton_split.csv`.
3. **VAE.ipynb** ⟶ train TVAE variants on **Real-only / Copula-only / Mixed(α)**; sample; evaluate.
4. **Evaluation cells** compute TSTR / TRTS / MMD / FAED; loop **seeds**; aggregate **mean ± std**.

> All scalers/marginals fitted **only on train**. Test remains untouched until final scoring.

---

## Copula Fitting

* **PIT (train-only)**: `QuantileTransformer(output_distribution='uniform')`
* **ε-clipping**: `eps = 1e-3`
* **Kendall’s τ → Gaussian ρ**: `rho = sin(π/2 * τ)`
* **Shrinkage**: `ρ ← (1−λ)ρ + λI`, with `λ = 0.05`; **PD repair** (eigenvalue floor) if needed
* **Model selection (train)**: **AIC per observation**; **Clayton** selected
* **Sampling**: draw `u` from copula ⟶ clip ⟶ inverse PIT back to physical units (CSV saved)

> **Note:** We report **train AIC/obs**; **test LL/obs** is left for future work to keep test held-out.

---

## TVAE Variants

**Three variants (standardised z-space; inverse-transform after sampling):**

* **Vanilla TVAE**: prior $p(z)=\mathcal N(0,I)$
* **Annealed TVAE**: same prior + **β-annealing** (linear / sigmoid) to mitigate posterior collapse
* **TVAE-GMM**: prior $p(z)=\sum_{k=1}^K \pi_k\,\mathcal N(\mu_k,\Sigma_k)$ (diag), learned jointly (MC KL with log-sum-exp)

**Training sources (recipes):**

* **Real-only**: train on `train_set.csv`
* **Copula-only**: train on `synthetic_clayton_split.csv`
* **Mixed(α)**: real\:copula = `1:1, 1:2, 1:4, 1:8` (shuffled concat)

**Tuning (no leakage)**: Optuna on **train-internal** 25% validation; then **retrain on full train**.

---

## Evaluation (Utility / Fidelity / Robustness)

**Utility (task)**

* **TSTR**: train XGBoost on **synthetic**, test on **real test**
* **TRTS**: train on **real train**, test on **synthetic** (optional)
* **XGBoost (fixed)**: `n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=seed`

**Fidelity (distribution)**

* **MMD**: RBF multi-kernel; lower is better
* **FAED**: AE latent Fréchet distance (AE trained on **real-train** only); lower is better

**Robustness (multi-seed)**

* Repeat **sampling + downstream fit** for S seeds; report **mean ± std**

---

## Results Summary

> Means across seeds; held-out real test for TSTR. (Full tables in the report)

**Copula (train)**: **AIC/obs** — **Clayton 2.56** (k=1), Student-t(ν=8) 6.44, Gaussian 8.19.
**Vanilla TVAE** (Real-only, TSTR): **R² ≈ 0.434**, **RMSE ≈ 2.489**.
**Copula-only / Mixed(α)**: utility **degrades** as copula share increases (R² down, RMSE up).
**Annealed TVAE** (Real-only): **R² ≈ 0.646**, **RMSE ≈ 1.967**, **MMD ≈ 0.164**.
**TVAE-GMM** (Real-only): **R² ≈ 0.736**, **RMSE ≈ 1.698**, **MMD ≈ 0.160**, **FAED ≈ 0.436**.

**Takeaway**

* **TVAE-GMM (real-only)** is the **default** for small-n, heterogeneous sites.
* **Annealed TVAE** is a good lighter-weight alternative.
* Avoid **copula-only** or **copula-heavy mixes** for training VAEs when the goal is downstream utility.

---

## Limitations

* **Small-n (n=262)**, numeric-only core features; some relevant properties are sparse/missing.
* **Single fixed split** with multi-seed repeats (no LOSO in this version).
* **No explicit memorisation audit (DCR / NNR)**; privacy risk not quantified.
* **Copula test LL/obs** not computed to keep test fully held-out for downstream evaluation.

---

## Future Work

* **Evaluation**: add **test LL/obs** for copulas; **95% CIs** & **paired permutation tests**; **LOSO** across sites; **DCR/NNR** memorisation checks (and consider DP-SGD).
* **Models**: **CTGAN / CTAB-GAN(+)**; **tabular diffusion** (PIT-Gaussian or direct numeric); **normalising flows**; **autoregressive Transformers**; **physics constraints** (plausible ranges / monotonicity).
* **Purpose-aligned synthesis**: conditional / TSTR-aware sampling; **tail-focused** augmentation.
* **Learning beyond small set**: transfer learning / pretraining on broader mining or materials datasets.

---

## Cite This Work

If you use this code or results, please cite the final report:

```
X. Zhao, “Copula-VAE Hybrid Synthesis for Small-Sample Geoscience Data,”
MSc IRP Final Report, Imperial College London, 2025.
```

---

## Acknowledgements & License

* Supervisors: **Dr. Paulo Lopes**, **Dr. Pablo Brito Parada**
* Thanks to the authors of the original datasets and open-source libraries used here.
* **License**: unless otherwise stated, this repository is released under the MIT License.
* **AI use disclosure**: LLM tools were used for *conceptual support and editing only*; all experiments, code, and interpretations are the author’s own.



> Questions or issues? Please open an issue or contact **[xz2821@ic.ac.uk](mailto:xz2821@ic.ac.uk)**.
