# prostate-mri-fp-mechanism

**A multi-architecture study of specificity refinement and false-positive mechanism analysis in prostate MRI**

Code, result data, and figures for a study of why false positives persist in
deep-learning prostate-MRI cancer detection, and whether a lightweight post-hoc
head can improve case-level specificity without sacrificing sensitivity.

Submitted to *European Radiology*.

## One-line summary

Residual false positives from prostate-MRI detectors are contrast-matched to
true cancer (they share raw T2-weighted and ADC imaging contrast with cancer
rather than resembling benign tissue) -- a data-level imaging property that
reproduces across five architectures; a parameter-light post-hoc head adds
in-domain specificity but is fold-conditional.

## Key findings

- False positives are **contrast-matched to cancer**: their raw imaging contrast
  sits with true cancer and away from benign tissue, not the reverse.
- The lesion-vs-benign and false-positive-vs-benign evidence directions
  reproduce in **35/35 observations across five distinct architectures**
  (bare nnU-Net, bare U-Net, bare Mamba, MIGF-Mamba, MIGF-nnUNet), with
  **Cohen's d 1.10** (lesion vs benign) and an **FP/benign evidence ratio of 2.38x**.
- The direction also reproduces **105/105** across seven modality-perturbation
  scenarios and replicates on an external cohort (T2-weighted and ADC channels).
- A frozen-backbone post-hoc refinement head (89,216 parameters) raises
  PI-CAI fold-0 case-level specificity by **+17.2% relative (+0.080 absolute)**
  at preserved sensitivity; 5-fold cross-validation shows the magnitude is
  **fold-conditional** (9/15 observations positive).
- On the external cohort both models saturate near full sensitivity under a
  documented ADC domain shift, so the in-domain specificity advantage is
  inconclusive there; the contrast-matching mechanism still replicates.

## Datasets

This study uses two public cohorts. No patient imaging is redistributed here;
only Paper 2's own aggregate result files are included.

- **PI-CAI** (primary): https://pi-cai.grand-challenge.org/
  Public Training and Development Dataset, official 5-fold split; method
  development and primary evaluation on fold-0 (1200 train / 300 validation;
  fold-0 validation = 84 csPCa / 216 negative). Full set = 425 csPCa / 1075 negative.
- **Prostate158** (external): https://github.com/kbressem/prostate158
  158 studies (102 csPCa / 56 negative).

## Repository structure

```
.
  README.md                 This file
  LICENSE                   MIT license
  CITATION.cff              Citation metadata
  requirements.txt          Python dependencies
  paper.pdf                 Compiled manuscript (self-contained)
  figures/                  Figure 1-6 (vector PDF) + the .py generators
  experiments/              Aggregate result JSONs + analysis / evaluation scripts
  graphical_abstract/       Editable graphical abstract (.pptx) + rendered .png
```

`experiments/` contains, per analysis, the per-seed and aggregate result JSONs
that the figures and the manuscript consume:

- `baselines/`            post-hoc recalibration baselines vs the refinement head
- `fp_stratification/`    false-positive suppression by cancer-similarity tertile
- `prostate158_p2b/`      external false-positive contrast-ratio analysis
- `prostate158_matched_sens/`  matched-sensitivity specificity on the external cohort
- `casewise_bootstrap/`   per-case paired bootstrap confidence intervals
- `prostate158_casewise/` paired McNemar analysis on the external cohort
- `prostate158_sensspec_curve/`  sensitivity-specificity sweep on the external cohort
- `p2a_5fold_cv/`         fold-0 bootstrap for the post-hoc head
- `prostate158_p2a/`, `prostate158_p2a_bare/`  refined vs bare per-seed results

## Scope and reproducibility

This repository is a **release-artifact and reproducibility-scaffold** package: it
provides the summarized experiment records (aggregate and per-seed result JSONs),
the figure/analysis scripts, the graphical abstract, and the paper PDF. It is
**not** a one-click, fully self-contained reproduction of every figure. The frozen
detection backbone, its trained checkpoints, and the preprocessed image caches are
external and are **not redistributed here**; scripts that need them mark those
locations with `ADJUST_PATH/...` placeholders.

In particular:

- **Figures 1, 2, and 6** regenerate directly from files in this repo (Figure 1 is
  self-contained; Figure 2 reads `figures/prostate158_casewise_probs.json`;
  Figure 6 reads the `baselines/` and `fp_stratification/` aggregates).
- **Figures 3-5 are NOT fully reproducible from this repo alone**: they reference
  raw per-seed evidence JSONs produced in the analysis workspace that are not
  redistributed (their input paths are marked `ADJUST_PATH/...`). The repository
  ships the summarized records behind these figures, not the raw per-seed inputs.

Scripts that re-run the frozen detection backbone (`experiments/*/eval_*.py`,
`compute_*.py`) require the backbone code and the preprocessed image caches,
which are external; their local paths are likewise marked `ADJUST_PATH/...`.
Regenerating the figures from the provided result JSONs does **not** require the
backbone or `torch`.

```
pip install -r requirements.txt
python figures/figure6_baselines_stratification.py
```

## Citation

If you use this repository, please cite Paper 2 (arXiv:2606.29977; see
`CITATION.cff`). Repository:
https://github.com/yosh3289/prostate-mri-fp-mechanism

The frozen detection backbone is described in our earlier preprint:
arXiv:2604.10702 (the detection-backbone / Paper 1 work).

## License

MIT (see `LICENSE`). Submitted to *European Radiology*.
