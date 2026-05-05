# REDDiffPP

Private review package for the REDDiffPP experiments.

This repository contains the core implementation, final algorithm configs, selected result figures, and lightweight reproduction notes for collaborator and advisor review. The work is currently unpublished. Please do not redistribute the contents without permission.

## Contents

- `src/algo/reddiffpp.py`: core REDDiffPP implementation.
- `src/algo/reddiffpp_rssinit.py`: optional MRI RSS/MVUE initialization variant.
- `src/algo/reddiffpp_rawsafe.py`: optional raw-MRI safety variant with capped prior weights.
- `configs/algorithm/`: final algorithm configs used for selected experiments.
- `figures/`: selected qualitative and quantitative figures.
- `docs/RedDiff_summary.pdf`: internal result summary PDF.
- `docs/reproducibility.md`: instructions for running this package as an overlay on InverseBench.
- `docs/hyperparameters.md`: final hyperparameter table.

## Relationship To InverseBench

This is a lightweight overlay for InverseBench, not a standalone benchmark release. To run experiments, copy the files under `src/algo/` and `configs/algorithm/` into an existing InverseBench checkout with the required datasets and pretrained checkpoints.

## Main Changes

- Adds the REDDiffPP noise-dependent prior-weight schedule.
- Uses a batch-averaged stochastic prior gradient for a lower-variance update.
- Provides tuned final hyperparameters across FFHQ, MRI, black-hole imaging, and inverse scattering settings.
- Includes optional MRI-specific initialization and raw-MRI safety variants for collaborator review.

## Selected Results

The selected figures under `figures/` summarize representative qualitative and quantitative behavior:

- `figures/ffhq256_paper_qualitative_reddiff_vs_reddiffpp/`
- `figures/inv_scatter_reddiff_compare_summary_metrics.png`
- `figures/mri_knee_reddiffpp_bad_compare/rssinit_stage3rerankbest_raw_x8/`
- `figures/blackhole_best_test/`

See `docs/RedDiff_summary.pdf` for the full internal summary.

## Status

Private research code for collaborator review. A cleaned public release can be prepared after publication.
