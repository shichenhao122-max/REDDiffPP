# Final Hyperparameters

The table below mirrors the final configs included under `configs/algorithm/`.

| Setting | Config | Method target | num_steps | observation_weight | base_lr | base_lambda | sigma_data | batch_size | Extra |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| FFHQ | `reddiffpp-ffhq` | `algo.reddiffpp.REDDiffPP` | 2000 | 1.0 | 0.02 | 0.25 | 0.5 | 4 | - |
| MRI raw x4 | `reddiffpp-mri-raw-x4` | `algo.reddiffpp.REDDiffPP` | 4000 | 5.457094 | 0.0033616 | 0.0406038 | 0.386336 | 4 | - |
| MRI raw x8 | `reddiffpp-mri-raw-x8` | `algo.reddiffpp.REDDiffPP` | 4000 | 3.313066 | 0.0058519 | 0.0090139 | 0.112187 | 4 | - |
| MRI raw x4 RSS init | `reddiffpp-mri-raw-x4-rssinit` | `algo.reddiffpp_rssinit.REDDiffPPRSSInit` | 4000 | 5.457094 | 0.0033616 | 0.0406038 | 0.386336 | 4 | `init_mode=rss`, `phase_mode=mvue` |
| MRI raw x8 RSS init | `reddiffpp-mri-raw-x8-rssinit` | `algo.reddiffpp_rssinit.REDDiffPPRSSInit` | 4000 | 3.313066 | 0.0058519 | 0.0090139 | 0.112187 | 4 | `init_mode=rss`, `phase_mode=mvue` |
| MRI rawsafe x4 | `reddiffpp-mri-rawsafe-x4` | `algo.reddiffpp_rawsafe.REDDiffPPRawSafe` | 4000 | 5.457094 | 0.0033616 | 0.0406038 | 0.386336 | 4 | `lambda_cap=0.1` |
| MRI rawsafe x8 | `reddiffpp-mri-rawsafe-x8` | `algo.reddiffpp_rawsafe.REDDiffPPRawSafe` | 2000 | 7.924398 | 0.0051613 | 0.0183389 | 0.0821706 | 4 | `lambda_cap=0.2` |
| MRI rawsafe x4 rerank-best | `reddiffpp-mri-rawsafe-x4-rerankbest-cap02` | `algo.reddiffpp_rawsafe.REDDiffPPRawSafe` | 5000 | 2.967594 | 0.0023775 | 0.0383599 | 0.390451 | 4 | `lambda_cap=0.2` |
| Black hole | `reddiffpp-blackhole` | `algo.reddiffpp.REDDiffPP` | 1000 | 0.0004 | 0.05 | 0.25 | 0.5 | 4 | - |
| Inverse scattering | `reddiffpp-inv-scatter-360-2000` | `algo.reddiffpp.REDDiffPP` | 2000 | 8584.180296 | 0.0846995 | 0.0062818 | 1.627403 | 5 | - |
