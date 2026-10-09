# Phase 2 CSV report

The development_results.csv and test_results.csv headers match the supplied example exactly.
Scores are fractions in CSV and percentages below. Empty cells mean unavailable or undefined; they are not zeros.

| Split | Mode | Domain / fold | Recipe | Trial | Seeds | Explicit F1 (%) | NULL F1 (%) | Combined F1 (%) |
|---|---|---|---|---|---:|---:|---:|---:|
| dev | crossdomain | coursera | mixed_025 | mixed_025_da1a88a51a5f | 5 | 36.30 | 0.00 | 32.75 |
| dev | crossdomain | food | mixed_025 | mixed_025_da1a88a51a5f | 5 | 37.66 | 0.00 | 34.71 |
| dev | crossdomain | hotel | mixed_025 | mixed_025_da1a88a51a5f | 5 | 37.09 | 0.00 | 33.19 |
| dev | crossdomain | laptop | mixed_025 | mixed_025_da1a88a51a5f | 5 | 38.09 | 0.00 | 34.04 |
| dev | crossdomain | phone | mixed_025 | mixed_025_da1a88a51a5f | 5 | 36.97 | 0.00 | 31.53 |
| dev | crossdomain | restaurant | mixed_025 | mixed_025_da1a88a51a5f | 5 | 33.76 | 0.00 | 30.18 |
| dev | crossdomain | sight | mixed_025 | mixed_025_da1a88a51a5f | 5 | 38.86 | 0.00 | 34.39 |
| dev | indomain | all_domains | asl_cls | asl_cls_ccd29d82c495 | 1 | 31.60 | 0.00 | 28.28 |
| dev | indomain | all_domains | bce_cls | bce_cls_72b9117e2b7f | 1 | 32.13 | 0.00 | 28.79 |
| dev | indomain | all_domains | focal_2 | focal_2_7911d73535e0 | 1 | 33.35 | 0.00 | 28.85 |
| dev | indomain | all_domains | focal_2 | focal_2_8e92909d4e3b | 1 | 30.86 | 0.00 | 26.32 |
| dev | indomain | all_domains | focal_cls | focal_cls_cb0f8a9e7ef7 | 1 | 32.18 | 0.00 | 28.83 |
| dev | indomain | all_domains | focal_ia | focal_ia_3a9090049047 | 1 | 32.65 | 0.00 | 29.26 |
| dev | indomain | all_domains | focal_ia_attention | focal_ia_attention_013a87dbfbd4 | 1 | 32.72 | 0.00 | 29.32 |
| dev | indomain | all_domains | focal_ia_attention | focal_ia_attention_e6801d313247 | 5 | 36.81 | 0.00 | 32.89 |
| dev | indomain | all_domains | focal_ia_attention_gamma | focal_ia_attention_gamma_6fd4c60ddd2d | 1 | 32.53 | 0.00 | 29.15 |
| dev | indomain | all_domains | focal_ia_attention_hard | focal_ia_attention_hard_95336217e8b1 | 1 | 32.32 | 0.00 | 28.93 |
| dev | indomain | all_domains | focal_ia_attention_random | focal_ia_attention_random_b3d539181258 | 1 | 32.28 | 0.00 | 28.89 |
| dev | indomain | all_domains | focal_ia_attention_weight | focal_ia_attention_weight_846ad93ada99 | 1 | 32.44 | 0.00 | 29.06 |
| dev | indomain | all_domains | mixed_025 | mixed_025_393b9e11646e | 5 | 36.99 | 0.00 | 33.02 |
| dev | indomain | all_domains | mixed_025 | mixed_025_728834a76d3e | 1 | 26.27 | 0.00 | 23.57 |
| dev | indomain | all_domains | mixed_025 | mixed_025_ab40cdd00db0 | 2 | 33.55 | 0.00 | 30.04 |
| dev | indomain | all_domains | mixed_025 | mixed_025_d578b358d050 | 5 | 36.75 | 9.29 | 30.84 |
| dev | indomain | all_domains | mixed_025 | mixed_025_da1a88a51a5f | 5 | 37.11 | 0.00 | 33.13 |
| dev | indomain | all_domains | mixed_025 | mixed_025_f8a8e7713832 | 1 | 36.24 | 7.54 | 24.02 |
| dev | indomain | all_domains | mixed_033 | mixed_033_b29abfd99fa6 | 2 | 33.07 | 0.00 | 29.77 |
| dev | indomain | all_domains | mixed_050 | mixed_050_28230633aa41 | 2 | 31.32 | 0.00 | 28.40 |
| dev | indomain | all_domains | null_off | null_off_e2306bf501df | 1 | 32.14 | 0.00 | 28.80 |
| dev | indomain | all_domains | null_off | null_off_e53896d9134f | 5 | 36.85 | 0.00 | 32.90 |
| dev | indomain | all_domains | standard | standard_dc1c37f41da3 | 2 | 30.35 | 0.00 | 26.01 |
| dev | indomain | all_domains | weighted | weighted_dbc809433ea0 | 2 | 22.78 | 0.00 | 21.05 |
| test | crossdomain | coursera | mixed_025 | mixed_025_da1a88a51a5f | 5 | 0.00 | 0.00 | 0.00 |
| test | crossdomain | food | mixed_025 | mixed_025_da1a88a51a5f | 5 | 3.80 | 0.00 | 2.50 |
| test | crossdomain | hotel | mixed_025 | mixed_025_da1a88a51a5f | 5 | 18.38 | 0.00 | 15.46 |
| test | crossdomain | laptop | mixed_025 | mixed_025_da1a88a51a5f | 5 | 0.00 | 0.00 | 0.00 |
| test | crossdomain | phone | mixed_025 | mixed_025_da1a88a51a5f | 5 | 0.00 | 0.00 | 0.00 |
| test | crossdomain | restaurant | mixed_025 | mixed_025_da1a88a51a5f | 5 | 17.82 | 0.00 | 15.10 |
| test | crossdomain | sight | mixed_025 | mixed_025_da1a88a51a5f | 5 | 0.00 | 0.00 | 0.00 |
| test | indomain | all_domains | asl_cls | asl_cls_ccd29d82c495 | 1 | 35.68 | 0.00 | 31.26 |
| test | indomain | all_domains | bce_cls | bce_cls_72b9117e2b7f | 1 | 35.81 | 0.00 | 31.42 |
| test | indomain | all_domains | focal_2 | focal_2_8e92909d4e3b | 1 | 32.65 | 0.00 | 27.20 |
| test | indomain | all_domains | focal_cls | focal_cls_cb0f8a9e7ef7 | 1 | 36.14 | 0.00 | 31.70 |
| test | indomain | all_domains | focal_ia | focal_ia_3a9090049047 | 1 | 36.33 | 0.00 | 31.86 |
| test | indomain | all_domains | focal_ia_attention | focal_ia_attention_013a87dbfbd4 | 1 | 36.35 | 0.00 | 31.88 |
| test | indomain | all_domains | focal_ia_attention | focal_ia_attention_e6801d313247 | 5 | 40.52 | 0.00 | 35.43 |
| test | indomain | all_domains | focal_ia_attention_gamma | focal_ia_attention_gamma_6fd4c60ddd2d | 1 | 36.43 | 0.00 | 31.93 |
| test | indomain | all_domains | focal_ia_attention_hard | focal_ia_attention_hard_95336217e8b1 | 1 | 35.84 | 0.00 | 31.40 |
| test | indomain | all_domains | focal_ia_attention_random | focal_ia_attention_random_b3d539181258 | 1 | 35.63 | 0.00 | 31.20 |
| test | indomain | all_domains | focal_ia_attention_weight | focal_ia_attention_weight_846ad93ada99 | 1 | 36.56 | 0.00 | 32.05 |
| test | indomain | all_domains | mixed_025 | mixed_025_393b9e11646e | 5 | 40.74 | 0.00 | 35.60 |
| test | indomain | all_domains | mixed_025 | mixed_025_728834a76d3e | 1 | 29.92 | 0.00 | 26.27 |
| test | indomain | all_domains | mixed_025 | mixed_025_ab40cdd00db0 | 1 | 37.43 | 0.00 | 32.82 |
| test | indomain | all_domains | mixed_025 | mixed_025_d578b358d050 | 5 | 40.59 | 8.89 | 33.44 |
| test | indomain | all_domains | mixed_025 | mixed_025_da1a88a51a5f | 5 | 40.28 | 0.00 | 35.20 |
| test | indomain | all_domains | mixed_025 | mixed_025_f8a8e7713832 | 1 | 40.26 | 7.18 | 26.21 |
| test | indomain | all_domains | mixed_033 | mixed_033_b29abfd99fa6 | 1 | 37.01 | 0.00 | 32.62 |
| test | indomain | all_domains | mixed_050 | mixed_050_28230633aa41 | 1 | 34.74 | 0.00 | 30.87 |
| test | indomain | all_domains | null_off | null_off_e2306bf501df | 1 | 35.82 | 0.00 | 31.42 |
| test | indomain | all_domains | null_off | null_off_e53896d9134f | 5 | 40.55 | 0.00 | 35.43 |
| test | indomain | all_domains | standard | standard_dc1c37f41da3 | 1 | 32.77 | 0.00 | 27.47 |
| test | indomain | all_domains | weighted | weighted_dbc809433ea0 | 1 | 24.58 | 0.00 | 22.31 |

- [Development runs](development_results.csv), [test runs](test_results.csv).
- [Means and sample SD](summary.csv), [run settings and provenance](run_parameters.csv).
- LODO development scores evaluate the six source domains; the domain column identifies the held-out fold. Only the corresponding test scores evaluate the held-out domain.
- run_directory identifies an archive and member prefix relative to the phase2 handoff root; no original machine path is required.
- One row per unique trained model, even when a checkpoint appears in both screening and finalist cohorts.
- run_parameters.csv marks finalists, screens and other saved research runs. Archive membership alone does not imply finalist status.
- No missing test score is filled with a development result; development-only models simply have no test row.
- [Interrupted/failed runs](incomplete_runs.csv) are excluded from completed score exports; their status and settings are retained separately.
