# Data Directory

This directory is reserved for locally generated train/test files.

It is intentionally ignored by Git to avoid uploading experimental data. After preprocessing, expected generated files include:

```text
interp_train_x_SSP_TLshape_ndrz10.mat
interp_test_x_SSP_TLshape_ndrz10.mat
train_y_SSP_TLshape_ndrz10.mat
test_y_SSP_TLshape_ndrz10.mat
split_seed2025.npz
preprocess_summary.json
```

Keep raw data in `raw_data/` or set `TAM_FNO_RAW_DATA=/path/to/raw_data`.
