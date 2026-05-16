# Contributing

Contributions should keep the repository reproducible and data-light.

Before opening a pull request:

```bash
PYTHONPATH=src pytest -q
```

Do not commit raw data, generated train/test splits, checkpoints, logs, normalizers, or large generated result folders.
