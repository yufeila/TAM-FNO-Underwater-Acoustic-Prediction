# TAM-FNO Underwater Acoustic Prediction

Code release for **TAM-FNO**, a time-adaptive modulated Fourier Neural Operator for underwater acoustic transmission-loss prediction from sound-speed profiles.

This repository is organized for reproducible academic use. It contains only the TAM-FNO implementation and its preprocessing, training, evaluation, visualization, tests, and GitHub Pages site. It intentionally does **not** contain comparison-model code, experimental data, checkpoints, logs, normalizers, or generated result archives.

## Repository Layout

```text
src/tam_fno/                 Core package: models, data utilities, time encoding, training helpers
experiments/tam_fno/         TAM-FNO training, evaluation, and visualization scripts
scripts/                     Preprocessing and TAM-FNO shell helpers
tests/                       Lightweight smoke tests
data/                        Local generated data directory, ignored except README
docs/                        Static GitHub Pages site
```

## Setup

```bash
conda env create -f environment.yml
conda activate tam_fno
pip install -e .
```

For pip-only environments:

```bash
pip install -r requirements.txt
pip install -e .
```

## Data Policy

No experimental data is tracked in Git.

Expected raw files, kept outside Git:

```text
raw_data/TL.mat   # contains square_TL
raw_data/SSP.mat  # contains square_SSP
```

You can also point to raw data explicitly:

```bash
export TAM_FNO_RAW_DATA=/path/to/raw_data
python scripts/preprocess_data.py --raw-tl /path/to/TL.mat --raw-ssp /path/to/SSP.mat
```

Preprocessing writes generated train/test files under `data/`, which is ignored.

## Training

Run TAM-FNO directly:

```bash
python experiments/tam_fno/train.py --epochs 100 --device cuda
```

Or use the shell helper:

```bash
bash scripts/train_tam_fno.sh --epochs 100 --device cuda
```

## Evaluation

```bash
python experiments/tam_fno/evaluate.py
python experiments/tam_fno/plot_rmse_curve.py
python experiments/tam_fno/visualize_perturbation.py
```

Model checkpoints and generated figures are ignored by default. Pass explicit paths if your artifacts are stored elsewhere.

## Tests

```bash
PYTHONPATH=src pytest -q
```

The included tests only check model import and tensor shapes, so they do not require private data.

## Citation

If this code is useful for your research, cite the associated paper and this repository. A machine-readable citation template is provided in `CITATION.cff`.

## License

Released under the MIT License.
