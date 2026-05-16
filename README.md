# TAM-FNO Underwater Acoustic Prediction

Code release for **TAM-FNO**, a time-adaptive modulated Fourier Neural Operator for underwater acoustic transmission-loss prediction from sound-speed profiles.

This repository is organized for reproducible academic use. It contains the TAM-FNO implementation, preprocessing utilities, training and evaluation entry points, and the GitHub Pages site. Comparison-model code, experimental data, checkpoints, logs, normalizers, tests, and generated result archives are kept outside version control.

## Repository Layout

```text
src/                         TAM-FNO core modules: model, data utilities, time encoding, training helpers
src/scripts/                 Minimal TAM-FNO training and evaluation entry points
scripts/                     Preprocessing and TAM-FNO shell helpers
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
python src/scripts/train.py --epochs 100 --device cuda
```

Or use the shell helper:

```bash
bash scripts/train_tam_fno.sh --epochs 100 --device cuda
```

## Evaluation

```bash
python src/scripts/evaluate.py
```

Model checkpoints and generated outputs are ignored by default. Pass explicit paths if your artifacts are stored elsewhere.

## Citation

If this code is useful for your research, cite the associated paper and this repository. A machine-readable citation template is provided in `CITATION.cff`.

## License

Released under the MIT License.
