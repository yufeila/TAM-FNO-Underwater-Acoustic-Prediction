# 仓库文件保留说明

这个文件用于判断当前仓库里哪些内容应该保留，哪些内容不应该上传到 GitHub。

## 建议保留并上传

### `README.md`

仓库首页说明。必须保留，用来介绍项目用途、安装方式、数据说明、训练和评估命令。

### `src/`

核心代码目录。必须保留。

其中 `src/tam_fno/` 是 TAM-FNO 的 Python 包，包含模型文件 `tam_fno_model.py`、数据读取、时间特征、归一化、训练逻辑等代码。

TAM-FNO 的最小复现脚本也放在 `src/tam_fno/scripts/` 下，例如：

- `src/tam_fno/scripts/train.py`
- `src/tam_fno/scripts/evaluate.py`

不再保留顶层 `experiments/` 目录。

### `scripts/`

顶层命令脚本。建议保留。

这里不放 Python 实验代码，只放很薄的 shell 入口，例如：

- `scripts/preprocess_data.py`
- `scripts/train_tam_fno.sh`
- `scripts/evaluate_tam_fno.sh`

### `docs/`

项目网站目录。需要网站就保留。

GitHub Pages 会读取这里的内容。`docs/index.html` 是网站主页，`docs/assets/` 存放网页图片。

### `.gitignore`

必须保留。

它负责阻止实验数据、模型权重、日志、缓存等文件被上传。

### `pyproject.toml`

建议保留。

它让这个项目可以用 `pip install -e .` 作为标准 Python 包安装。

### `environment.yml`

建议保留。

Conda 环境配置文件，方便别人复现运行环境。

### `requirements.txt`

建议保留。

给不用 Conda 的用户安装依赖。

### `LICENSE`

建议保留。

说明代码开源许可。

### `CITATION.cff`

建议保留。

GitHub 和论文工具可以用它生成引用信息。

### `CONTRIBUTING.md`

可保留。

用于说明贡献规范，尤其提醒不要提交数据和实验产物。

## 不应该上传

### `data/`

不应该上传。

这里是预处理后生成的数据目录，属于本地运行产物。包括 `data/README.md` 也不再保留。

### `graphify-out/`

不应该上传。

这是本地代码图工具生成的辅助目录，只用于本地理解代码结构。它不是 TAM-FNO 代码本身。

### `tests/`

不应该上传。

当前仓库面向论文复现者，只保留 TAM-FNO 的代码和运行入口，不保留额外测试目录。

### `__pycache__/`

不应该上传。

这是 Python 自动生成的缓存。

### `logs/`

不应该上传。

训练日志属于实验产物。

### `results/`

不应该上传。

评估结果、统计表和生成图都属于实验产物。

### `runs/`

不应该上传。

训练输出目录通常包含模型权重、loss 曲线、预测图等。

### `checkpoints/`

不应该上传。

模型权重文件通常很大，而且属于实验产物。

### `normalizers/`

不应该上传。

归一化参数来自训练数据，和实验数据相关。

## 不应该上传的文件类型

以下文件类型一般都不应该进入 GitHub：

- `*.mat`：原始或处理后的 MATLAB 数据
- `*.npz`、`*.npy`：NumPy 数据
- `*.pt`、`*.pth`：PyTorch 权重或归一化参数
- `*.pkl`：序列化对象
- `*.h5`、`*.hdf5`：HDF5 数据
- `*.log`：日志文件
- `.DS_Store`：macOS 系统文件

## 当前判断标准

这个仓库应该只保留三类内容：

1. TAM-FNO 代码。
2. 运行 TAM-FNO 所需的脚本和配置。
3. 介绍项目和网站展示所需的文档。

不应保留：

1. 实验数据。
2. 模型权重。
3. 训练日志。
4. 生成结果。
5. 其他模型或对比实验代码。
6. 额外测试目录。
7. 本地工具缓存。
