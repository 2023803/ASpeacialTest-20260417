# NGAFID before/after MiniRocket baseline

本仓库复现论文 *A Large-Scale Annotated Multivariate Time Series Aviation Maintenance Dataset from the NGAFID* 中，NGAFID `2days` 子集上的 **before_after（二分类）** 任务。

当前主入口：

- `run_minirocket_before_after.py`

该脚本会：

- 使用数据集中预先给定的 **5-fold split**
- 运行 **5-Fold Cross Validation**
- 输出并保存每一折的评估指标
- 汇总 `mean ± std`

## 1. 环境安装

建议先创建虚拟环境，再安装依赖：

```bash
pip install -r requirements.txt
```

## 2. 数据准备

仓库 **不包含数据文件**，因为 `2days.tar.gz` 体积较大，不适合上传到 GitHub。

你需要自行准备以下两种形式之一，并放在仓库根目录：

### 方式 A：放置压缩包

```text
./2days.tar.gz
```

程序会自动解压到：

```text
./2days/
```

### 方式 B：提前解压完成

如果你已经解压好了，请确保仓库根目录下存在：

```text
./2days/flight_data.pkl
./2days/flight_header.csv
./2days/stats.csv
```

如果本地既没有 `2days/` 也没有 `2days.tar.gz`，原始数据加载逻辑会尝试从 Google Drive 下载；若下载失败，请手动放置数据文件后再运行。

## 3. 快速开始

### Smoke test

建议先跑 1 折 + 少量 epoch，确认环境、数据和训练流程正常：

```bash
python run_minirocket_before_after.py --folds 0 --epochs 3
```

### 完整 5-Fold 运行

```bash
python run_minirocket_before_after.py --folds 0 1 2 3 4 --epochs 20
```

如果不显式传入 `--folds`，脚本默认运行全部 5 折。

## 4. 输出结果

运行完成后，会在 `results/` 目录下生成：

```text
results/minirocket_before_after_cv.csv
results/minirocket_before_after_cv_summary.json
```

其中：

- `minirocket_before_after_cv.csv`：每一折的结果
- `minirocket_before_after_cv_summary.json`：5 折结果的均值与标准差

## 5. 评估指标

当前脚本会输出：

- Accuracy
- F1
- ROC-AUC

并打印：

- 每一折结果
- 交叉验证汇总结果（mean ± std）

## 6. 结果文件格式

`results/minirocket_before_after_cv.csv` 至少包含以下列：

- `fold`
- `train_size`
- `test_size`
- `accuracy`
- `f1`
- `roc_auc`

`results/minirocket_before_after_cv_summary.json` 至少包含以下字段：

- `num_folds`
- `accuracy_mean`
- `accuracy_std`
- `f1_mean`
- `f1_std`
- `roc_auc_mean`
- `roc_auc_std`

## 7. 关键文件

- `run_minirocket_before_after.py`：训练与评估入口
- `requirements.txt`：复现依赖
- `ngafiddataset/dataset/dataset.py`：数据读取、解压、下载与 fold 切分
- `results/`：实验输出目录

## 8. 设备说明

脚本会自动检测运行设备：

- 若 PyTorch + CUDA 可用，则使用 GPU
- 否则自动回退到 CPU

普通复现流程直接执行：

```bash
pip install -r requirements.txt
```

即可。

如果你本地需要 GPU 加速，可按自己的 CUDA 版本额外安装匹配的 PyTorch 版本。

## 9. GitHub 提交建议

建议在提交到 GitHub 时：

- 提交代码、README、requirements、results 中的小型结果文件
- **不要提交** `.venv_clean/`
- **不要提交** `2days/`
- **不要提交** `2days.tar.gz`

推荐先运行完整 5 折，再把最终的 CSV 和 JSON 结果文件一并提交，这样审阅者可以直接看到：

- 每折 accuracy
- mean ± std
- 仓库确实已经完整跑通

## 10. 当前仓库已生成的完整 5-Fold 结果

当前仓库中的结果文件对应一次完整 5-Fold CV 运行，结果如下：

- Accuracy: `0.5883 ± 0.0158`
- F1: `0.5706 ± 0.0143`
- ROC-AUC: `0.6297 ± 0.0210`

每折 accuracy 为：

- Fold 0: `0.5821`
- Fold 1: `0.5850`
- Fold 2: `0.6007`
- Fold 3: `0.5671`
- Fold 4: `0.6068`

对应结果文件：

- `results/minirocket_before_after_cv.csv`
- `results/minirocket_before_after_cv_summary.json`
