# NGAFID before/after MiniRocket baseline

基于复现论文 *A Large-Scale Annotated Multivariate Time Series Aviation Maintenance Dataset from the NGAFID* 中，NGAFID `2days` 子集上的 **before_after（二分类）** 任务。

当前主入口：

- `run_minirocket_before_after.py`

- 使用数据集中预先给定的 **5-fold split**
- 运行 **5-Fold Cross Validation**
- 输出并保存每一折的评估指标
- 汇总 `mean ± std`

## 1. 环境安装

```bash
pip install -r requirements.txt
```

## 2. 数据准备

仓库 **不包含数据文件**


### 假如你已经自己下载了数据集

```text
./2days.tar.gz
```

程序会自动解压到：

```text
./2days/
```

### 你想通过Google Driver下载

原始数据加载逻辑会尝试从 Google Drive 下载；若下载失败，请自行下载。

## 3. 测试

### Smoke test

```bash
python run_minirocket_before_after.py --folds 0 --epochs 3
```

### 完整 5-Fold 运行

```bash
python run_minirocket_before_after.py --folds 0 1 2 3 4 --epochs 20
```

## 4. 输出

- `minirocket_before_after_cv.csv`：每一折的结果
- `minirocket_before_after_cv_summary.json`：5 折结果的均值与标准差

## 5. 结果文件格式

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

## 6. 关键文件

- `run_minirocket_before_after.py`：训练与评估入口
- `requirements.txt`：依赖
- `ngafiddataset/dataset/dataset.py`：数据读取、解压、下载与 fold 切分
- `results/`：实验输出目录

## 7. 设备说明

脚本会自动检测运行设备：

- 若 PyTorch + CUDA 可用，则使用 GPU
- 否则自动回退到 CPU

```bash
pip install -r requirements.txt
```

如果你本地需要 GPU 加速，可按自己的 CUDA 版本额外安装匹配的 PyTorch 版本。
