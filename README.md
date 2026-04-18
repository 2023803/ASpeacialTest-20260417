# NGAFID before/after MiniRocket baseline

基于论文 *A Large-Scale Annotated Multivariate Time Series Aviation Maintenance Dataset from the NGAFID*，复现并扩展 NGAFID `2days` benchmark subset 上的 **before/after（二分类）** 任务。

当前主入口：

- `run_minirocket_before_after.py`

该脚本支持：

- 使用数据集自带的 **5-fold split**
- 运行 **5-Fold Cross Validation**
- 输出并保存每折结果与 `mean ± std`
- 切换论文式 `global` 归一化与更严格的 `fold` 归一化
- 调整输入长度 `max_length`
- 导出逐样本预测结果，用于轻量化错误分析

---

## 1. 项目目标

本项目围绕论文 Section 3.2 的 benchmark subset 展开，任务是：

- 输入：23 维飞行传感器多变量时间序列
- 输出：判断该 flight 更接近 **维护前** 还是 **维护后** 状态

在复现论文 MiniRocket baseline 的基础上，本仓库进一步加入了 4 个小型改进/增强：

1. **更严格的归一化对比**
   - 对比论文式 `global normalization` 与 `fold-only normalization`
2. **输入长度消融实验**
   - 对比 `4096 / 2048 / 1024` 三种输入长度
3. **补充评估指标**
   - 在 Accuracy 基础上补充 F1 与 ROC-AUC
4. **轻量化错误分析**
   - 导出逐样本预测结果，分析类别错误率与序列长度分布

---

## 2. 环境安装

```bash
pip install -r requirements.txt
```

---

## 3. 数据准备

仓库 **不包含原始数据文件**。


### 自己下载

```text
./2days.tar.gz
```
```text
./2days/
```

### 自动下载说明

原始加载逻辑会尝试从 Google Drive 下载；若下载失败，请手动放置数据文件。

---

## 4. 快速开始

### 4.1 Smoke test

建议先跑 1 折 + 少量 epoch，确认环境、数据和训练流程正常：

```bash
python run_minirocket_before_after.py --dataset-dir . --folds 0 --epochs 3
```

### 4.2 论文式 baseline（global + 4096）

```bash
python run_minirocket_before_after.py --dataset-dir . --folds 0 1 2 3 4 --epochs 20 --normalization global --max-length 4096 --results-dir results/global_norm
```

### 4.3 更严格归一化对比（fold-only）

```bash
python run_minirocket_before_after.py --dataset-dir . --folds 0 1 2 3 4 --epochs 20 --normalization fold --max-length 4096 --results-dir results/fold_norm
```

### 4.4 输入长度消融实验

#### `max_length = 2048`

```bash
python run_minirocket_before_after.py --dataset-dir . --folds 0 1 2 3 4 --epochs 20 --normalization global --max-length 2048 --results-dir results/global_norm_len2048
```

#### `max_length = 1024`

```bash
python run_minirocket_before_after.py --dataset-dir . --folds 0 1 2 3 4 --epochs 20 --normalization global --max-length 1024 --results-dir results/global_norm_len1024
```

### 4.5 轻量化错误分析 baseline

```bash
python run_minirocket_before_after.py --dataset-dir . --folds 0 1 2 3 4 --epochs 20 --normalization global --max-length 4096 --results-dir results/error_analysis_baseline
```

---

## 5. 输出文件说明

每次运行至少会生成：

- `minirocket_before_after_cv.csv`：每折汇总结果
- `minirocket_before_after_cv_summary.json`：5 折均值与标准差

对于错误分析版本，还会额外生成：

- `minirocket_before_after_cv_predictions.csv`：逐样本预测结果

### 5.1 `*_cv.csv` 包含字段

- `fold`
- `train_size`
- `test_size`
- `normalization`
- `accuracy`
- `f1`
- `roc_auc`

### 5.2 `*_cv_summary.json` 包含字段

- `num_folds`
- `accuracy_mean`
- `accuracy_std`
- `f1_mean`
- `f1_std`
- `roc_auc_mean`
- `roc_auc_std`
- `normalization`
- `max_length`

### 5.3 `*_cv_predictions.csv` 包含字段

- `fold`
- `id`
- `target_class`
- `class`
- `hclass`
- `true_label`
- `pred_label`
- `positive_score`
- `correct`
- `seq_len`

该文件主要用于：

- 按维护类别统计错误率
- 分析正确/错误样本的序列长度分布

---

## 6. 当前实验结果汇总

### 6.1 论文式 baseline（global + 4096）

对应目录：`results/global_norm/`

- Accuracy: `0.5902 ± 0.0122`
- F1: `0.5733 ± 0.0112`
- ROC-AUC: `0.6272 ± 0.0155`

### 6.2 更严格归一化对比（fold + 4096）

对应目录：`results/fold_norm/`

- Accuracy: `0.5929 ± 0.0086`
- F1: `0.5804 ± 0.0099`
- ROC-AUC: `0.6317 ± 0.0113`

### 6.3 输入长度消融（global）

#### `max_length = 2048`

对应目录：`results/global_norm_len2048/`

- Accuracy: `0.5962 ± 0.0067`
- F1: `0.5752 ± 0.0066`
- ROC-AUC: `0.6364 ± 0.0071`

#### `max_length = 1024`

对应目录：`results/global_norm_len1024/`

- Accuracy: `0.5992 ± 0.0152`
- F1: `0.5837 ± 0.0087`
- ROC-AUC: `0.6359 ± 0.0144`

### 6.4 轻量化错误分析 baseline

对应目录：`results/error_analysis_baseline/`

- Accuracy: `0.5949 ± 0.0134`
- F1: `0.5789 ± 0.0085`
- ROC-AUC: `0.6310 ± 0.0158`

---

## 7. 关键结论

- 在更严格的 `fold-only normalization` 下，指标没有下降，F1 还有小幅提升，说明更严格的预处理设定是可行的。
- 输入长度从 `4096` 缩短到 `2048` 或 `1024` 后，整体表现没有变差，说明与维护状态相关的判别信息可能更多集中在飞行后段。
- 逐样本错误分析表明，不同 maintenance class 的错误率存在差异；相比之下，序列长度对错误的影响相对较弱。

---

## 8. 关键文件

- `run_minirocket_before_after.py`：训练、评估与结果导出入口
- `requirements.txt`：依赖列表
- `ngafiddataset/dataset/dataset.py`：数据读取、解压、下载与 fold 切分
- `results/`：实验输出目录

---

## 9. 设备说明

脚本会自动检测运行设备：

- 若 PyTorch + CUDA 可用，则使用 GPU
- 否则自动回退到 CPU

如果你本地需要 GPU 加速，可按自己的 CUDA 版本安装匹配的 PyTorch。
