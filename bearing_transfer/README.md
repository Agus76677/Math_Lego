# Bearing Transfer Pipeline

该工程实现高速列车轴承故障诊断跨域迁移的自动化管线，包含数据预处理、源域建模、迁移对齐与可解释性分析。详细的实验流程请参考 `reports/report.md`。若要运行完整实验，请确保将题目提供的数据放置于 `data/` 目录（或通过参数指定），目录结构示例：

```
data/
├── Y_datasets      # 源域 .mat 文件（12kHz / 48kHz）
└── MBY_datasets    # 目标域 .mat 文件（32kHz）
```

几何参数如滚动体数 `Z`、滚动体直径 `bd`、节圆直径 `pd` 和接触角 `θ` 默认设置为 CWRU 轴承典型值，如需覆盖其他轴承请在 `src/io_loader.py` 中调整 `DEFAULT_GEOMETRY`。

## 快速开始

```bash
python scripts/prepare_features.py --data_root ./data
python scripts/train_source.py
python scripts/transfer_and_predict.py
python scripts/explainability.py
python scripts/make_report.py
```

运行后所有中间件会存放在 `outputs/`，报告写入 `reports/report.md`。

## 运行环境

- Python >= 3.10
- 主要依赖见 `requirements.txt`
- 自动检测 GPU，如不可用则回退到 CPU

## 测试

```bash
pytest
```
