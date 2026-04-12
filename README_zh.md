# 公理化 Tau 空间 PPN

这是一个以 **公理化 Tau 空间 PPN** 为核心的项目化仓库，基于 Time-Series-Library 框架进行方法开发与基准评测。

仓库目标只有一个：
- 将 tau-space 理念做成可复现、可评测、可对比的预测系统。

## 1. 项目定位

本仓库主要关注：
- 内在时间（tau-space）公理化建模
- 非自举式整段 horizon 生成
- 全局映射 + 交叉调整预测路径
- few-shot 条件下的稳定性评测

不是通用模型大全文档，TSLib 在这里主要是基础设施。

## 2. 方法主线（公理化 Tau 空间）

当前 PPN 实现遵循以下流程：
1. 结构证据编码
2. 因果单调时钟
3. 在 tau 空间做主预测
4. 全局映射与交叉调整
5. 回投到观测时间域

核心思想：
- tau 是“内在坐标系”，不是单一手工统计量。
- 加速度可以作为证据，但不是 tau 定义本身。

## 3. 对应公理（实现目标）

实现中对齐以下性质：
- 因果性：tau 仅依赖当前及历史观测。
- 单调性：内在时间严格递增。
- 重参数化一致性：观测过程通过 tau 空间状态建模。
- 动力学简化：鼓励 tau 空间动力学更平坦。
- 预测有效性：改进需体现在 MSE/MAE。
- 可回投性：tau 空间表示需保留可重建信息。
- 规范不变意识：训练关注结构与顺序，不依赖绝对 tau 数值。

## 4. 关键代码文件

- `models/ppn.py`：公理化 tau-space PPN 模型
- `exp/exp_long_term_forecasting.py`：训练/验证及 tau 辅助损失
- `run.py`：参数入口与实验调度
- `utils/print_args.py`：参数打印
- `scripts/run_ppn_horizon_residual_fewshot.ps1`：few-shot 批量脚本
- `scripts/run_ppn_tau_axiom_overnight.ps1`：长时与断点续跑脚本

## 5. 数据与可比性

实验默认使用本地 TSLib 兼容数据格式，例如：
- `./dataset/ETT-small/ETTh1.csv`

要与外部结果可比，必须保证：
- 切分与预处理一致
- 任务设置一致（seq/label/pred 等）
- 训练预算一致

## 6. 快速开始（Windows PowerShell）

### 6.1 环境

```powershell
conda activate py39env
```

### 6.2 快速冒烟

```powershell
powershell -ExecutionPolicy Bypass -Command "& {
  .\scripts\run_ppn_horizon_residual_fewshot.ps1 \
    -Epochs 1 -Patience 1 -PredLen 96 \
    -Seeds @(42) -Ratios @(0.1) \
    -IncludeGatedResidual 0 -IncludeWarmupResidual 1 \
    -WarmupEpochs 1 -IncludeTauFieldVariant 0 \
    -Tag tau_axiom_smoke
}"
```

### 6.3 过夜与续跑

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_ppn_tau_axiom_overnight.ps1 -Resume 1
```

## 7. 输出文件说明

每次运行在 `logs/` 下生成三类文件：
- 明细：`... .csv`
- 汇总：`... _summary.csv`
- 排名：`... _rank.csv`

过夜任务主日志：
- `logs/tau_axiom_overnight_master.log`

## 8. 当前实践结论（随实验更新）

近期实验中：
- `horizon_residual` / `horizon_residual_warmup` 通常优于 `base`。
- `ratio=0.05` 相比 `0.1` 波动更明显。
- phase schedule 提供开关能力，但是否增益需按设置验证。

## 9. 复现实验建议

对外汇报请固定并记录：
- commit hash
- 数据路径与文件名
- seeds 与 ratios
- epochs / patience
- pred_len
- 完整运行命令

## 10. 致谢

本项目复用了 Time-Series-Library 的评测基础设施。
方法创新与实验推进聚焦于公理化 Tau 空间 PPN。
