<div align="center">

# OpenBot-Sim · 统计学版

**视觉模仿学习策略的统计设计与评估。**

模拟器与 [openbot-sim](https://github.com/LuC-lis/openbot-sim) 相同，但换一副眼镜看它。
项目始于一次捷径学习（shortcut learning）失效——一个不看摄像头也能拿满闭环纠错
成功率的策略——如今被组织成一个设计好的实验：条件用拉丁超立方抽样、结果用区间
估计报告、策略比较按配对种子做、失效模式用回归拟合而不是写成一个成功率。

</div>

---

## 把问题写成可以被否证的形式

> **这个策略到底有没有在用摄像头，它又为什么失效？**

前半句是关于**机制**的断言。这种断言很容易说得含混到任何结果都无法反驳它。所以这里
把它写成一个原假设：

> **H₀** —— 策略输出的转向指令不依赖于摄像头图像。

在 H₀ 下，把同一条件里的真实图像换成噪声图像，转向幅度（span）不应改变，配对差值
是可交换的。检验用**配对置换检验**：不依赖分布假设，给定种子后完全可复现
（`statistics/tests.py`、`experiments/probe_permutation.py`）。

后半句"为什么失效"不是一个成功率能回答的，所以直接拟合：以失败指示变量为因变量、
以条件因子为自变量做 **logistic 回归**，给出 Wald 标准误与优势比
（`statistics/regression.py`）。

---

## 关键结果

下面有两类数字，故意分开写。

**（a）分析层——可以从提交进仓库的逐回合记录重算。**

`results/outcomes_design.csv` 里是 240 个回合（2 个策略 × 40 个条件 × 3 次重复，
每次 400 步上限）。下表每个区间都由 `python -m statistics.analyze` 从该文件重新算出。

| 结果 | 数值 | 出处 |
|---|---|---|
| 特权老师策略跑满全程的比例 | **62.5 %**（75/120），95 % Wilson 区间 **[53.6 %, 70.6 %]** | `results/report_design.md` |
| 随机游走基线 | **0 %**（0/120），95 % Wilson 区间 **[0 %, 3.1 %]** | 同上 |
| 配对差值（老师 − 随机，同条件同种子） | **+62.5 个百分点** [53.3, 71.7]，置换检验 p = **0.001** | 同上 |
| 跟踪误差的配对差值 | **−0.275 m** [−0.335, −0.216] | 同上 |
| 老师的失败**全部是碰撞**（45/120，越界 0、打转 0）；条件因子对失败**均无显著预测力**（p 均 ≥ 0.15） | logistic 拟合，n = 120 | 同上 |
| 设计中每个因子在每个重复内都恰好占满各分层 | 见 `results/design_coverage.png` | 同上 |
| 这个设计能检测出多大的差异 | 成功率上约 **16.4 个百分点**、配对跟踪误差上 **dz = 0.256**（80 % 功效） | `statistics/power.py` |

最后一行的"无显著结果"就按无显著结果写：在 120 个回合、45 个事件下，这个模型只能
排除起点位姿、障碍位置与扰动率的**强**效应，别的都排除不了。

**（b）学习器的记录性结果——来自早期诊断实验。**

这些是训练流水线单次跑出来的数字（`models/dg_s3.tflite`），不由分析 CLI 重算。

| 发现 | 证据 |
|---|---|
| 默认网络结构会悄悄忽略摄像头输入，退化成开环控制 | `experiments/probe_ablation.py` |
| 根因锁定在 **dropout**，在 p = 0.4 与 0.5 之间出现断崖；用噪声图像对照按**效应量**判定阈值 | `results/dropout_sweep.png` |
| **DAgger** 三轮把闭环纠错成功率从 **20 % 提到 100 %** | `results/dagger_progress.png` |
| GPU 流程峰值显存 **8.4 GB → 1.9 GB** | `scripts/train.py` |

---

## 为什么这算"设计过的"实验

| 设计决定 | 理由 |
|---|---|
| 六个因子（初始位姿、航向、指令、障碍位置、外观种子、扰动率）上用**拉丁超立方** | 40 个条件即可均匀覆盖每个因子的取值域，不会成团（见 `results/design_coverage.png`） |
| **每个策略用完全相同的回合种子** | 比较是**配对**的，条件间差异被消掉，而不是被算进区间宽度里 |
| **显式的外观随机化**（地面纹理、光照、色温、暗角、传感器噪声） | 成功率应当是对场景分布的估计，而不是对一个"记住的"场景的估计 |
| **障碍物是几何，不是贴图** | 被占住的车道会逼出车道保持永远不会遇到的决策；老师的失败 100 % 来自它 |
| 日志里把失效模式分开记（`off_road` / `turned` / `collision`） | "失败了"不是一条结论 |
| 用 **Wilson** 区间而不是 Wald | 策略的成功率正好落在 0 % 和 100 %，正态近似在那里根本不成立 |
| 用**置换检验**而不是 t 检验 | 可交换性假设被写下来并检查过，而不是默认成立 |
| **事先的功效分析**（62.5 % vs 40 % 在 80 % 功效下需每个策略 77 个回合；实际跑了 120） | 样本量在看数据之前定好，事后同时报告"最小可检测效应" |

完整协议（含"这套分析做不到什么"）：`docs/statistical_protocol.md`。

---

## 复现

```bash
git clone https://github.com/LuC-lis/openbot-sim-statistics && cd openbot-sim-statistics
pip install -r requirements.txt

python -m tests.test_statistics                                        # 12/12，约 10 秒
python -m statistics.evaluate_design --out results/outcomes_design.csv # 约 3 分钟，纯 CPU
python -m statistics.analyze results/outcomes_design.csv --report results/report_design.md \
        --figure results/design_summary.png
```

`evaluate_design` 默认跑两个基线；加 `--model models/student.tflite` 可让学习到的策略
在完全相同的条件与种子上一起评估。

```bash
# 把"策略对输入不敏感"当成一个检验来跑（需要 TensorFlow 读 .tflite）
python experiments/probe_permutation.py --model models/dg_s3.tflite --json results/permutation_test.json
```

---

## 目录

```
statistics/            分析层：只依赖 numpy，不用 SciPy，不用 TensorFlow
  design.py              拉丁超立方 + 配对种子集 + 因子范围
  outcomes.py            按设计跑策略，每个回合记一行
  intervals.py           Wilson 区间、自助区间、配对自助
  tests.py               置换检验；把输入不敏感写成原假设
  regression.py          logistic 回归（IRLS）+ Wald 标准误 + 优势比
  power.py               事前样本量 + 最小可检测效应
  analyze.py             CLI：CSV → 区间表、模型表、markdown 报告
  evaluate_design.py     CLI：跑设计并写出 CSV

experiments/           诊断实验
openbot_sim/           从零写的模拟器（动力学、相机、世界、任务）
tests/test_statistics.py  分析层单元测试
docs/statistical_protocol.md  统计协议
results/               图 + 提交进仓库的逐回合 CSV 与报告
```

---

## 接口契约

模拟器完整复刻 OpenBot 真实接口，因此在这里训练出的 `.tflite` 可以直接丢进安卓应用：

| 量 | 规格 | OpenBot 出处 |
|---|---|---|
| 图像 | `(96, 256, 3)` float32，取值 `[0, 1]` | `Autopilot.java`、`assets/config.json` |
| 指令 `cmd` | 标量 `{-1, 0, +1}` | `Autopilot.java`、`utils/Enums.java` |
| 动作 | `[left, right] ∈ [-1, 1]` | `vehicle/Control.java` |

---

## 环境说明

- **分析层与 demo 只需 CPU**：`numpy`、`matplotlib`、`Pillow`。
- **TensorFlow（>= 2.9）**只在训练、`.tflite` 推理与
  `experiments/probe_permutation.py` 里需要。

## 归属与许可

接口契约与网络结构参照 [OpenBot](https://github.com/ob-f/OpenBot)
（Müller & Koltun, *OpenBot: Turning Smartphones into Robots*, ICRA 2021,
Apache-2.0）。`openbot_sim/nets.py` 是极简重写，本仓库不依赖 OpenBot 源码树。
MIT，见 [LICENSE](LICENSE)。
