# 上传材料说明

这份文件夹供放入小组仓库的 `reproduction/task-a/`。其中包含本次使用的运行脚本、固定版本的原版核心代码、环境版本和实际评分记录。本次按确认的范围保留 700 条官方提示词、编辑指令和文字标注，以及实际运行清单。原图、生成图片、模型权重和本地环境均不包含在上传包中。模型权重按后文步骤下载。

本文件原有的实验结果说明保留在下方，末尾补充从这个上传包开始准备资源的步骤。下方“主要文件”和“在当前主机复跑”描述 Linux 实验目录；GitHub 上传包的实际文件范围见本文末尾。

---

# ChordEdit 原版复现

本项目使用作者发布的 ChordEdit 和 SD-Turbo，在 PIE-Bench v1 的全部 700 条样本上运行原版模型并评分，作为小组后续改进模型的对照结果。

## 已完成的工作

- 配置实验环境，固定作者代码和模型版本。
- 先运行 30 条调试样本，再跑完 700 条全量样本。
- 保存每条的原图、前后提示词、编辑指令、运行参数和原版输出。
- 自动计算七项质量评分，保存逐图分数和整体汇总。
- 核对全量运行与调试集重叠的 30 条，输入和输出文件均一致。

700 条全部出图，推理失败和跳过均为 0。当前已完成原版配置的全量运行和质量评分。

## 数据与运行环境

数据为官方 PIE-Bench v1，共 700 张 512×512 原图、10 类编辑。原图、source prompt、target prompt、edit_instruction 和编辑区域标注均由数据集提供。

- `source_prompt` 描述编辑前的图片。
- `target_prompt` 描述希望得到的图片。
- `edit_instruction` 说明要改什么。

原版模型实际接收原图及前后两条描述。编辑区域标注用于评分。

| 项目 | 本次设置 |
| --- | --- |
| 系统与显卡 | Ubuntu 24.04.5，RTX 3090 24 GB |
| Python | 3.12.3 |
| PyTorch / CUDA | 2.5.0+cu124 / 12.4 |
| TorchVision | 0.20.0+cu124 |
| Diffusers / Transformers | 0.35.2 / 4.51.3 |
| 模型 | SD-Turbo |
| 输入 / 随机种子 | 512×512，FP32，seed=42 |

采用官方默认编辑参数，启用 `cleanup`。完整运行设置见 `run_baseline.sh` 和 `pie-700/manifest.jsonl`，完整依赖见 `pie-700/package-versions.txt`。

## 评分结果

下表对照论文中带默认后处理的 ChordEdit（SD-Turbo）配置。MSE、LPIPS、SSIM 和结构距离按论文的显示单位换算。

| 指标 | 方向 | 本次 | 论文 | 有效样本 |
| --- | --- | ---: | ---: | ---: |
| 背景 PSNR | 越高越好 | 22.63 | 22.20 | 556 |
| 背景 MSE × 1000 | 越低越好 | 7.99 | 6.84 | 556 |
| 背景 LPIPS × 1000 | 越低越好 | 122.64 | 128.25 | 556 |
| 整图 CLIP | 越高越好 | 24.82 | 25.58 | 700 |
| 编辑区域 CLIP | 越高越好 | 22.13 | 22.96 | 700 |
| 背景 SSIM × 100 | 越高越好 | 76.71 | 75.91 | 556 |
| 结构距离 × 1000 | 越低越好 | 28.79 | 16.58 | 700 |

背景 PSNR 接近论文，两项 CLIP 均低约 0.8，结构距离差异较明显。具体原因尚未确定。

**为什么背景指标只有 556 条？** 这里的“背景”指编辑区域以外、应保留的部分。700 条样本中，有 144 条按官方函数处理标注后整张图都属于编辑区域，剩余背景为空。原评测函数对这些样本的背景指标返回 `NaN`，汇总时使用其余 556 条有效分数。700 条图片均已运行并保存输出。

本次已核对，四项背景指标中出现 `NaN` 的样本编号，与背景为空的 144 个编号完全一致。论文的逐图有效样本集合尚未与作者记录对齐。

**评分代码来源。** `evaluation-tools/score_pie.py` 负责批量调用和汇总，具体评分函数使用 PIE-Bench 作者公开的实现。PSNR、MSE、SSIM 按公式计算；CLIP、LPIPS 和结构距离分别使用 CLIP、SqueezeNet 和 DINO 提取特征后计算分数。

## 主要文件

```text
upstream/                         作者发布的 ChordEdit 代码
sd-turbo/                         编辑模型文件
.venv/                            正式实验环境
prepare_samples.py                样本整理与检查脚本
run_baseline.sh                   原版运行入口
evaluation-tools/score_pie.py     批量评分入口

pie-30/                          30 条调试实验
pie-700/                         700 条全量实验
  annotation_images/             完整的 700 张输入原图
  mapping_file.json              官方文字与编辑区域标注
  manifest.jsonl                 输入、参数、输出路径和文件校验值
  output/ChordEdit/
    annotation_images/           原版输出图片
  metrics_per_image.csv          逐图原始分数
  metrics_summary.json           整体均值与论文数值对照
  run.log / scoring.log          推理与评分日志
  package-versions.txt           完整依赖版本
  runtime.txt / validation.json  环境与核对记录
```

完整输入位于 `pie-700/annotation_images/`。`pie-official/annotation_images/` 在 Linux 上只留有最初的 30 张图。

组员可先用 30 条调试，再按相同样本编号、原图和两条提示词测试改进模型。评分时沿用同一份标注和同一套评测实现。

## 在当前主机复跑

以下命令在本实验总目录中执行，复用现有环境和权重，并将结果保存在新的 `pie-700-rerun/` 目录。

```bash
mkdir pie-700-rerun
cp pie-700/mapping_file.json pie-700-rerun/
cp -a pie-700/annotation_images pie-700-rerun/

.venv/bin/python prepare_samples.py pie-full pie-700-rerun
bash run_baseline.sh pie-700-rerun

export HF_HUB_CACHE="$PWD/evaluation-cache/hf"
export TORCH_HOME="$PWD/evaluation-cache/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
.venv/bin/python evaluation-tools/score_pie.py pie-700-rerun
```

本机环境已验证可用，换机器从零安装仍待验证。

## 来源与版本

- [ChordEdit 论文](https://arxiv.org/html/2602.19083v1)，本次对照表 1 与附录 G。
- [作者代码固定版本](https://github.com/ChordEdit/ChordEdit/tree/a33d8cc26108851c6545b998c0e939006f89c9bd)，版本号 `a33d8cc26108851c6545b998c0e939006f89c9bd`。
- [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo)，权重版本 `b261bac6fd2cf515557d5d0707481eafa0485ec2`。
- [PIE-Bench 官方入口](https://github.com/cure-lab/PnPInversion#benchmark-download)与[评测代码固定版本](https://github.com/cure-lab/PnPInversion/tree/07f97f448150e2ca220bebd54c8f687c5c50c67a/evaluation)。


## 从上传包准备复跑

以下命令在上传包所在目录执行，适用于 Ubuntu、NVIDIA 显卡和 Python 3.12。Linux 主机上的原始环境已经完成实跑验证，本上传包新增的资源准备脚本需要联网下载依赖，跨机器的完整安装仍待验证。

### 1. 安装记录中的环境版本

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

### 2. 准备评分代码和模型

`upstream/` 已包含本次使用的作者核心代码和 MIT 许可证。`prepare_resources.py` 按 `evaluation-sources.json` 下载固定版本的评分代码和 DINO 代码，并检查文件校验值。

```bash
.venv/bin/python prepare_resources.py sources
.venv/bin/python prepare_resources.py models
```

模型从公开来源下载。若本机连接 Hugging Face 较慢，可在运行 `models` 前设置 `HF_ENDPOINT=https://hf-mirror.com`，脚本固定的模型版本保持一致。

### 3. 准备图片并使用随包的提示词

`pie-700/mapping_file.json` 包含 700 条官方前后描述、编辑指令和区域标注（数字形式的蒙版，不是图片）。`pie-700/prompts.csv` 提供便于查看的文字表，包括样本编号、原图相对路径、source prompt、target prompt、edit_instruction 和 seed。文字均直接来自官方标注，没有改写。

原图通过 [PIE-Bench 官方入口](https://github.com/cure-lab/PnPInversion#benchmark-download)申请并下载，在本地放到 `pie-700/annotation_images/`，保留官方子目录结构。该目录被 Git 忽略。随后按前文命令复制标注和原图到新建的 `pie-700-rerun/`，再整理、运行和评分。

逐图 CSV 中的样本编号是文本，使用 Excel 导入时请将 `id` 列设置为文本，保留开头的零。`NaN` 表示该项没有有效数值；背景四项各有 556 条有效分数，其余三项各有 700 条。

### 4. 区分已有实跑记录与新运行

`pie-700/metrics_per_image.csv`、`metrics_summary.json` 和 `validation.json` 来自本次实验主机实跑。新环境使用 `pie-700-rerun/` 另存结果，按编号与这些记录比较。

数据使用遵循 PIE-Bench 的授权要求。各组员取得官方数据后，可在本地生成含图片和提示词的清单。


### 5. 本次 Git 上传的具体范围

- 运行和评分脚本、依赖版本、固定版本的原版核心代码与许可证。
- `pie-700/prompts.csv`：700 条样本的提示词、编辑指令、seed 和原图相对路径。
- `pie-700/mapping_file.json`：官方原始标注，供原版读取与评分，保留编辑区域数字标注。
- `pie-700/manifest.jsonl`：从 Linux 实验目录取回的实际运行记录，含每条样本的输入输出相对路径、SHA256、参数、模型版本与生成状态。记录中写有路径，不代表图片包含在仓库中。
- `pie-700/metrics_per_image.csv`、`metrics_summary.json`、`validation.json`：实测分数、汇总与核对结果。

原图和原版输出图片仍保存在 Linux 实验主机。下载本仓库后需要自行准备原图才能复跑。`requirements-lock.txt` 对应原说明中的完整依赖记录；上传包没有 Linux 的 `.venv/`、`sd-turbo/`、运行日志和生成图片。
