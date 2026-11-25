# Overleaf 工程（CVPR 格式）

本目录是按 **CVPR「Author Guidelines for Proceedings」** 模板组织的论文工程，
负责内容为团队架构图中的 **High Frequency Completion（蓝色块）**。

```
overleaf/
├── main.tex            # 正文（CVPR 双栏，review 模式）
├── references.bib      # 参考文献（部分条目待核实，见文末 TODO）
├── figures/
│   ├── architecture.png  # 团队架构图（SHARP-Edit 全图，我们负责最右蓝块）
│   └── comparison.png    # 定性对比图（由 scripts/eval_hfc.py 产出）
└── README.md
```

## 在 Overleaf 上跑起来

1. Overleaf → New Project → **CVPR 2026 Paper Template**（官方模板）。
2. 用本目录的 `main.tex` **替换**模板里的 `main.tex`（模板自带的
   `cvpr.sty` / `cvpr_eso.sty` / `eps-pic.sty` / `ieee_fullname.bst` 保持不动）。
3. 上传 `figures/` 下两张图。
4. 编译器选 **pdfLaTeX**，直接编译。
5. 投稿匿名版保持 `\usepackage[review]{cvpr}`；录用后去掉 `[review]` 并填作者信息。

> 如果 Overleaf 提示 `cvpr.sty not found`，说明项目不是从 CVPR 模板创建的：
> 把官方模板压缩包里的 `*.sty` / `*.bst` 一起上传到工程根目录即可。

## 与团队架构图的对应关系

| 架构图模块 | 颜色 | 归属 | 本论文是否描述 |
|---|---|---|---|
| Temporal Adaptive Field | 黄 | 其他成员 | 仅在 §3.1 作为上下文一句话带过 |
| Spatial Adaptive Localization | 绿 | 其他成员 | 仅在 §3.1 作为上下文 |
| **High Frequency Completion** | **蓝** | **本论文** | **§3.2–§3.6 全文** |

蓝块内部与本论文的对应：

| 架构图步骤 | 论文位置 |
|---|---|
| `r = I_src − Dec(Enc(I_src))`，`I_VRR = I_local + (1−M)⊙r` | §3.2 式 (2)（VRR） |
| Gaussian Blur & Unsharp Masking（背景分支） | §3.2 式 (3)，默认 `γ_um = 0`，作为消融 |
| Latent-Space Detail Probe：`x_probe = pred_x0(x_edit, t_probe, e_tar)` | §3.3 式 (4) |
| Calculate gating | §3.4 式 (5)（门控） |
| Determine gain | §3.4 式 (6)（能量匹配，默认 `s = 1`，§4.3 说明为何） |
| Low-frequency retention, high-frequency replacement | §3.5 式 (7) |
| `I_out = M ⊙ I_final + (1−M) ⊙ I_UM` | §3.5 式 (8) |
| （实现细节）近二值 mask、引导滤波贴轮廓、接缝色差补偿 | §3.6 |

## 数据来源与复现

表中所有数字来自本仓库的评测脚本：

```bat
.venv\Scripts\python.exe scripts\eval_hfc.py --limit 8 --tag final
```

指标口径（严格背景 `α<0.1`、严格前景 `α>0.9`、接缝带单独统计）见
`scripts/eval_hfc.py` 头部说明，以及 `docs/HFC_EBC_实现说明.md` §5。

## 投稿前必须补的东西（TODO）

- [ ] **PIE-Bench 全量**（700 张）+ GT mask 重新评测；当前只有 8 张示例图。
- [ ] **CLIP-Whole / CLIP-Edited / LPIPS / 结构距离** 四列指标。
- [ ] `t_probe ∈ {0.05, 0.10, 0.20}` 与 `σ_hf ∈ {0.8, 1.0, 1.5}` 的定量扫描
      （正文 §4.3 末尾目前只有定性描述，代码里留了 `% TODO(exp)`）。
- [ ] 门控开关（有/无 gating）的定量消融。
- [ ] 参考文献：`swiftedit`、`hipa`、`fds`、`piebench` 四条在 `.bib` 里标了
      `TODO verify`，请核对准确出处后再投。
- [ ] 若最终决定采用架构图里的「背景分支 unsharp」：需要同条件跑
      `γ_um ∈ {0, 0.3, 0.6}` 的消融；目前 §3.2 默认 `γ_um = 0`。
