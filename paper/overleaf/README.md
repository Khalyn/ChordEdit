# Overleaf 工程（CVPR 格式）

按 **CVPR「Author Guidelines for Proceedings」** 模板组织的论文工程，
负责内容为团队架构图中的 **High Frequency Completion（蓝色块）**。
代码部分已冻结，本目录只做论文。

```
overleaf/
├── main.tex            # 正文：12 个公式 + 算法 1 + 1 跨栏表 + 2 图
├── references.bib      # 17 条文献，已按 CVF 页面逐条核对
├── figures/
│   ├── architecture.png  # 团队架构图（SHARP-Edit 全图，我们负责最右蓝块）
│   └── comparison.png    # 定性对比（scripts/eval_hfc.py 产出）
└── README.md
```

## 在 Overleaf 上跑起来

1. Overleaf → New Project → **CVPR 2026 Paper Template**（官方模板）。
2. 用本目录的 `main.tex` **替换**模板的 `main.tex`
   （模板自带的 `cvpr.sty` / `cvpr_eso.sty` / `eso-pic.sty` / `ieee_fullname.bst` 保留）。
3. 上传 `figures/` 两张图。
4. 编译器 **pdfLaTeX**，直接编译。
5. 匿名投稿保持 `\usepackage[review]{cvpr}`；录用后去掉 `[review]` 并填作者。

> 如果报 `cvpr.sty not found`，说明工程不是从官方模板创建的：
> 把模板压缩包里的 `*.sty` / `*.bst` 一起上传到根目录即可。

## 论文结构（8 页 CVPR 双栏）

| 章节 | 内容 |
|---|---|
| Abstract | 三个测量 → 三个结论（+6.8 dB / HF 0.944→1.002 / 前景 +13%） |
| §1 Introduction | 缺口不是语义而是细节分布；三条观察 (i)(ii)(iii)；三条贡献 |
| §2 Related Work | 一步生成与编辑 / 背景保持 / 细节与频域 / 现有像素锐化模块 |
| §3 Method | 3.1 上下文 → 3.2 背景分支 VRR → 3.3 细节探针 → 3.4 门控与增益 → 3.5 LF 保留/HF 替换 → 3.6 合成与两项精修 → 算法 1 + 复杂度 |
| §4 Experiments | 协议（固定分区）/ 指标 / 主表 / 三组消融 / 定性 |
| §5 Limitations | 规模、mask 来源、编辑保真、开销、接缝感 |
| §6 Conclusion | 含两条负面结论（VRR 与能量匹配何时失效） |

## 与团队架构图的对应

| 架构图模块 | 颜色 | 归属 | 论文位置 |
|---|---|---|---|
| Temporal Adaptive Field | 黄 | 其他成员 | §3.1 一句话上下文 |
| Spatial Adaptive Localization | 绿 | 其他成员 | §3.1 式(1) |
| **High Frequency Completion** | **蓝** | **本论文** | **§3.2–§3.6** |

蓝块内部逐步对应：

| 架构图步骤 | 论文位置 |
|---|---|
| `r = I_src − Dec(Enc(I_src))`，`I_VRR = I_local + (1−M)⊙r` | §3.2 式(2)，并给出"何时失效"的推导 |
| Gaussian Blur & Unsharp Masking（背景分支） | §3.2 式(3)，默认 `γ_um = 0`，消融项 |
| `x_probe = pred_x0(x_edit, t_probe, e_tar)` | §3.3 式(4) |
| Calculate gating / Determine gain | §3.4 式(5)(6) |
| Low-frequency retention, high-frequency replacement | §3.5 式(7) |
| `I_out = M⊙I_final + (1−M)⊙I_UM` | §3.5 式(8) + 与实测配置的对应说明 |
| 近二值 mask / 引导滤波贴轮廓 / 接缝色差补偿 | §3.6 式(9)–(11) |

## 数据

表中所有数字来自 `scripts/eval_hfc.py`（8 张示例图，fp16，seed 42）：

```bat
.venv\Scripts\python.exe scripts\eval_hfc.py --limit 8 --tag final
```

指标口径（严格背景 `α<0.1`、严格前景 `α>0.9`、接缝带单独统计）见脚本头部与
`docs/HFC_EBC_实现说明.md` §5。

## 还没做（投稿前）

- [ ] **PIE-Bench 全量**（700 张）+ GT mask；当前仅 8 张示例图。
- [ ] **CLIP-Whole / CLIP-Edited / LPIPS / 结构距离** 四列指标。
      §5 已如实写明"无法支撑 SOTA 结论"。
- [ ] 门控开关（有/无 gating）的**定量**消融；正文目前只做了定性描述。
- [ ] `t_probe` 与 `σ_hf` 的定量扫描表（正文给出的是定性趋势）。
- [ ] 若采纳架构图的背景分支锐化：补 `γ_um ∈ {0, 0.3, 0.6}` 的消融。

## 附录/补充材料建议

如果允许 supplementary，建议在单独文件里补：超参表（`t_p, τ, w_g, σ_hf, λ, w_α, r, κ`）、
掩码阈值敏感性曲线、失败案例（全图级编辑如 006 秋→冬）、以及 8 张图的逐图指标。
