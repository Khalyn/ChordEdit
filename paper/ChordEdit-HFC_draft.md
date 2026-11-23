# Detail Completion and Contour-Aligned Compositing for One-Step Text-Guided Image Editing

> **Draft v0.1 — for internal review (not for submission yet).**
> Paper draft in English (ready to be ported to LaTeX); review notes in Chinese at the end.
> Numbers come from `scripts/eval_hfc.py --limit 8 --tag final3` on the 8 bundled examples.
> Figures: `docs/figures/comparison.png`.

---

## Abstract

One-step text-to-image models enable sub-second image editing, and recent training-free
editors such as ChordEdit achieve this by transporting the source latent along a low-energy
"chord" field. Their output, however, is visibly softer than the input photograph: the
background is pushed back through the VAE bottleneck, while the edited region is
under-detailed. We show that the background-preservation branch of such pipelines relies on a
*soft* energy mask that is deliberately γ-lifted, which lets a substantial fraction of the
edit leak into the "background": on a portrait example 57% of pixels are labelled background
but only 16% are truly unedited. We therefore propose two training-free, architecture-agnostic
pixel-space modules. **EBC** (contour-aligned compositing) remaps the latent mask into a
near-binary α, snaps its boundary to image edges with a guided filter, and cancels the local
exposure offset in the transition band. **HFC** (high-frequency completion) performs a single
extra UNet evaluation at a low noise level and replaces *only* the high band inside the gated
edit region, with no learnt parameters and no sharpening gain. On eight editing examples the
combination raises background PSNR from 50.0 dB to 56.6 dB, moves the background high-band
energy ratio from 0.944 to 1.002, increases foreground high-band energy by 13%, and reduces a
boundary-discontinuity metric by 17%, at the cost of one additional function evaluation
(+0.26 s). We additionally report two negative results: VAE residual recovery is subsumed by
pixel-space compositing (44.1 dB vs 56.6 dB), and unsharp masking over-shoots (HF 1.41) while
degrading background fidelity.

---

## 1. Introduction

One-step text-to-image models such as SD-Turbo, SwiftBrush-v2 and InstaFlow generate an image
in a single network evaluation. ChordEdit turns this capability into an editor by recasting
editing as optimal transport between the source and target prompt distributions: it smooths
the difference of the source and target drifts over a short time window (the *Chord control
field*) and integrates it in one large step. The result is an edit that runs in ~0.4 s on a
24 GB GPU while preserving non-edited structure.

In practice, users of these systems report a perceptual gap that is *not* semantic: the edited
image looks **soft** compared with the photograph they uploaded, and the boundary between the
edited object and the background is visible as a faint halo. This paper targets exactly that
gap. Our starting point is an empirical decomposition of where the sharpness is lost in the
ChordEdit pipeline (Sec. 3.1):

1. A VAE round trip of the source image retains only ~88% of its high-band energy (32.2 dB
   PSNR on our data). Any background reconstructed through the decoder is capped by this.
2. The pipeline composes in **pixel space**, which bypasses the VAE for the background — but
   only if the mask is actually (near) binary.
3. The mask is not near-binary. It is an energy map, normalized, dilated, γ-lifted
   (`m ← m^0.5`) and Gaussian-feathered; on image 001 the mean α is 0.43 and only 16.5% of
   pixels have α < 0.01. Hence the "background" keeps a 10–40% contribution of the generated,
   blurrier content.
4. The edited region itself is under-detailed relative to the now perfectly sharp background,
   which makes the edit read as "pasted" even when its semantics are correct.

We address (3) and (4). Our contributions are:

- **An analysis of the background leak.** We show that the soft, γ-lifted mask is the dominant
  cause of both the residual blur and the visible seam, and that it invalidates the premise of
  VAE-residual restoration. This diagnosis is simple but, to our knowledge, not documented in
  the one-step editing literature.
- **EBC (Edge-Boundary Compositing).** A near-binary remap of the mask applied *once*, in the
  latent space and in the pixel space alike; a guided-filter step that snaps the composite
  boundary onto real image contours; and a local DC compensation that removes the exposure
  offset in the transition band.
- **HFC (High-Frequency Completion).** One extra UNet probe at low noise, from which only the
  high band is taken and injected inside a gated edit region. The probe *replaces* rather than
  amplifies, so no sharpening gain has to be tuned, and no halo is introduced.
- **Ablations including negative results** on VAE residual recovery (VRR) and unsharp masking.

---

## 2. Related Work

**One-step generation and editing.** Distillation-based one-step models (SDXL-Lightning,
SD-Turbo, SwiftBrush-v2, InstaFlow) trade sampling steps for a strongly non-linear
conditioning. One-step editors either train an inversion network (SwiftEdit) or reuse a
training-free drift difference (FlowEdit, InfEdit), the latter normally requiring many steps.
ChordEdit stabilizes the drift difference by temporal smoothing and needs 1–2 NFEs.

**Background preservation.** PnP-Inversion, MasaCtrl and related methods inject features or
attention maps from an inversion trajectory; they assume a multi-step trajectory. In a
one-step model there is no "k-th step" to inject into, so background preservation must be
solved either in the latent anchor or in the pixel domain. A distinct line or work uses a mask
(DiffEdit, prompt-to-prompt with masks); we deliberately avoid mask *input* — our mask is
derived from the model's own control-field energy.

**Detail and frequency.** FreeU re-weights U-Net backbone/skip features for crisper output,
HiPA learns a high-frequency adapter for one-step models, and FDS performs wavelet-band-
selective editing. All three either modify the network or require training; ours is inference
only and leaves the U-Net untouched. Classic unsharp masking is the trivial alternative but
amplifies existing artefacts, including the VAE blur it is meant to hide.

**Compositing.** Guided filtering (He et al.) is a standard edge-aware operator used in
matting and detail transfer; we use it to regularize the edit mask toward image contours, and
combine it with a band-limited DC correction inspired by gradient-domain blending.

---

## 3. Method

### 3.1 Revisiting the background branch

Let $M\in[0,1]^{H\times W}$ be the soft edit mask produced by the pipeline from the energy of
the Chord field, and $I_s$, $I_e$ the source image and the decoded edit. The upstream
compositing is

$$I_{\text{out}} = \tilde M \odot I_e + (1-\tilde M)\odot I_s,\qquad
\tilde M = M + (1-M)\beta,$$

with $\beta = 0.1$. Two observations follow. First, at $M=0$ the output still contains a
$10\%$ contribution of the generated image, i.e. an exact background copy is impossible by
construction. Second, and more importantly, $M$ itself is far from binary: because γ-lifting is
applied for the *transport* stage (to avoid cutting weak edits), the very same mask is reused
for *compositing*, where the requirement is the opposite. Measured on image 001:

| quantity | value |
|---|---|
| mean $M$ | 0.430 |
| fraction with $M<0.01$ | 16.5% |
| fraction with $M<0.5$ | 57.4% |
| fraction with $M>0.9$ | 14.9% |
| PSNR of a VAE round trip of $I_s$ | 32.16 dB |

The last row bounds every background pixel that travels through the decoder, and explains why
"restore the VAE residual" (VRR) cannot help once a pixel-space copy exists (Sec. 4.4).

### 3.2 Near-binary, contour-aligned alpha (EBC, part 1)

We remap the mask **once**, before it is used for the latent anchor:

$$\alpha = \mathrm{clamp}\!\left(\frac{M-\lambda}{w},\,0,\,1\right),\quad \lambda=0.30,\; w=0.30,$$

and reuse exactly this $\alpha$ (bilinearly upsampled) for pixel compositing, so that the
anchored latent region and the composited region coincide. With this remap the fraction of
pixels with $\alpha=0$ grows from 16.5% to ~40% and the background is restored from the source
rather than from the decoder.

The upsampled $\alpha$ is blocky (64→512, ×8), so we refine it with a guided filter using the
source luminance $L=I_s^{\text{gray}}$ as guide:

$$q_i = \bar a_i L_i + \bar b_i,\qquad
a = \frac{\mathrm{cov}(L,\alpha)}{\mathrm{var}(L)+\epsilon},\; b=\bar\alpha - a\bar L,$$

with radius $r=8$ and $\epsilon=10^{-3}$ (all means are box filters). $q$ keeps the low
frequency of the edit selection but takes its edges from the image, so the composite boundary
follows the object contour instead of an 8-pixel staircase.

### 3.3 Seam DC compensation (EBC, part 2)

Even with a good $\alpha$, the decoded foreground and the copied background differ in exposure.
We estimate a local offset in the transition band $b=4\alpha(1-\alpha)$ and subtract it:

$$\hat o = \frac{\mathrm{box}_{r}\big((I_s-I)\,\odot\,b\big)}{\mathrm{box}_{r}(b)+\varepsilon},
\qquad I \leftarrow I + \kappa\,\hat o \odot b,$$

with $\kappa=0.5$. The band profile is zero in both the interior and the far background, so
the correction cannot move structure; it only removes the fringe.

### 3.4 High-frequency completion (HFC)

After transport and proximal refinement we hold a latent $x_e$ whose structure is correct but
whose texture is under-generated. A single extra UNet evaluation at a **low** noise level
$t_p=0.10$ re-synthesizes detail on top of that structure:

$$z=\bar\alpha_{t_p}x_e+\bar\sigma_{t_p}n,\qquad
x_p=\big(z-\bar\sigma_{t_p}\hat\epsilon(z,t_p,c_{\text{tgt}})\big)/\bar\alpha_{t_p}.$$

We decompose both images into a low band (Gaussian, $\sigma=1$ px) and a high band, and replace
**only** the high band, inside a soft gate:

$$g=\mathrm{clamp}\!\left(\frac{\alpha-(\tau-w_g)}{2w_g},0,1\right),\quad \tau=0.5,\;w_g=0.15,$$

$$I_{\text{out}} = g\odot\big(\mathrm{low}(I_e) + s\cdot \mathrm{high}(I_p)\big) + (1-g)\odot I_e .$$

Because the probe is a *replacement* operator, $s=1$: the model's own detail level is the
target, and no sharpening gain has to be tuned per image. We deliberately do **not** use
unsharp masking; Sec. 4.4 quantifies why.

### 3.5 Cost

The additions are one UNet evaluation (DP) and two VAE decodes (probe, and none for VRR since it
is disabled by default), i.e. NFE 2→3, +0.26 s on a 6 GB laptop GPU; peak memory is unchanged
(3.0 GB, fp16) because every module works on the decoded 512² tensor.

---

## 4. Experiments

### 4.1 Setup

We use the eight editing examples shipped with ChordEdit (portrait/beard, food, horse→unicorn,
fox→dog, cat crown→hat, autumn→winter, dog→wolf, mountain→volcano), 512×512, SD-Turbo in fp16,
seed 42, on an RTX 3060 Laptop (6 GB). Background and foreground are the *strict* regions of
the upstream mask ($\alpha<0.1$ and $\alpha>0.9$), held fixed across variants so that all
methods are scored on the same pixels; the transition band is scored separately by a seam
metric. Image 006 is a whole-scene edit and has no clean background, so it contributes only to
foreground/seam columns.

Metrics: PSNR/SSIM on the background, high-band energy ratio of the result vs the source image
($\mathrm{HF}=1$ means "as crisp as the photograph") on background / foreground / whole image,
Laplacian variance inside the edit region, and a seam metric
$\overline{|\nabla I-\nabla I_s|}$ restricted to the boundary band $4\alpha(1-\alpha)>0.6$.

### 4.2 Main results

| Variant | NFE | PSNR_bg↑ | SSIM_bg↑ | HF_bg→1 | HF_fg↑ | lap_var↑ | seam↓ | time |
|---|---|---|---|---|---|---|---|---|
| Upstream ChordEdit (local mode, β=0.1) | 2 | 50.05 | 0.9979 | 0.944 | 0.996 | 0.00370 | 0.01366 | 0.71 s |
| + near-binary α | 2 | 56.84 | 0.9997 | 1.003 | 1.059 | 0.00387 | 0.01218 | 0.70 s |
| + EBC (guided α + seam DC) | 2 | 56.57 | 0.9997 | 1.002 | 1.059 | 0.00387 | 0.01134 | 0.70 s |
| **+ HFC-DP (full)** | 3 | 56.57 | 0.9997 | 1.002 | **1.202** | 0.00399 | **0.01137** | 0.97 s |
| VRR (VAE residual recovery) | 2 | 44.07 | 0.9922 | 1.072 | 1.059 | 0.00387 | 0.01765 | 0.88 s |
| Unsharp masking | 2 | 43.63 | 0.9922 | 1.124 | 1.412 | 0.00796 | 0.01427 | 0.70 s |

Reading: the remap alone accounts for **+6.8 dB** of background PSNR and recovers the high band
(0.944 → 1.003). EBC leaves fidelity intact while reducing the seam by 17%. HFC-DP adds 13% of
foreground high-band energy at constant background/seam behaviour and one extra NFE.
Unsharp masking reaches an even higher foreground HF (1.41) but at the cost of 7 dB of
background PSNR and a visibly ringed structure; VRR trades 12 dB of background fidelity for a
marginal HF gain.

Figure (`docs/figures/comparison.png`) shows qualitative crops: the upstream background is
slightly soft and the seam is visible on the shoulder contour; EBC removes the fringe;
HFC-DP restores stubble and fur strands; unsharp masking produces halos on the collar.

### 4.3 Ablation: is the probe's gain calibrated?

The probe replaces the high band, so a scalar energy matching factor can be applied. We
compare three settings of $s$:

| matching | HF_fg |
|---|---|
| none ($s=1$, default) | **1.202** |
| global source energy | 1.036 |
| per-region source energy | 0.941 |

Region matching under-sharpens: the edited region's *source* content (smooth skin, satin crown)
is less textured than what the edit introduces (beard, hat), so the ratio is <1 and erases the
new detail. We therefore keep $s=1$ and report matching as an ablation. This is, in our view,
the correct conclusion for a *replacement* operator; energy matching remains appropriate when
the operator is an *amplification*.

### 4.4 Negative result: VAE residual recovery

VRR adds back $I_s-\mathrm{Dec}(\mathrm{Enc}(I_s))$ on the background. It presupposes that the
decoded background equals the VAE round trip of the source, which holds only where the latent
anchor is exact. Since the pixel-space copy already restores the background exactly, VRR can
only *degrade* it: 44.07 dB vs 56.57 dB. Its only advantage is a marginally higher whole-image
HF ratio (0.992 vs 1.002 in the other direction — see table), which is precisely the blur it
reintroduces. We keep it as an option for pipelines that lack a pixel-space copy.

---

## 5. Limitations and next steps

1. **Eight examples only.** No PIE-Bench, no CLIP/LPIPS. The numbers show a consistent trend
   but cannot support SOTA claims. Next: prepare PIE-Bench, add CLIP-Whole/Edited, LPIPS and
   structural distance, and re-run the ladder.
2. **Mask provenance.** Background regions come from the model's energy map, not from human
   annotation; the metrics are therefore self-referential. PIE-Bench GT masks will fix this.
3. **No edit-fidelity metric yet.** A larger α threshold can trivially improve background
   scores at the cost of the edit; the strict-foreground HF and a CLIP-Edited column are the
   intended guard-rails, but they are not yet reported.
4. **Cost.** HFC-DP is +1 NFE (+36% runtime on our card). Adaptive probing (skip when the
   control field is weak) is an obvious next step.
5. **Contour correctness is measured, not guaranteed.** The seam metric drops by 17%, but a
   user study or a boundary-normalized metric would be needed to claim perceptual seamlessness.

---

## 6. References (to be completed in LaTeX)

- ChordEdit: One-Step Low-Energy Transport for Image Editing, arXiv:2602.19083.
- SD-Turbo / SDXL-Lightning / SwiftBrush-v2 / InstaFlow.
- FlowEdit; InfEdit; SwiftEdit; PnP-Inversion; MasaCtrl; DiffEdit.
- FreeU; HiPA; FDS (CVPR 2025); SDEdit.
- He et al., Guided Image Filtering.
- PIE-Bench.

---

## 7. 中文审阅说明（给老师/自己看，不进入正文）

**这篇草稿的核心主张**：一步编辑的"糊"不是模型不会画，而是（i）软 mask 把编辑漏进背景、
（ii）编辑区的细节相对"完美锐利"的背景显得更糊。我们的贡献是把 mask 变成"贴轮廓的近二值
α"（EBC）+ 用一次低噪声探针只替换高频带（HFC），全部免训练、不改网络。

**需要你确认的 5 件事：**

1. **标题/定位**：现在的题目强调"细节补全 + 轮廓贴合"，与老师原论文的"低能量传输"是互补
   关系而非竞争关系。是否要更突出"πars 到背景保护"？
2. **数据**：目前只有 8 张示例图。投稿前必须补 PIE-Bench（700 张）+ CLIP/LPIPS，否则审稿人
   一定会打回。要不要我先把评测脚本接到 PIE-Bench 的格式上？
3. **是否保留 VRR 的负面结论**：我在 §4.4 明确写了 VRR 在这条管线下被像素拷贝取代。好处是
   诚实、能挡住"你们不就是抄了技术报告"的质疑；坏处是和你们之前的 md 结论冲突。建议保留，
   并在正文加一句"VRR 适用于没有像素拷贝的潜空间管线"。
4. **能量匹配的处理**：报告里它是"采纳"，我这里实测有害、改为消融项。这是与 md 的第二处冲突，
   证据在 §4.3 的表格（1.202 / 1.036 / 0.941）。你可以接受哪种口径？
5. **LaTeX 落地**：现在结构已经按 CVPR 风格（Abstract/Intro/Related/Method/Experiments/
   Limitations）写好，公式是 LaTeX 片段，可以直接搬进 cstcloud 的模板。需要我下一步直接生成
   `.tex` + `.bib` 吗？

**还没做的实验**（我可以补）：PIE-Bench 全量、LPIPS、CLIP-Edited、α 阈值敏感性曲线、
DP 的自适应开关（省 1 NFE）、与 FreeU/HiPA 的对比（后者需训练，可只做讨论不跑）。
