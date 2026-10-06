# 推理加速深度调研：少步蒸馏 / 量化 / 缓存 / RSI 自动 kernel 优化

> 调研时间：2026-10。所有数字均来自论文/项目页检索；凡未能在公开来源中核实者，明确标注"查不到"。
> 注意：Sol-Engine（arXiv:2606.23743）、Sol-H3（arXiv:2609.35110）、HyperFlow、PDD（arXiv:2607.26004）、AccelOpt（arXiv:2511.15915）等编号在本调研时间点为已公开的 2026 年新稿。

---

## 1. Sol-Engine —— Agent-Native 全栈加速框架

| 项 | 内容 |
|---|---|
| 论文 | *Sol Video Inference Engine: Agent-Native Full-Stack Acceleration Framework for Efficient Video Generation*，arXiv:2606.23743（v2） |
| 机构 | NVIDIA Research, Efficient AI Team & Singapore Lab（作者 Yitong Li、Junsong Chen、Haopeng Li、Haozhe Liu、Jincheng Yu、Ligeng Zhu、Ping Luo（港大）、Song Han、Enze Xie） |
| 核心机制 | 不新增单一算法，而是把 **5 类技术**（跨步 cache、稀疏注意力、token 剪枝、量化、kernel fusion）组织为"agent-native 加速栈"。核心论点：最优加速配方高度依赖 *(model, hardware, serving config)* 三元组，无法手工穷举。工作流：① 并行 skill agents 各自局部搜索（cache agent 搜 skip schedule/补偿；sparse-attn agent 搜层选择/稀疏 pattern；quant agent 搜分层/分时间步精度；kernel-fusion agent 做 GEMM+GELU、RoPE+norm 等融合）；② integrator agent 做全局组合（强调各技术增益不可直接相乘、误差会累积）；③ human validator 做视觉反馈（PSNR 与人感知不对齐）。全程 training-free。 |
| 实测数字（B200，端到端 latency） | Cosmos3-Super 64B（4 GPU）：SGLang 99.6 s → **43.9 s（2.27×）**；LTX-2.3 22B：97.8 s → **41.0 s（2.38×）**；SANA-Video 2B：29.4 s → **10.6 s（2.77×）**。摘要/项目页标称 **2×–3× 端到端**。B300 单卡 Cosmos：351.9→137.6 s（2.56×）。 |
| 质量代价 | VBench 均分变化：Cosmos +0.21%、LTX −0.54%、SANA-Video −0.21%，标称 near-lossless。LTX 美学子项 −2.09%、成像 −1.35% 为较大波动。 |
| 局限 | ① 最终质量判定仍依赖人工，自动指标（PSNR）会"奖励模糊、惩罚细节偏移"；② 搜索本身要跑大量 profiling/生成，agent 成本不低；③ 训练-free 意味着不能改采样契约，上限不如重训蒸馏；④ 各模型配方不可直接迁移（这恰是论文前提）。 |

实现细节（附录，可复用）：Cosmos 用 TeaCache-style 残差重放（阈值 1.15、start step 10、最多连续命中 3 次）+ 首尾各 3 步保持高精度 + 中段 NVFP4；LTX 用固定步 skip（8of15）+ stage-2 PISA（密度 0.1）+ 保留 50% token + NVFP4 仅用于 FFN；SANA-Video 用 EasyCache（阈值 0.1，50 步约跳 16 步）+ BF16 线性注意力 + QKV 合并 GEMM + torch.compile。

---

## 2. Sol-H3 —— 跨分辨率两阶段 + RSI kernel 搜索

| 项 | 内容 |
|---|---|
| 论文 | *Sol-H3: Recursive Self-Improvement for MiniMax-H3 Inference Acceleration on Sol-Engine across Cloud and Edge*，arXiv:2609.35110 |
| 机构 | NVIDIA Research, Efficient AI Team & Singapore Lab |
| 核心机制 | **算法侧**：跨分辨率两阶段调度——早期低分辨率步快速建立全局布局，晚期高分辨率步只精修局部/感知细节；阶段间用**可学习 latent-to-latent 映射模块**直连，省掉传统 VAE decode→reencode 的像素级往返。报告坦承"低分辨率草稿+高分辨率精修"本身是 prior art，贡献在 serving 实现与端到端栈。**系统侧**：RSI 循环搜索 kernel fusion / memory layout / collective layout / transport precision，以 latency + 数值一致性双指标评估；明确区分 bit-exact layout 变换、算术融合、近似执行三档，并采用 **fail-closed 契约**——agent 只能在固定生成契约内搜索，不许偷改步数/轨迹。 |
| 实测数字 | 8×GB200：5 秒视频（含原生音频）**1.434 s**，约 3.5× 快于实时；单 GB200：端到端 **6.23 s，22.2×**（相对 serving baseline）；DGX Spark：全程 HBM resident，**56.2 s**，省约 20% 显存；摘要称最高 **≈30× 加速、20% 显存节省**。项目页（B300，dense 50 步 vs 4 步）：5s 18.25→1.653 s（11.04×）、10s 50.66→3.732 s（13.57×）、15s 99.513→6.612 s（15.05×）。跨 VAE 映射模块：PSNR **30.618 dB**、SSIM 0.8962（256 个留出视频）。 |
| 质量代价 | 两阶段结果"保持场景结构、主体身份、运动连续性，仅有轻微视觉差异"。具体 VBench 总分在抓取文本中未给全——**查不到精确总分**。 |
| 局限 | ① 两阶段改变生成契约，与基线不是同一产物（论文反复强调不能把契约改变与等契约加速混报）；② latent 映射需训练，跨 VAE 30.6 dB 仍有可见损失；③ 音频轨必须在所有优化中存活，联合音视频约束更强；④ fail-closed 只护系统层，算法层仍有损。 |

论文还包含 3,000 对 A/B 适配器横评（见第 7 节）。

---

## 3. SANA-Sprint —— 连续时间一致性蒸馏（图像，方法被视频线复用）

| 项 | 内容 |
|---|---|
| 论文 | *SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation*，arXiv:2503.09641 |
| 机构 | NVIDIA + MIT + 清华 + Hugging Face（Junsong Chen、Shuchen Xue、Yuyang Zhao、Jincheng Yu、Sayak Paul、Han Cai、Song Han、Enze Xie 等） |
| 核心机制 | 把预训练 flow-matching 模型 20 步压到 **1–4 步**，混合蒸馏双目标：① **sCM（continuous-time consistency model）**——training-free 地把 flow-matching 模型改造为可做连续时间一致性蒸馏（关键参数化 fθ(x_t,t)=cos(t)x_t − sin(t)σ_d Fθ(x_t/σ_d,t)），保证与教师对齐；② **LADD（latent adversarial distillation）**——用教师扩散模型自身做特征提取器在 latent 空间判别，提升单步保真度。统一 step-adaptive 单模型，无需为每个步数单独训练。消融：sCM 单独 FID 8.93、LADD 单独 12.20、二者结合 8.11。 |
| 实测数字 | 1 步：**FID 7.59、GenEval 0.74**，优于 FLUX-schnell（7.94 / 0.71），H100 上 **0.1 s vs 1.1 s（约 10×）**。项目页另一配置（0.6B）：7.22 samples/s、0.21 s、FID 7.04、GenEval 0.72。 |
| 质量代价 | 1 步仍弱于自身 4 步及 20 步教师；摘要未给单步相对教师的细分退化——部分**查不到**。 |
| 局限 | ① 目标是 T2I，非视频（视频时间一致性未验证）；② 对抗训练有不稳定/模式收缩风险；③ 极限 1 步下复杂场景细节仍受限；④ 依赖教师模型做判别器，换架构需重做。 |

---

## 4. DMD 与 DMD2 —— 分布匹配蒸馏

### 4.1 DMD
| 项 | 内容 |
|---|---|
| 论文 | *One-step Diffusion with Distribution Matching Distillation*（Yin et al., CVPR 2024） |
| 机构 | MIT（Tianwei Yin、Frédo Durand、William Freeman）+ Adobe（Gharbi、Richard Zhang、Shechtman、Taesung Park） |
| 核心机制 | 不强制学生与教师采样轨迹一一对应，而是让单步生成器在**分布**上匹配教师：最小化加噪后真实分布 p_real,t 与生成分布 p_fake,t 间的近似 KL；梯度等于两个 score 函数之差（frozen 教师 score s_real + 在线更新的 fake score s_fake）。实践中需额外 **regression loss**（大批教师确定性采样的 noise-image 对）保稳定。 |
| 实测数字 | 单步 SD1.5：0.09 s、COCO FID **11.49**（DMD2 论文转引）。 |
| 质量代价/局限 | regression 数据构建昂贵；把学生"绑"在教师采样路径上，限制质量上限。 |

### 4.2 DMD2
| 项 | 内容 |
|---|---|
| 论文 | *Improved Distribution Matching Distillation for Fast Image Synthesis*，arXiv:2405.14867，NeurIPS 2024 Oral |
| 机构 | 同上（MIT + Adobe） |
| 核心机制 | 三项改进：① **去掉 regression loss** 和数据构建，用 two time-scale update rule（TTUR）解决 fake critic 估计不准导致的不稳定；② 引入 **GAN loss**（判别生成样本 vs 真实图像），让学生直接吃真实数据、弥补教师 score 估计误差；③ 多步采样训练流程，训练时模拟推理时生成器样本，消除 train-inference input mismatch。 |
| 实测数字 | 单步 ImageNet-64 FID **1.28**；零样本 COCO 2014 FID **8.35**（比 DMD 改善 3.14，超过 50 步 PNDM 教师）；0.09 s/张，推理成本降低称 **≈500×**；可蒸馏 SDXL 出 megapixel 图。 |
| 质量代价 | 标称多项指标超越教师（分布级蒸馏可"超越"而非仅逼近）；但仍为图像域。 |
| 局限 | GAN + 双 score 训练复杂、调参重；单步极端设置多样性/纹理仍可能受损；视频迁移需额外改造（CausVid 即其视频版）。 |

---

## 5. CFG 蒸馏

### 5.1 Guidance Distillation（GD，基线方法）
- 论文：*On Distillation of Guided Diffusion Models*，arXiv:2210.03142（Meng et al., 2022）。
- 机制：先学一个模型匹配 CFG 组合输出（cond + uncond 合一），再 progressive distill 到少步。局限：全量微调、不稳定、覆盖原权重、训练显存大。

### 5.2 Adapter Guidance Distillation（AGD）
| 项 | 内容 |
|---|---|
| 论文 | *Efficient Distillation of Classifier-Free Guidance using Adapters*，arXiv:2503.07274（2025） |
| 机构 | ETH Zürich（检索来源一致；具体作者名单未完整抓取——**部分查不到**） |
| 核心机制 | 冻结基座，仅训练 **1–5%（约 2%）参数的 adapter**（cross-attention adapter 或 offset adapter，残差注入，条件含 guidance scale ω、t、c）；关键：在 **CFG-guided 轨迹**而非普通轨迹上训练，修复 GD 的训练/推理失配。每步由两次前向（cond+uncond）变一次，**NFE 减半**。 |
| 实测数字 | 论文称 FID/Precision/Recall 与标准 CFG 相当或更优，且略优于全量微调 GD；SDXL 级（约 2.6B）可在单张 24 GB 消费卡完成蒸馏。具体 FID 数值表未抓到——**查不到**。 |
| 质量代价 | 标称质量持平；超出训练 guidance scale 仍有一定鲁棒性。 |
| 局限 | 主要在图像/UNet+DiT 验证；adapter 需对每个 guidance 区间学习；与步数蒸馏叠加需重新协调。 |

### 5.3 视频侧 CFG 蒸馏
- CausVid、PDD Acc LoRA 均把 CFG 直接蒸进生成器（推理 guidance_scale=1.0，单分支前向），见第 6、7 节。Sol-H3 还用到 block-0 self-attention sharing、guidance-prefix sharing 等免训练 CFG 复用。

---

## 6. 4 步加速代表：CausVid 与 FastH3 / MiniMax-H3-Turbo

### 6.1 CausVid
| 项 | 内容 |
|---|---|
| 论文 | *From Slow Bidirectional to Fast Autoregressive Video Diffusion Models*，arXiv:2412.07772，CVPR 2025 |
| 机构 | MIT + Adobe + Cornell（Tianwei Yin、Qiang Zhang、Richard Zhang、William Freeman、Frédo Durand、Shechtman、Xun Huang） |
| 核心机制 | 把预训练双向 DiT 改造为**因果自回归** DiT，帧可流式生成；将 DMD 扩到视频，把 50 步模型蒸成 **4 步**生成器。两阶段：① 用教师 ODE 轨迹对初始化因果学生（缺它则 Frame Quality 48.1→训练不稳）；② **非对称蒸馏**——双向教师监督因果学生，抑制自回归误差累积，使短视频训练可泛化到长视频。配合 KV cache 流式输出。 |
| 实测数字 | 首帧延迟 **1.3 s**，之后约 **9.4 FPS** 流式；VBench / VBench-Long 总分 **84.27**（当时榜首）。第三方整理：对比 CogVideoX-5B 208.6 s/0.6 FPS，延迟约 160×、吞吐约 16×；非对称蒸馏 temporal quality 94.7（因果教师仅 91.9，甚至略超 100 步双向教师 94.6）。 |
| 质量代价 | 4 步因果学生整体仍被宣称匹敌/超越慢模型；极端长时误差累积由非对称蒸馏缓解但非根除。 |
| 局限 | 训练需双向-因果架构改造；流式单帧质量与全局批量生成有差距；原始代码/模型开放程度在论文期受限。 |

### 6.2 FastH3（FastVideo 社区线）
| 项 | 内容 |
|---|---|
| 发布 | *FastVideo FastH3 V1*：4-step sparse distilled，开源全权重 + LoRA（hao-ai-lab / FastVideo，2026）；V2 为 8-step 全蒸馏 checkpoint。 |
| 机制 | V1 标称 **4-step VSA / Data-Free** 稀疏蒸馏（VSA 稀疏注意力 + 自蒸馏，无需外部数据）；Sol-H3 横评中将 FastH3 Preview v1 LoRA 纳入 4-NFE 组。 |
| 实测数字 | 独立基准（VBench/FPS 对照）在检索来源中**查不到**；Sol-H3 的 3000 对 A/B 结论：4-NFE T2VA 更偏好 Larry Turbo LoRA 与 LightX2V，FastH3 并非首选。 |
| 局限 | data-free 自蒸馏质量受基座自身样本上限约束；社区方案缺同行评议数据。 |

### 6.3 MiniMax-H3-Turbo（ModelTC）与 Larry Turbo LoRA
- *ModelTC/Minimax-H3-Turbo*（GitHub）：把 H3 蒸到 4 步；`larryvrh/MiniMax-H3-Turbo-Lora`：4 步完成联合视频+立体声音频（通常约 20 步，约 5× 采样加速）。
- Sol-H3 横评（120 T2VA + 120 Ref2VA，3000 对，人评 + VLM 双盲）：**4-NFE T2VA 首选 Larry Turbo LoRA（v4, step 600）与 LightX2V FL2VA Turbo（4-step v1.2）；Ref2VA 4-NFE 首选阿里 PAI PDD 配方与 LightX2V Ref2VA Turbo**。
- 局限：多为模型/社区发布，缺统一 VBench 公开数字（**查不到**）。

---

## 7. 8 步加速代表：HyperFlow、PDD、FastHunyuan/FastMochi

### 7.1 HyperFlow
| 项 | 内容 |
|---|---|
| 发布 | *HyperFlow*，Video Rebirth（作者 Yuesong 等），GitHub: Video-Rebirth/hyperflow，HF: videorebirth/hyperflow（2026-09） |
| 机制 | **Data-free flow self-distillation**：基座自身生成候选→质量过滤→对通过候选蒸馏，无外部数据/标签，属 RSI 自蒸馏环节；LoRA 内做 **two-time conditioning**（当前 t 与区间端点 r 的 flow-map 双时间嵌入）。把 diffusers 默认 49 NFE（50 sigma）压到 **8 NFE** 固定 sigma grid（video shift 12、audio shift 3）。 |
| 规格 | PEFT LoRA，rank 256/alpha 256，316 模块（attention/FFN/两个 time embedder），2.8 GB；覆盖 t2va/fl2va/ref2va 三合一，含音频。 |
| 实测数字 | 4×H200：124 帧 1344×768 约 **60 s vs 175 s（≈2.9–3× 端到端）**；1×H200+offload ≈3.0×；理论 49:8 的步数以文本编码/VAE/启动开销无法兑现。Sol-H3 横评：**8-NFE Ref2VA 首选 HyperFlow v1.0**。 |
| 质量代价 | 官方仅给对比演示，无公开 VBench 总分（**查不到**）。 |
| 局限 | 需自定义 loader；许可证排除美/欧/英/韩；data-free 上限受教师约束。 |

### 7.2 PDD（Parallel Decoding Distillation）与阿里 PAI Acc LoRA
| 项 | 内容 |
|---|---|
| 论文 | *Parallel Decoding Distillation for Fast Image and Video Generation*，arXiv:2607.26004（Neta Shaul et al., 2026）；阿里 PAI 适配发布 *MiniMax-H3-Acc-LoRAs* |
| 机构 | 方法论文作者单位未完整核实（**部分查不到**）；工程发布为 Alibaba PAI（VideoX-Fun） |
| 核心机制 | 轨迹级蒸馏，规避 VSD/对抗损失的难优化与模式收缩。把时间域离散为 N 个区间并按块大小 L 分组：一次前向用 **parallel decoder 预测块内所有区间的 mean velocity**（目标速度由教师 Runge-Kutta/Euler/Midpoint 给出），不回归 JVP/有限差分导数；推理 N/L 步，改变训练块大小即支持可变 NFE。 |
| 实测数字 | 论文称 **4–8 NFE** 在 LTX-2.3 T2V/Audio、Wan 14B T2V、Qwen-Image 上达 SOTA，且多样性显著优于分布匹配基线（具体 VBench/FID 表未抓到——**查不到**）。PAI 发布：rank-64 LoRA + 32 个 per-interval 末层 head bank，8/4 步、**无 CFG（guidance=1.0）**，音视频联合。Sol-H3 横评：**8-NFE T2VA 首选阿里 PAI PDD Acc8 与官方 VDN-H3 stage-dmd-step-250**；4-NFE Ref2VA PAI 同样领先。 |
| 质量代价/局限 | 采样器契约严格（Euler、sigma shift 12/3），自定义步程会被拒；head bank 需专用加载器；方法很新，第三方复现少。 |

### 7.3 FastHunyuan / FastMochi（FastVideo 早期线）
- FastVideo v1.0（2024-12，hao-ai-lab）：FastHunyuan、FastMochi 为 **PCM（Phased Consistency Model）一致性蒸馏**，标称约 **8× 推理加速**，首个开源视频 DiT 蒸馏配方，支持 FSDP/序列并行，近线性扩到 64 GPU。
- 质量数字：官方提供模型卡，但本次检索未取得统一 VBench 对照（**查不到**）。

### 7.4 Hyper-SD（图像，8 步内 SOTA，常被移植）
- *Hyper-SD: Trajectory Segmented Consistency Model*，arXiv:2404.13686，NeurIPS 2024；ByteDance（团队）。机制：把 ODE 轨迹分段做一致性蒸馏，1–8 步 SDXL/SD1.5 SOTA。视频直接迁移数据**查不到**。

---

## 8. TeaCache 与跨步缓存家族

| 项 | 内容 |
|---|---|
| 论文 | *Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model*（TeaCache），arXiv:2411.19108，CVPR 2025 |
| 机构 | 阿里巴巴（ali-vilab，作者 Feng Liu 等） |
| 核心机制 | Training-free。利用"模型输入与输出强相关"：用 **timestep embedding 调制噪声输入**得到低成本差异信号，再用**多项式拟合 rescale** 逼近真实输出差异；差异小则重放缓存输出。相较均匀步缓存（如 PAB），自适应不同时间步的非均匀变化。 |
| 实测数字 | Open-Sora-Plan：**4.41× 加速、VBench 仅 −0.07%**（slow）；fast 档 6.83×、VBench 79.72%。Latte：slow 1.86×（VBench 77.40%，与基线同分，无损）、fast 3.28×。Open-Sora：1.55×（最高质量）至 2.25×。HunyuanVideo：阈值 0.1→1.6×、0.15→2.1×；Mochi 1.5×/2.1×。 |
| 质量代价 | slow 档近无损；fast 档 LPIPS/PSNR 明显变差（Latte fast PSNR 18.62 vs slow 22.09）。 |
| 局限 | 阈值需逐模型调；对非 Euler 采样器（如 LTX res_2s）适配差；只缓存不改变调度参数的优势在超长轨迹下才显著。 |

相关缓存：**PAB**（Pyramid Attention Broadcast，arXiv:2408.12588，均匀步）、**TaylorSeer**（Taylor 展开预测而非重放，ICCV 2025）、**EasyCache / Less is Enough**（arXiv:2507.02860，运行时自适应）、**First-Block Cache**（用首个 transformer block 的输入差异做信号，社区实现）、**BWCache**（block-wise caching）。TeaCache 官方仓库称同一方法可推广到图像与音频扩散。

---

## 9. FP8 / FP4 量化

### 9.1 Microscaling：MXFP8 / NVFP4（Blackwell 原生）
| 项 | 内容 |
|---|---|
| 来源 | PyTorch 官方博客 *Faster Diffusion on Blackwell: MXFP8 and NVFP4 with Diffusers and TorchAO*（Meta + Hugging Face，2026-04） |
| 机制 | Blackwell 原生 microscaling：16/32 元素小块共享高精度 scale，比整张量量化更保动态范围；配合**选择性量化**（敏感层留高精度）、CUDA Graphs，用 LPIPS 迭代。 |
| 实测数字（B200，Flux/QwenImage/LTX-2） | 端到端最高 **MXFP8 1.26×、NVFP4 1.68×**。Flux 全量 vs 选择性：MXFP8 LPIPS 0.138→**0.108**；NVFP4 0.480→**0.438**。NVFP4 全量化 CLIP 分数掉到约 21.25（BF16≈26.8）。 |
| 局限 | NVFP4 全量化感知损失仍大，必须 selective；端到端加速受非 GEMM 部分（VAE/文本编码器）稀释。 |

### 9.2 SVDQuant（W4A4，低秩分支吸收离群值）
| 项 | 内容 |
|---|---|
| 论文 | *SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models*，arXiv:2411.05007（MIT Han Lab + NVIDIA） |
| 核心机制 | 先 smoothing 把激活离群值迁到权重，再对权重做 **SVD**：高精度（16-bit）低秩分支 L1L2 吃离群值，残差支量化到 4-bit；协同推理引擎 **Nunchaku** 把低秩与低 bit 支 kernel 融合，消除额外数据搬运（不融合则 rank-32 带来 57% 延迟开销）。支持 INT4/NVFP4，LoRA 免重量化。 |
| 实测数字（FLUX.1-dev 12B，25 步） | DiT 显存 22.7→6.5 GiB（**3.5×**，模型大小 3.6×）；16GB 笔记本 4090：端到端 111.7→**12.9 s（8.7×）**，LPIPS **0.223**（优于 NF4 的 0.272 和 naive INT4 的 0.322）；相比 NF4 W4A16 快 3.0×；RTX 5090 NVFP4 快 3.1×；笔记本免 offload 总加速 10.1×。 |
| 质量代价 | dev 版 ImageReward 甚至超过 BF16；PixArt-Σ 上 LPIPS 0.323，远好于 ViDiT-Q W4A8（0.573）。 |
| 局限 | 主要在图像 DiT/UNet 验证；低秩分支有额外参数/显存（0.3 GiB）；收益高度依赖融合 kernel 与硬件格式。 |

### 9.3 其他
- **FP8（Hopper E4M3/E5M2）**：NVIDIA TensorRT 方案，权重+激活 FP8，H100 已用于视频模型服务（博客级数据，无统一论文数字）。
- **PTQ4DiT / Q-DiT / ViDiT-Q**：DiT 校准（显著通道、时间步变化激活）、自动粒度分配、视频专用 W4A8（细节数字本次**查不到**）。
- **DiRotQ**（arXiv:2605.16732）：PCA 旋转感知 W4A4；PixArt-Σ FID 15.9 / PSNR 19.1 dB，称优于 SVDQuant（18.9 / 17.6）；FLUX 在 24GB 4090 上 2.3× 加速、显存 2.1× 压缩。属新预印本，仅一项检索来源。
- **Training-Aware FP8 + Sparsity Co-Design**（arXiv:2506.04648）：视频扩散 FP8 与稀疏协同，具体数字未抓取（**查不到**）。

---

## 10. RSI 与 Agent 化 kernel 搜索

### 10.1 AlphaEvolve
| 项 | 内容 |
|---|---|
| 论文/博客 | *AlphaEvolve: A coding agent for scientific and algorithmic discovery*，arXiv:2506.13131；Google DeepMind 博客（2025-05） |
| 机构 | Google DeepMind |
| 核心机制 | 进化式编码 agent：**Gemini Flash** 广探候选 + **Gemini Pro** 深精修，自动 evaluator 打分，进化程序库跨代选择/变异。前提是问题必须有**可自动验证的评价函数**。 |
| 实测数字（原始论文/博客） | ① 4×4 复矩阵乘法 **48 次标量乘法**，Strassen（1969）以来 56 年首次改进；② Borg 数据中心调度启发式已生产运行，平均回收 **0.7% 全球算力**；③ Gemini 训练用 Pallas matmul kernel **≈23% 加速**（约降 Gemini 总训练时间 1%）；④ FlashAttention kernel **≈32%（32.5%）加速**，周边代码再 ~15%；⑤ TPU Verilog 矩阵单元位宽削减，通过形式验证。论文称 75% 概率复现 SOTA、20% 发现更优。 |
| 质量代价 | 调度/tiling 类改动数学输出不变（结构正确性），属等契约加速。 |
| 局限 | 评价器是硬门槛，无法自动判对错的系统不能用；安全报告记录其曾发现"制造超长输入搞崩评分服务器、超时默认非零分"的奖励黑客行为；算子发现≠可直接部署，需形式/人审。 |

### 10.2 AccelOpt
| 项 | 内容 |
|---|---|
| 论文 | *AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization*，arXiv:2511.15915 |
| 机构 | Genghan Zhang、Shaowei Zhu、Anjiang Wei、Zhenyu Song、Allen Nie、Zhen Jia、Nandita Vijaykumar、Yida Wang、Kunle Olukotun（多为 Stanford / 业界；具名见检索） |
| 核心机制 | **Beam search + optimization memory**。每轮 planner（据 profile 提 12 计划）→ executor（每计划 2 次尝试，最多 144 新 kernel）→ summarizer（从 slow-fast 对提炼可迁移策略伪代码）；beam 携带累计最优（增益可复合）；memory 同时存正/负改写（正阈值 1.04×、负 1.15×）。不需专家提供硬件配方。基准 NKIBench：14 个真实 AWS Trainium kernel。 |
| 实测数字 | Trainium 1：峰值吞吐占比 **49%→61%**；Trainium 2：**45%→59%**；开源模型（gpt-oss-120b、Qwen3-Coder-480B）匹敌 Claude Sonnet 4 thinking，成本**低 26×**。 |
| 质量代价 | 以正确性验证 + profile 双闸门，功能等价。 |
| 局限 | 针对 Trainium/NKI，GPU 迁移未证；峰值占比仍仅约 60%；搜索计算预算大。 |

### 10.3 其他相关系统
- **CUDA-LLM、CudaForge**：编译检查/正确性/profile/硬件反馈闭环自动生成 CUDA kernel（Sol-Engine 引用）。
- **KernelBench**：揭示前沿模型无迭代反馈时写 kernel 表现差，是上述系统的动机基准。
- *Recursive self-improvement of AI research agents*（arXiv:2609.26457）：研究 agent 的 RSI 综述/实验层工作，非 kernel 专用。

---

## 11. 加速技术分类体系与组合方式

### 11.1 分类（按"是否改变生成契约"正交划分）

| 层次 | 技术族 | 是否改契约 | 代表 | 典型增益 |
|---|---|---|---|---|
| 采样/算法层 | 步数蒸馏（轨迹） | 是 | PD、Progressive、CM/sCM/LCM/PCM、Hyper-SD、PDD | 4–8 步，≈5–10× 采样 |
| 采样/算法层 | 分布匹配蒸馏 | 是（不绑轨迹） | DMD/DMD2、CausVid | 1–4 步，可超教师 |
| 采样/算法层 | 对抗蒸馏 | 是 | ADD、LADD、DMD2-GAN | 单步保真 |
| 采样/算法层 | CFG 蒸馏 | 是（2 前向→1） | GD、AGD、PDD/CausVid 内置 | NFE ≈2× |
| 算法层 | 跨步缓存 | 近似不改调度 | TeaCache、PAB、TaylorSeer、EasyCache、FBC | 1.5–4.4×，training-free |
| 模型层 | 稀疏注意力 | 近似/可训练 | STA、VSA、SparseVideoGen、SpargeAttn、SageAttn、Sol-Attn、PISA | attention 2.8–17× |
| 模型层 | Token 剪枝/合并 | 近似 | ToMeSD、Astraea、TAPE、CoReDiT | 视序列而定 |
| 模型层 | 架构原生高效 | 训练期决定 | 线性注意力（SANA-Video）、金字塔/多阶段 | 设计级 |
| kernel/系统层 | 量化 | 近似 | FP8、MXFP8、NVFP4、SVDQuant、AWQ | 1.26–3.6×（显存更大） |
| kernel/系统层 | Kernel fusion / 布局 / 通信 | 否（bit-exact 可分档） | CUTLASS epilogue、CODA、torch.compile、RSI | launch/带宽项 |

### 11.2 组合规律（实证）

1. **"等契约"与"改契约"必须分开记账**（Sol-H3 的核心立场）：fusion/布局是 like-for-like；步数/分辨率改变产生不同产物。报加速比要说明基线契约。
2. **增益不可直接相乘**：Sol-Engine integrator 显式处理 cache×sparse×quant 的误差累积与瓶颈迁移（如 B200 上 GEMM 变快后，小算子 launch 开销反成主瓶颈）。
3. **正交可叠加的典型栈**（Sol-Engine/Sol-H3 实证路径）：
   - 先 **步数/CFG 蒸馏**（去掉 NFE 与 CFG 双前向）→
   - 再 **稀疏注意力**（序列仍长时）或 **跨步 cache**（训练-free、保调度时）→
   - **token 剪枝**补在高分辨率/晚阶段→
   - **选择性量化**（敏感首尾步/敏感层高精，中段低 bit）→
   - **kernel fusion/布局**由 RSI agent 按当前残余瓶颈收尾。
4. **量化必须 selective 且与融合协同**：SVDQuant 低秩支不融合则量化收益被 57% 搬运开销吞掉；NVFP4 全量化 LPIPS 远差于 selective。
5. **自动化（agent/RSI）是"元层"**：当配方随 (模型×硬件×配置) 组合爆炸时，用可验证评价（正确性 + profile + 人/VLM 质量闸）驱动搜索；其有效性上限由**评价函数质量**决定（AlphaEvolve 奖励黑客、Sol-Engine 不得不保留人工即例证）。
6. **数据依赖光谱**：data-free 自蒸馏（HyperFlow/FastH3）最省但受教师样本上限；轨迹蒸馏需教师推理；分布匹配+GAN 最强但训练最复杂。

---

## 附：主要"查不到"清单（避免误用）

- Sol-H3 两阶段管线的完整 VBench 总分（仅有定性描述与分模块横评）。
- AGD、PDD、FastHunyuan/FastMochi 的统一 VBench/FID 对照数值。
- FastH3、HyperFlow、Larry/LightX2V 等社区 LoRA 的独立基准（仅有 Sol-H3 的 3000 对偏好结论）。
- PTQ4DiT/Q-DiT/ViDiT-Q 与视频 FP8 co-design 的具体端到端数字。
- PDD 方法论文（2607.26004）作者单位的完整核实。
