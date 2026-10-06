# 视频生成模型系统调研（Video Generation Survey）

> 建立日期：2026-10-06
> 范围：从传统 UNet 扩散到现代 DiT；音画同步；骨架/姿态融合；架构演进；最低硬件门槛与选型。
> 目标：一份可复用、可追溯的系统知识库，而非模型罗列。

## 快速导航

| 章节 | 内容 |
|---|---|
| [01 · 理论基础](docs/01-theory.md) | 扩散模型原理、UNet vs DiT、采样/VAE/CFG、注意力分类 |
| [02 · 音画同步机制](docs/02-audio-video-sync.md) | 联合扩散、条件生成、时间轴对齐、后处理 |
| [03 · 模型全清单](docs/03-model-catalog.md) | 通用视频 DiT / 音视频 / 传统UNet / 对嘴型 / 姿态舞蹈 / 高效边缘 / 世界模型 |
| [04 · 架构对比分析](docs/04-architecture-comparison.md) | 全注意力 vs 线性 vs 混合 vs 稀疏，优缺点 |
| [05 · 姿态骨架融合](docs/05-pose-fusion.md) | 三种融合层级 + 两段式管线 |
| [06 · 最低硬件门槛](docs/06-minimum-hardware.md) | 显存阶梯账（含真实权重字节数） |
| [07 · 结论与选型](docs/07-conclusions.md) | 场景→模型推荐、架构趋势 |
| [08 · 架构演进前景](docs/08-future-directions.md) | 混合注意力、扩散↔AR融合、世界模型、流式、边缘 |
| **[Deep Dive · 五方向论文深挖](docs/deep-dive/)** | 注意力 / AR融合 / 多模态VAE / 世界模型 / 推理加速 |

## 一页速查：当前开源格局（2026-10）

| 模型 | 厂商 | 参数 | 最低显存 | 原生音频 | 架构特征 | 定位 |
|---|---|---:|---:|---|---|---|
| **MiniMax-H3** | MiniMax | 33B | 24GB(offload) | ✅ 32k立体声 | Full Attention 单流，AdaLN占39% | 质量/通用叙事标杆 |
| **LTX-2.3 / 2.5** | Lightricks | 22B | 16GB(NVFP4) | ✅ 单pass | 14B视频+5B音频流，48块共享 | 原生音频+4K+速度 |
| **Wan 2.2** | 阿里通义 | 5B / A14B MoE | 8GB | ❌(有video-to-audio) | MoE 高低噪专家 | 最通用、可商用 |
| **HunyuanVideo 1.5** | 腾讯 | 13B | 14GB(FP8) | ❌ | DiT，电影感运动 | 电影画质 |
| **CogVideoX** | 智谱 | 2B/5B | 16GB | ❌ | 3D 注意力、文本准 | 长叙事/文本 |
| **Mochi 1** | Genmo | 10B | 20GB(FP8) | ❌ | Asymm DiT、运动流畅 | 研究/微调底座 |
| **SANA-Video 2.0** | NVIDIA | 5B | 单卡720p | ❌ | 混合线性+Attention Residual | 高效长序列 |
| **VDN-H3** | UC Berkeley | H3改造 | 数据中心级 | ✅(随H3) | 局部Softmax+双向线性，逐帧VDA | 近无损加速 2.65× |
| **LTX-Video (旧)** | Lightricks | 2B | 8GB | ❌ | 小 DiT | 低显存/长视频 |
| **SVD / AnimateDiff** | Stability | ~1.5B | 8GB | ❌ | UNet | SD 生态 img2vid |
| **MM-Diffusion** | 微软 | 小(双UNet) | ~16GB | ✅联合 | 双耦合UNet | 音乐+舞蹈联合生成 |
| **LatentSync / MuseTalk** | 快手等 | 轻 | 12GB / 实时 | 音频驱动 | SD1.5 UNet | 对嘴型 |

> 说明：显存为"推理可运行"下限（含量化/offload 手段），非全精度驻留；详见第 06 章。

## 证据与可追溯性

- 权重字节数来自 HuggingFace API 实查（MiniMax-H3 / Comfy-Org / LTX-2.5），见第 06 章。
- 性能数字标注来源（论文/项目主页），未实测的明确标记"未真机复验"。
- 本地已拉取代码：`~/codespace/ai/sol-h3-min-research/models/minimax_h3`（NVlabs/Sana @ sol-engine）。
