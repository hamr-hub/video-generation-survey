# 01 · 理论基础

## 1.1 扩散模型（Diffusion）在做什么

核心思想：学会把"纯噪声"逐步去噪成"数据"（图像/视频/音频）。

- **前向过程**：往真实数据 x₀ 上按固定噪声表（schedule）逐步加噪，T 步后近似纯高斯噪声。
- **反向过程**：训练一个神经网络 εθ 预测每一步的噪声（或预测 x₀、或预测速度 v），从纯噪声开始反复去噪，还原数据。
- **推理 = 迭代采样**：默认要跑几十步（如 50 步），每步一次网络前向，这是生成慢的根本原因。

视频扩散：在 2D 图像基础上加**时间维**，网络要同时建模空间（画面）和时间（运动）。

## 1.2 两大主干：UNet vs DiT

### UNet（传统，2022–2023 主流）

- 卷积编码器-解码器，跳跃连接，空间下采样/上采样。
- 视频 UNet：在 ResNet 块中插入**时间卷积/时间注意力**（如 ModelScope、SVD、AnimateDiff）。
- 优点：成熟、小、卷积对硬件友好、SD 生态完善。
- 缺点：规模化收益弱，长程/跨镜头一致性差，多模态统一困难。

### DiT（Diffusion Transformer，2024 起主流）

- 把 backbone 换成 Transformer，token 化 latent + 自注意力 + FFN。
- 代表：Sora 带火，随后 HunyuanVideo / Wan / CogVideoX / Mochi / H3 全是 DiT。
- 优点：Scaling law 好（越大越强）、长程依赖强、天然适合多模态 token 统一。
- 缺点：注意力对序列长度二次复杂度，显存/算力重。

**结论**：现代高质量视频生成几乎已全面转向 DiT；UNet 保留在轻量、边缘、对嘴型、SD 生态。

## 1.3 关键组件

| 组件 | 作用 |
|---|---|
| **VAE** | 像素空间↔压缩 latent。视频 VAE 在空间和时间上都压缩，大幅降低 token 数。例：H3 视频 16× 空间压缩；LTX Conv VAE 32×。 |
| **Patchify** | DiT 把 latent 切成 token 块（如 1×2×2）。 |
| **Text Encoder** | 把 prompt 编码为条件：T5/UMT5、Qwen3-VL（H3）、CLIP 等。 |
| **AdaLN** | Adaptive Layer Norm，用时间步/条件调制每个 Transformer 块。H3 的 AdaLN 约 13B 参数（占 39%）。 |
| **CFG** | Classifier-Free Guidance：条件+无条件两路前向，提升文本遵循，代价约 2×。可蒸馏成单路（H3 用 CFG 蒸馏）。 |
| **Scheduler** | 采样轨迹：DDIM/DPM/Flow Matching/Shift 等。步数越少越快但易损质。 |

## 1.4 注意力分类（架构演进核心）

| 类型 | 复杂度 | 质量 | 代表 |
|---|---|---|---|
| **Full Softmax** | O(N²) | 最强，长程身份/布局稳 | MiniMax-H3 |
| **Sparse（稀疏）** | 近似 O(N·k) | 高，训练免 | Sol-Attn、SageAttention |
| **Linear（线性）** | O(N) | 弱，长程身份易漂移 | SANA-Video（纯线性） |
| **Hybrid（混合）** | 介于之间 | 近无损 | VDN、SANA-Video 2.0、SLA |
| **SSM / RNN 类** | O(N) | 中长程、状态压缩 | Gated DeltaNet、RWKV、Mamba |

详见第 04 章。

## 1.5 加速的两条路线

1. **算法加速（有损）**：减步数、蒸馏（一致性/DMD）、缓存（TeaCache/FirstBlock）、稀疏注意力、跨分辨率两阶段。
2. **系统加速（数学等价）**：kernel 融合、内存布局重排、通信调度、量化、并行 VAE。

Sol-H3 = 两类叠加（跨分辨率两阶段 + RSI 自动 kernel 搜索）。
