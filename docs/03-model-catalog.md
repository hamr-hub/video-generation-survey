# 03 · 模型全清单

> 参数/显存为 2026-10 公开值；显存是"可运行"下限（量化/offload），非全精度。带 ⚠️ 表示未真机复验。

## 3.1 通用视频生成 DiT（无原生音频）

| 模型 | 厂商 | 参数 | 最低显存 | 分辨率/时长 | 许可证 | 优点 | 缺点 |
|---|---|---:|---:|---|---|---|---|
| **Wan 2.2** | 阿里通义 | TI2V-5B / A14B MoE | 8GB / 12GB | 720p，长短 | Apache 2.0 | 最通用、可商用、MoE高低噪专家、生态好 | 无原生音频；偶尔过饱和 |
| **HunyuanVideo 1.5** | 腾讯 | 13B | 14GB(FP8) | 1080p，5s | 社区许可 | 电影感运动、综合画质强 | 慢、吃显存、无音频 |
| **CogVideoX** | 智谱 | 2B / 5B | 16GB | 720p/1440×960，6s | Apache(2B) | 文本遵循准、长叙事、中文 | 动态较弱 |
| **Mochi 1** | Genmo | 10B | 20GB(FP8) | 480p，5.4s | Apache 2.0 | 运动流畅/物理、训练代码开放 | 分辨率上限低 |
| **LTX-Video** | Lightricks | 2B | 8GB | 最长2min | Apache 2.0 | 快、低显存、长视频 | 质量有差距 |
| **SVD / SVD-XT** | Stability | ~1.5B | 8GB | 576×1024，14/25帧 | 社区许可 | SD生态 img2vid、AnimateDiff | 无文本、短视频 |

## 3.2 原生音视频（单 pass 联合生成）

| 模型 | 厂商 | 参数 | 最低显存 | 音频规格 | 架构 | 优点 | 缺点 |
|---|---|---:|---:|---|---|---|---|
| **MiniMax-H3** | MiniMax | 33B | 24GB(offload) | 32kHz 立体声 | Full Attention 单流，50层，AdaLN 13B | 质量/多镜头对白/音画同步标杆，可上2K | 极重、49次forward |
| **LTX-2.3** | Lightricks | 22B | 16GB(NVFP4) | 同步 | 14B视频+5B音频，48共享块 | 原生音频+4K@50fps+速度 | 商用有营收门槛 |
| **LTX-2.5** | Lightricks | 22B | 类似 | 同步 | Conv Video VAE + dev 22B | 精修/蒸馏生态完善（Sol-H3 用） | 大 |
| **MM-Diffusion** | 微软 | 小 | ~16GB ⚠️ | 联合 | 双耦合 UNet | 首个联合音视频、轻、音乐舞蹈 | 非通用叙事、单人场景 |

## 3.3 传统 UNet / 音频驱动对嘴型

| 模型 | 输入 | 最低显存 | 架构 | 强项 |
|---|---|---:|---|---|
| **LatentSync (1.5)** | 视频+目标语音 | ~12GB（有8G方案⚠️） | SD1.5 UNet + Whisper | 唇形同步最优、纯PyTorch |
| **MuseTalk** | 人像+语音 | 实时级 | nehead + UNet | 实时对嘴 |
| **EchoMimic v1/v2** | 音频 | 消费级 | UNet | 半身、手+脸 |
| **Hallo / Hallo2** | 音频 | 消费级 | UNet | 肖像长视频 |
| **FLOAT** | 音频 | 轻（运动latent） | flow matching | 唇形+头动，运动潜空间 |
| **SadTalker / MimicMotion** | 音频/姿态 | 轻 | UNet | 早期/姿态驱动 |

## 3.4 姿态 / 舞蹈 / 骨架

| 模型 | 管线 | 说明 |
|---|---|---|
| **Bailando** | 音乐→VQ动作码→舞蹈 | 音乐舞蹈骨架生成，踩点准 |
| **EDGE** | 音乐→SMPL 动作（扩散） | 生成 SMPL 姿态 |
| **Animate Anyone / Moore-AnimateAnyone** | 参考图+pose→视频 | ControlNet 姿态，开源复现 |
| **MagicAnimate** | 图+pose | ControlNet 系 |
| **Champ** | spline pose | 可控人体 |
| **MimicMotion** | pose（含置信度） | DWPose，动作质量好 |
| **DWPose / OpenPose** | 姿态提取器 | 关键点（身/手/脸） |

## 3.5 高效 / 边缘 / 线性注意力

| 模型 | 参数 | 架构 | 亮点 |
|---|---:|---|---|
| **SANA-Video** | 小 | Linear DiT，Block Linear Attention | 恒内存KV，分钟长视频，720p |
| **SANA-Video 2.0** | 5B/14B | 混合线性+Attention Residual | 720p 质量匹敌全softmax，3.2×快 |
| **VDN-H3** | H3改造 | 局部Softmax+双向线性，逐帧VDA | 单层2.65×，近无损 |
| **Sana / Sana 1.5（图）** | 小 | Linear DiT | 高分辨率高效图像 |
| **SANA-Sprint** | 小 | 一致性蒸馏 | 单步生成 |
| **h3.c** | — | C/Metal 重写 | Apple Silicon 端到端 H3 |
| **LongSANA** | — | 27FPS 实时分钟长视频 | 与 LongLive 合作 |

## 3.6 世界模型（World Model）

| 模型 | 参数 | 亮点 |
|---|---:|---|
| **SANA-WM** | 2.6B | 720p、1分钟、6-DoF 相机控制 |
| **NVIDIA Cosmos** | 多规格 | Physical-AI 世界基础模型 |
| **SANA-Streaming** | — | 实时流式视频编辑 |

## 3.7 关键加速/适配插件（非独立底座）

| 名称 | 作用 |
|---|---|
| **Sol-Engine / Sol-H3** | 全栈推理框架：稀疏+融合kernel+多卡通信+并行VAE |
| **Sol-Attn** | query 自适应阈值稀疏注意力，训练免，1.14–1.44×、MLP峰值−37% |
| **FastH3 (4-step LoRA)** | H3 4 步 VSA DataFree 蒸馏适配器 |
| **HyperFlow** | H3 8 步生成 |
| **FastVideo Acc LoRA** | 4/8 步加速 |
| **TeaCache / FirstBlock Cache** | 块级缓存，减少重复前向 |
| **Universe-1** | 专家拼接统一音视频 |
