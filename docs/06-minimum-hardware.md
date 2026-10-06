# 06 · 最低硬件门槛

## 6.1 已验证的"能跑"事实（来自代码与文档，非宣传）

| 方案 | 硬件 | 显存 | 手段 | 耗时 | 来源 |
|---|---|---:|---|---:|---|
| 原生 H3 两阶段 | RTX 5090 | 32GB（实测峰值 21.6GB） | CPU offload，串行阶段 | 39.8s | Sol-H3-RTX5090/docs/validation |
| 原生 H3（50步） | **RTX 4090** | **24GB** | SGLang 逐层组件 offload | 504s | RTX4090/README |
| H3 全栈 | DGX Spark (GB10) | 119.7GB 统一内存 | FP8 DiT + AdaLN预计算 + prompt缓存 | 56.2s | Sol-H3-Spark + 论文 |
| H3 | 8×GB200 | 8卡 | 全栈 | 1.434s | 论文 Tab.1 |

**关键结论：单卡 24GB（4090）已是 H3 可运行下限**，但靠主机内存 offload，速度慢（504s）。测试机主机 RAM 188GB。

## 6.2 两阶段管线权重账（HF API 实查，字节数）

Sol-H3-RTX5090 在线组件：

| 组件 | 大小 |
|---|---:|
| H3 DiT（BF16，14分片） | **66.28 GB** |
| LTX-2.5 22B transformer (BF16) | 42.02 GB |
| LTX distilled LoRA-450 | 8.90 GB |
| Qwen3-VL encoder (NVFP4 AWQ) | 15.69 GB |
| FastH3 VSA LoRA | 5.34 GB |
| LTX Conv video VAE | 1.45 GB |
| H3 latent ×2 upscaler | 0.69 GB |
| H3→LTX latent adapter | 0.39 GB |
| LTX audio VAE | 0.36 GB |
| **在线合计（含H3 DiT）** | **≈141 GB 磁盘权重** |

离线一次性（生成 prompt cache 用）：INT8 Gemma 15.4GB + INT8 connector 21.5GB ≈ 36.9GB。

> 磁盘权重 ≠ 显存：offload 时仅当前组件驻留，故 24GB 显存可跑；但需主机内存容纳权重。

## 6.3 更低显存从哪来：量化档位（Comfy-Org 实查）

H3 DiT 单文件各精度：

| 精度 | 大小 |
|---|---:|
| BF16 全量 | 66.28 GB |
| INT8 convrot | 34.04 GB |
| Pruned BF16 | 40.23 GB |
| **Pruned FP8 / INT8** | **20.96 GB** |
| **Pruned W6A8** | **15.98 GB** |

Qwen3-VL 编码器：BF16 51.5 / INT8 27.1 / **NVFP4 AWQ 15.7 GB**。

H3 VAE：FP16 5.2 / INT8 2.8 GB。

**理论极限**：W6A8 DiT 16GB + NVFP4 Qwen（可offload/缓存）+ 小 VAE。量化本身把单组件压到 16GB 级，但视频激活 + 多组件共存仍使整体需要更大内存，故需 offload 或两阶段。

## 6.4 显存阶梯选型

| 显存 | 可跑方案 |
|---|---|
| **8GB** | Wan 2.2 TI2V-5B、LTX-Video 2B、SVD、LatentSync（部分⚠️）|
| **12–16GB** | HunyuanVideo(FP8 14GB)、CogVideoX、LTX-2.3(NVFP4)、MM-Diffusion、LatentSync/MuseTalk、pose ControlNet |
| **20–24GB** | Mochi(FP8)、Wan A14B、**H3（4090 offload）**、多数 768p DiT |
| **32GB** | H3 两阶段（5090）、Wan BF16 |
| **80GB+** | LTX-2.5 dev BF16、全精度大型 DiT |
| **多卡/GB200** | H3 实时（快于播放）|

## 6.5 本机（Jetson Orin 8GB, 7GB 可用）判断

- **H3/LTX-2.x 完全无望**：最小 W6A8 单组件 16GB，超整机内存。
- **可跑**：8GB 档方案——Wan 2.2 5B（量化GGUF）、LTX-Video 2B、SVD；对嘴型轻量版。
- **OOM 硬约束**：多模型拆独立进程单训→落盘→离线合成；可用 ~650MB available 看门狗。
- **最稳妥研究路径**：MM-Diffusion 音乐舞蹈（小）或"音乐→骨架→UNet"两段式，逐段塞得进。

> 所有 7GB 级具体跑通结果均需真机实测后再下结论，当前为架构/权重推断。
