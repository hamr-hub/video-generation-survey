# 07 · 结论与选型

## 7.1 场景 → 模型

| 需求 | 首选 | 备选 |
|---|---|---|
| 最高质量/多镜头对白/音画同步，不计成本 | **MiniMax-H3** | LTX-2.5 |
| 原生音频 + 4K + 速度 | **LTX-2.3** | H3 |
| 通用、可商用、消费卡 | **Wan 2.2** | HunyuanVideo |
| 电影感运动 | HunyuanVideo 1.5 | Wan A14B |
| 8GB 低显存 | Wan 2.2 5B / LTX-Video | SVD |
| 长视频（>10s）/实时 | LTX-Video、SANA-Video、LongSANA | SANA-Video 2.0 |
| 对嘴型 | LatentSync（质量）/ MuseTalk（实时） | EchoMimic、Hallo |
| 音乐+舞蹈同步 | MM-Diffusion；两段式 Bailando/FLOAT+pose | VDN 思路 |
| 骨架可控人体 | Moore-AnimateAnyone / MimicMotion | MagicAnimate、Champ |
| 研究/微调底座 | Mochi 1、CogVideoX | SANA |
| 世界模型/相机控制 | SANA-WM | Cosmos |
| 边缘/Apple Silicon | h3.c、SANA 系列 | — |

## 7.2 核心架构判断

1. **质量线**：H3 的 Full-Attention 单流统一仍是标杆，尤其多镜头对白同步。
2. **效率线**：VDN / SANA-Video 2.0 代表的 **"局部精确 Softmax + 全局双向线性 + 逐帧记忆更新"** 是目前最有说服力的演进方向，以极小质量代价（~3.5% 注意力密度）换 2.6–3.2×。
3. **纯线性在视频上单独不够**（身份/布局漂移），混合是必由；Full Attention 被策略性使用而非到处铺。
4. **同步**：联合扩散 > 条件驱动 > 后处理对齐。物理因果强的场景（舞蹈/演奏）小模型也能做好同步。

## 7.3 对本机 7GB 的可行路线

- 放弃直接跑 H3/LTX-2.x。
- 推荐两条研究路径：
  - **A · 音乐→骨架→画面**（Bailando/FLOAT + pose ControlNet），每段塞得进，同步精确。
  - **B · 8GB 档轻量 DiT**（Wan 2.2 5B GGUF / LTX-Video 2B）做受限画质验证。
- 遵守 OOM 硬约束：拆进程、落盘、离线合成、看门狗。

## 7.4 待办 / 后续实测

- [ ] MM-Diffusion 拉取并真机验证最低显存与踩点。
- [ ] Thin-Slice：音乐→骨架 段单独跑通。
- [ ] Wan 2.2 5B 在 Orin 的量化推理实测（GGUF/CPU offload）。
- [ ] 所有"⚠️ 未真机复验"项逐个落实为真实数据。

## 7.5 参考入口

- 论文：Sol-H3 arXiv:2609.35110；VDN arXiv:2609.20744；SANA-Video arXiv:2509.24695；SANA-Video2.0 arXiv:2607.21553；HunyuanVideo arXiv:2412.03603。
- 代码：NVlabs/Sana（sol-engine）、researchmm/MM-Diffusion、Wan-Video、Lightricks、genmoai/mochi。
- 权重：MiniMaxAI/MiniMax-H3、Comfy-Org/MiniMax-H3、Lightricks/LTX-2.5、Wan-AI。
- 生态：github.com/MiniMax-AI/awesome-minimax-h3-integration。
