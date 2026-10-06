# Deep Dive · 五方向论文深挖

> 基于原始论文的深度整理（2026-10-06，5 路并行研究）。每篇含机构、核心机制/公式、实测数字、局限与关系。
> 数字一律以原文为准，论文未报告的明确标注，不做推测。

| 章节 | 方向 | 覆盖代表工作 |
|---|---|---|
| [01 · 高效注意力深挖](01-attention-deep.md) | 线性/混合/稀疏 | VDN、SANA-Video/2.0、Sol-Attn、Gated DeltaNet、KDA、SLA/SLA2、SageAttention |
| [02 · 扩散↔自回归融合](02-ar-diffusion-fusion.md) | AR 与 DiT 统一 | ARLON、Pyramid Flow、MAGI-1、FlowCache、MDLM、Diffusion Forcing、Show-o |
| [03 · 原生多模态与 VAE](03-multimodal-vae.md) | MM-DiT 与潜空间 | H3、LTX-2/2.3/2.5、跨VAE适配器、Wan VAE、Cosmos Tokenizer |
| [04 · 世界模型/物理/具身](04-world-models.md) | 生成→世界模型 | SANA-WM、Cosmos、Genie 1/2、UniSim、GameNGen、Sora |
| [05 · 推理加速](05-inference-acceleration.md) | 蒸馏/量化/缓存/RSI | Sol-Engine、SANA-Sprint、DMD/DMD2、AGD、CausVid、FastH3 |

## 跨方向综合判断

1. **注意力层**：线性单独不够（身份/布局漂移），混合（局部softmax+全局线性+锚点层）定型，比例/放置走向自动搜索。
2. **范式层**：AR 与扩散收敛为统一底座。**因果 chunk 主干 + 金字塔省 token + cache/蒸馏实时化 + AR 语义规划**是最清晰的下一代技术栈。
3. **模态层**：两种统一路线并存——H3 式（token 统一、全自注意力，需稀疏化）与 LTX 式（流不统一、时间轴与去噪进程统一、交叉注意力软对齐）。
4. **边界层**：世界模型比通用视频生成多出四类能力——逐步动作接口、状态记忆、闭环长程一致性、可决策的交互语义。
5. **效率层**：栈顺序通常为 步数/CFG蒸馏 → 稀疏attn/cache → token剪枝 → 选择性量化 → RSI fusion 收尾；量化须 selective 且与融合协同。
