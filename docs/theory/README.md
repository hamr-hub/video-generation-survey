# Theory · 核心公式推导

> 纯理论章节（2026-10-07，3 路核对原文）。用公式推导讲明白生成模型、注意力与蒸馏的数学原理。
> 关键公式对照原文；原文未直接给出、由作者补出的步骤显式标注"（本文推导/待核）"。

| 章节 | 内容 | 核心恒等式 |
|---|---|---|
| [01 · 扩散 / Score / Flow](01-diffusion-score-flow.md) | DDPM 的 ELBO、前向闭式解、DSM、SDE、Flow Matching/Rectified Flow、DDIM、Consistency | $\nabla_{x_t}\log p_t=-\frac{1}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta$ |
| [02 · 注意力数学](02-attention-math.md) | Softmax、线性注意力核分解与循环态、Delta/Gated DeltaNet、Performer、VDA 逐帧方程、混合锚点 | $o_t=\phi(q_t)^\top S_t,\ S_t=S_{t-1}+\phi(k_t)v_t^\top$ |
| [03 · 蒸馏 / AR-扩散统一](03-distillation-ar-unification.md) | 渐进/Guided 蒸馏、DMD/DMD2、一致性蒸馏、MDLM 加权 MLM、Diffusion Forcing | $\nabla_\theta\mathrm{KL}\propto \mathbb E[\epsilon_{\rm fake}-\epsilon_{\rm real}]$ |

## 一图统管三套范式

1. **同一对象的不同参数化**：DDPM 学噪声 ε、Score-based 学 ∇log p、Flow Matching 学向量场 v；高斯路径下三者只差确定性线性变换。
2. **注意力的关键分裂**：softmax 分母把所有 key 耦合（无法结合律，O(N²)，但全秩可精确寻址）；线性注意力核可分解（结合律 O(N)，退化为 RNN 态，但秩受限）。
3. **混合是必然**：周期性 softmax 锚点层恢复高秩交互；VDA 把逐 token 更新提升为逐帧联合正规方程 $S(I+A)=\bar S+B$。
4. **加速的数学**：蒸馏让少步学生匹配多步教师轨迹；DMD 用两 score 之差作梯度；MDLM 证明离散扩散 ELBO 就是加权 MLM；Diffusion Forcing 让每 token 独立噪声从而统一 AR 与扩散。
