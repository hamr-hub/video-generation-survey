# 扩散模型与自回归模型融合方向深度调研

> 调研方式：直接检索 arXiv / OpenReview / 项目主页等一手资料；凡未能取得原文验证的内容均明确标注。检索时间 2026-10。

## 0. 概览：两类模型的互补性

| 维度 | 自回归（AR, next-token / next-chunk） | 扩散 / 流匹配（Diffusion / FM） |
|---|---|---|
| 强项 | 长程语义规划、组合性、可变长度、KV cache 可扩展、与 LLM 同构 | 高保真细节、纹理、并行去噪、生成质量上限高 |
| 弱项 | 误差累积、逐 token 串行慢、细节受 tokenizer 信息瓶颈限制 | 全序列二次复杂度、固定长度、长时一致性弱、多步去噪贵 |
| 融合动机 | 用 AR 管"语义骨架/时间因果"，用扩散管"像素血肉/局部细节" | |

---

## 1. ARLON（2024–2025）

- **论文**：*ARLON: Boosting Diffusion Transformers with Autoregressive Models for Long Video Generation*，arXiv:2410.20502（v1 2024-10，v3 2025-04）。
- **机构**：Microsoft（Zongyi Li、Shujie Hu、Shujie Liu、Xun Guo、Jinyu Li、Furu Wei 等），合作 KAIST、港中文、华中科大。
- **定位**：松散耦合（loose coupling）的两段式 AR→DiT 流水线，AR 做"粗"，DiT 做"精"。

**核心机制**
1. **3D latent VQ-VAE 桥接器**：将 DiT 的连续输入 latent 再压缩、量化为紧凑离散 token（AR codes）。AR 所用 VQ-VAE 压缩率为 4×8×8（时×空×空），平衡 AR 学习难度与信息密度。
2. **decoder-only AR Transformer**：给定文本 prompt，自回归生成全片粗粒度离散视觉单元，提供粗空间布局 + 长程时序信息。
3. **语义注入模块（semantic injection）**：比较了 MLP adapter / adaptive norm / ControlNet 三种方式，最终采用 adaptive norm 类注入；AR codes 分段送入 DiT，下一片段同时以上一段最后若干帧为条件。
4. **抗 AR 噪声训练（关键创新）**：teacher-forcing 到自回归推理存在 exposure bias，AR 推理出的 token 带噪。对策：①用**压缩率更高的另一个 VQ-VAE** 产出的更粗 latent token 训练 DiT；②注入模块加 **uncertainty sampling（不确定性采样）**，提升 DiT 对噪声条件的容忍度。
5. 训练目标仍是标准扩散去噪：$X^t=\sqrt{\alpha_t}X^0+\sqrt{1-\alpha_t}\,\epsilon$，在给定 AR 语义条件 $F$ 下学习去噪。

**实测数字（原文）**
- VBench 上 **11 项指标中 8 项优于基线 OpenSora-V1.2**，动态程度（dynamic degree）与美学质量提升显著，其余 3 项持平；600 帧长视频对比 FreeNoise / StreamingT2V / OpenSora-V1.2 为开源最优。
- **推理效率**：AR codes 作初始化后 DiT 仅需 5–10 步即可达到基线 30 步质量。单 A100 生成 68 帧：OpenSora 30 步约 47 秒；ARLON DiT 5–10 步约 11–19 秒 + AR 约 6 秒 → **相对提速 47–64%**；附录 Table 4 给出推理速度相对提升 48–49%、FLOPs 下降约 64%（68f 与 578f 一致）。
- 支持渐进式 prompt（progressive text prompts）的长视频生成。

**局限（原文 A.4）**
- 构建于 OpenSora-V1.2 之上，画质上限受底座限制（可换 CogVideoX-5B 等缓解）；
- 2K 分辨率下 AR 码序列过长，训练/推理不可行，需更高压缩 VQ-VAE 或并行预测；
- 化妆、进食等手部细节仍易失真。

**与其他工作关系**：ARLON 是"AR 规划 + 扩散精修"松散级联范式的代表，思路与民间对 GPT-4o（tokens→AR→diffusion→pixels）的猜测一致；BLIP3-o（arXiv:2505.09568，Salesforce）即以此类混合流水线为动机；MammothModa2 是该路线的端到端加强版（见 §6）。

---

## 2. Pyramid Flow（ICLR 2025）

- **论文**：*Pyramidal Flow Matching for Efficient Video Generative Modeling*，arXiv:2410.05954（v2 2025-03），**ICLR 2025**。
- **机构**：北大（Yadong Mu、Zhouchen Lin 等）、快手（Quzhe Huang 等）、南洋理工等；项目 pyramid-flow.github.io，代码/权重开源。
- **定位**：单模型内把"空间多尺度"与"时间自回归"统一进流匹配，替代传统级联（cascade）多模型方案。

**核心机制与公式要点**
1. **空间金字塔流**：把整条去噪轨迹重解释为若干金字塔阶段，仅最后阶段在全分辨率。借助 FM 可在任意分布间插值的灵活性，定义跨分辨率的分段（piecewise）流：时间窗内，在"下采样+更噪" latent 与"上采样+更干净" latent 之间插值：
   $$u_t^{(i)}=(1-\tilde t)\,\sigma^{(i)}+\tilde t\,u_c^{(i)}$$
   形式从"低分辨率噪声"连续生成并解压到"高分辨率数据"；空间 $L$ 级（实验 3 级）均匀划分时算力近似降至 $\frac1L$ 量级。
2. **统一训练**：生成与超分由**同一个 DiT、单一 FM 目标**端到端优化（基线 MM-DiT，SD3 Medium 风格，2B 参数），各阶段流相互衔接而非从纯噪声重启；端点噪声同向耦合以保持轨迹"直"。
3. **跳转点 renoising（Eq.12–15）**：推理时跨分辨率上采样后，用缩放 + 校正高斯噪声匹配相邻阶段高斯分布的均值/协方差，保证概率路径连续（借鉴 jump diffusion 的跨维处理）。
4. **时间金字塔自回归**：以自回归方式逐段预测下一视频 latent，但历史条件用**逐级压缩的低分辨率历史金字塔**（越久远的帧分辨率越低）：
   $$h_{t-1}^{(\ell)},\ h_{t-2}^{(\ell+1)},\dots$$
   并对历史条件加小幅腐蚀噪声（类似 Diffusion Forcing）以缓解自回归退化。10 秒 241 帧视频 token 数由 119,040 降至 **15,360**。
5. 块级因果注意力（blockwise causal attention）保证严格左→右；位置编码在空间金字塔外推、在时间金字塔插值对齐。

**实测数字（原文）**
- **仅 20.7k A100 GPU 小时**训练出 768p、24FPS、5 秒（最长 10 秒、241 帧）模型；对比 Open-Sora 1.2 用 4.8k Ascend + 37.8k H100 小时只训 97 帧，算力 2 倍以上且画质更差。
- VBench：总分 **81.72**、质量分 **84.74**（高于 Gen-3 的 84.11）、运动平滑度 99.12、动态程度 64.63；总分/质量分超过 2 倍体量的 CogVideoX-5B。语义分偏低（69.62），作者归因于粗粒度合成 caption。
- EvalCrafter 总分 244；人类偏好研究中优于 Open-Sora、CogVideoX-2B，尤其运动平滑性。
- 推理：生成 5 秒 384p 约 **56 秒**；T2V 训练的模型无需微调即可做 I2V（首帧即图像条件）。

**局限**：语义对齐分受 caption 质量拖累；384p 推理仍不算快；本质仍是分阶段插值，跨尺度连续性依赖精细的 renoising 设计。

**与其他工作关系**：与 ARLON 同样追求"省 token + AR 长视频"，但 Pyramid Flow 把多尺度融进一个连续流模型（单模型端到端），而非两个模型级联；其"历史加噪/独立噪声水平"直接呼应 **Diffusion Forcing**，时间因果 chunk 路线与 MAGI-1 同源思想但实现更早、更轻。

---

## 3. MAGI-1（2025）

- **论文**：*MAGI-1: Autoregressive Video Generation at Scale*，arXiv:2505.13211（2025-05），Sand.ai；代码/权重开源（SandAI-org/MAGI-1，配套 MagiAttention）。
- **定位**：大规模"自回归去噪"世界模型——chunk 级单调噪声时间表 + 流水线并行，是 AR↔diffusion 在工程规模上的标志性融合。

**核心机制与公式要点**
1. **chunk 自回归**：视频切成固定 chunk（每 chunk 24 帧 = 24FPS 下 1 秒），逐 chunk 自回归；每 chunk 整体去噪，条件为所有前序 chunk：
   $$v_\theta(z_t^{[i]},\,t_i\mid z^{[<i]},\,c)$$
2. **单调噪声时间表（区分于普通双向扩散）**：训练时每 chunk 的噪声水平随时间（chunk 序号）单调递增/对齐，强制因果时间建模：$t_1\le t_2\le\cdots$；普通双向模型则是 $t_1=\cdots=t_n$ 同一噪声水平。
3. **流水线并行推理**：一个 chunk 去噪到一定程度（不必全干净）即启动下一 chunk，最多 **4 个 chunk 并行**，显著降低后继 chunk 延迟；峰值算力/显存与视频总长度无关（常数），支持流式实时生成。
4. Transformer 主干：空间双向、时间因果去噪；VAE 为自研 **Transformer-VAE**（空间 ×8、时间 ×4 压缩；PSNR 36.55、解码 12.28ms，比多数开源 VAE 快）。
5. **三分解引导（Eq.9）**：将 score 分解为无条件先验 + 时序上下文 + prompt 条件三项，时间引导权重 $\lambda_t$ 取 1.5 以消除 chunk 间接缝；蒸馏模型在最后去噪区间反向降低前序影响（Eq.10）。
6. timestep 采样用 Logit-Normal；配套 prompt enhancement（蒸馏到约 7B 小模型，约 2M 样本）。

**实测数字（原文）**
- 最大变体 **24B 参数**，支持上下文长达 **4M token**。
- VBench-I2V 与 Physics-IQ 上相对前代模型显著提升，尤其复杂运动、主体完整性、物理合理交互（论文报告为 substantial improvements；**具体分项数值在 v1 HTML 中以图表呈现，本次未取得可逐字引用的完整数值表，标注待核**）。
- 原生统一 T2V、视频续写、I2V 于同一预训练，无需任务微调；chunk-wise prompting、可控镜头转场（通过不同去噪阶段调整 KV range 实现）。

**局限**：24B + 超长上下文依赖 MagiAttention 等专用基础设施，工程门槛高；蒸馏模型存在随播放累积的饱和/色彩失真（文中专门处理）；质量随 chunk 数仍有漂移风险。

**与其他工作关系**：MAGI-1 是 Diffusion Forcing 思想（独立/单调 per-chunk 噪声 + 因果）规模化的代表；FlowCache（§4）直接以 MAGI-1 为主要加速对象；SkyReels-V2（快手，arXiv:2504.13074）为同路线无限长度模型。

---

## 4. FlowCache（ICLR 2026）

- **论文**：*Flow Caching for Autoregressive Video Generation*，arXiv:2602.10825（2026-02-11）；**ICLR 2026 poster**（OpenReview vko4DuhKbh）。
- **机构**：厦门大学（Rongrong Ji 组，Yuexiao Ma 等）+ 字节跳动（一作实习期间完成）。
- **定位**：首个专为**自回归视频生成**设计的缓存加速框架（training-free），解决普通扩散缓存"所有帧同一去噪阶段"假设在 AR 模型中失效的问题。

**核心机制与公式要点**
1. **经验与理论观察**：AR 模型中不同 chunk 在同一时间步处于**异构去噪阶段**。论文证明（Theorem 1，幂律调度 $\alpha(t)=(1-t/T)^p$、最优速度场假设下）相邻步输出的相对 L1 距离随去噪推进**单调增大**（越接近干净，相邻步越不相似）；Corollary 1 进一步说明不同 chunk 因状态范数不同而相似度不等。
   $$d_i(t)=\frac{\|F_i(t+1)-F_i(t)\|_1}{\|F_i(t)\|_1}$$
2. **chunkwise 独立缓存策略**：每个 chunk 维护独立缓存策略，按自身 $d_i(t)$ 决定重算或复用：
   $$d_i(t)<\tau \Rightarrow \text{reuse cache};\quad \text{else recompute}$$
   并排除每 chunk 前 $k$ 个时间步不缓存（MAGI-1 与 SkyReels 取不同 $k$）。阈值：MAGI-slow 0.01 / fast 0.015；SkyReels-slow 0.1 / fast 0.15。
3. **联合重要性–冗余度 KV cache 压缩**：在段/帧/像素区三个层级度量 KV 重要性与冗余，保留"重要且非冗余"token，使显存有固定上界，是首个针对 AR 视频生成 KV 压缩的系统研究。

**实测数字（原文摘要，可逐字引用）**
- MAGI-1 上加速 **2.38×**；SkyReels-V2 上加速 **6.7×**。
- 质量变化极小：VBench 在 MAGI-1 上 **+0.87**（甚至略升），SkyReels-V2 上 **−0.79**。
- 背景成本参照：MAGI-1-4.5B-distill 全 KV cache 生成高分辨率视频需数十 GB 显存、数分钟（论文据此论证缓存必要性）。

**局限**：阈值/预热步数需按模型手工配置；强依赖 chunk 间异构性这一结构性前提；training-free 缓存与 KV 压缩叠加的极端长视频质量仍有边界。

**与其他工作关系**：正交于量化/稀疏化/蒸馏；相较 TeaCache、ToCa、DiCache（均面向同步扩散）首次适配 AR chunk 时间轴；是 MAGI-1 / SkyReels-V2 走向实时超长视频的关键推理层，不改变训练范式。

---

## 5. Masked / 离散扩散语言模型（融合的理论支线）

### 5.1 MDLM（NeurIPS 2024）
- **论文**：*Simple and Effective Masked Diffusion Language Models*，arXiv:2406.07524；Cornell Tech（Sahoo、Rush、Kuleshov 等）。
- **机制**：仅用吸收态（[MASK]）前向过程；提出 substitution（SUBS）参数化与连续时间 **Rao-Blackwellized ELBO**，目标化简为经典 MLM 损失的加权混合——把 BERT 式 encoder-only 模型变成可生成模型；采样器比既有扩散快 3–4×，支持半自回归（semi-AR）任意长度生成。
- **数字**：LM1B 扩散类 SOTA（报告 ≤23.00，对比 SEDD ≤32.79、D3PM ≤76.90），与 AR 的差距缩到约 14–25%；官方仓库 SAR 解码 Gen. PPL 27.18、89.3 s/seq，比 SSD-LM（35.43、2473.9）快 25–30×。
- **意义**：证明"masked diffusion ≈ 有序 MLM 损失"，是 AR 与扩散在目标层面统一的数学桥梁。

### 5.2 Block Diffusion（2025）
- **论文**：*Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models*，arXiv:2503.09573；Cornell Tech（Arriola 等）。
- **机制**：在**离散 token 块**上定义自回归分布，块内条件分布用离散去噪扩散（BD3-LM）；块大小为 1 退化为 AR，块为全序列退化为扩散——显式在二者间插值。支持变长生成与 KV cache；推导梯度方差估计器，提出数据驱动噪声调度最小化方差（解释了块大小为 1 时扩散目标仍劣于 AR 的原因）。
- **结果**：语言建模基准上扩散类新 SOTA，生成任意长度。后续 **Set Diffusion**（arXiv，Arriola & Kuleshov）以可变位置/长度 token 集泛化固定块，支持每步更新 KV 与任意顺序解码；**Scaling Beyond MDLM**（arXiv:2602.15014）把 MDLM / Duo（uniform-state）/ Eso-LM（interpolating）扩到 1.7B，指出困惑度跨族不可比、uniform-state 在 GSM8K 上反超。
- **与视频路线关系**：语言侧的"块级 AR + 块内扩散"正是 MAGI-1"chunk 级 AR + chunk 内扩散去噪"在文本域的同构对应，两条线共享同一套插值哲学。

### 5.3 先驱：MaskGIT / D3PM / VQ-Diffusion
- D3PM（Austin et al., 2021）确立离散类别转移矩阵扩散，发现吸收态（mask）最稳；MaskGIT（Chang et al., 2022）用 mask-predict 做并行图像 token 生成；VQ-Diffusion（微软，cientgu/VQ-Diffusion）在 VQ-VAE latent 上用离散扩散免除 AR 逐 token 累积误差。它们是 Show-o / MDLM / MAGI 路线的共同前身。

---

## 6. AR↔Diffusion 统一架构

### 6.1 Show-o（ICLR 2025）
- **论文**：*Show-o: One Single Transformer to Unify Multimodal Understanding and Generation*，arXiv:2408.12528；NUS Show Lab（基于 Phi-1.5，**1.3B**）。
- **机制**：**单个 Transformer 内按模态切换目标**——文本走 AR（next-token），图像 token 走离散扩散（MaskGIT 式并行去噪）；统一 prompting + omni-attention，无需独立文本编码器。VQA/ caption 用 AR，T2I/inpainting/extrapolation 用扩散。
- **数字**：MSCOCO zero-shot FID-30K **9.24**（优于 GLIDE 5B、DALL·E 2 6.5B）；GenEval **0.68**，比同级 LDM(1.4B) 高约 0.24；理解任务可比肩 LLaVA-1.3B。约 50 步出图（AR 基线约千步量级）。
- **意义与局限**：首次在单模型中真正统一 AR 与（离散）扩散，是"下一代底座"叙事的开山之作；但图像生成仍基于离散 token、分辨率/保真度受 tokenizer 限制，未覆盖连续高保真视频。

### 6.2 Diffusion Forcing（NeurIPS 2024）
- **论文**：*Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion*，arXiv:2407.01392；MIT/IBM（Boyuan Chen、Vincent Sitzmann 等）。
- **机制**：序列中**每个 token 配独立随机噪声水平**，用因果 next-token 模型按任意 per-token 调度去噪；噪声=部分掩码，统一 AR（变长、因果）与全序列扩散（引导、连续信号）二者优点。理论上优化所有子序列似然的变分下界。
- **数字**：可稳定 rollout 超过训练 horizon 的视频（teacher forcing 基线发散）；fruit-swap 机器人记忆任务成功率约 **80%**（无记忆基线近 0）；时序预测 Electricity/Traffic 的 CRPSsum 具竞争力。
- **关系**：是 MAGI-1、Pyramid Flow 时间条件、SkyReels 等"因果 chunk + 去噪"的直接理论源头；v2（History Guidance）改用 DiT + latent diffusion 做超长视频。

### 6.3 BLIP3-o（2025）
- **论文**：arXiv:2505.09568，Salesforce。
- **机制与发现**：系统研究统一多模态中 AR 与扩散的组合；以 **DiT 生成 CLIP 语义特征**（而非传统 VAE latent）提高效率与质量；提出"先理解、后生成"的顺序预训练以保住理解能力；并引用外界对 GPT-4o"tokens→AR→diffusion→pixels"混合架构的猜测作为动机。

### 6.4 MammothModa2（2025-11，arXiv 预印本）
- **论文**：arXiv:2511.18262；字节跳动 MammothModa 团队，约 13B（基于 Qwen3-VL-8B，加 5B 生成参数）。
- **机制**：串行 **AR–Diffusion**：AR 路径（生成专家 + MammothTok 统一视觉分词器）做全局离散语义建模，单流 DiT 做高保真合成，中间用多层特征聚合 + 统一条件编码 + in-context conditioning 的对齐模块桥接；**端到端联合 NTP 与 Flow Matching** 训练，再 SFT + RL（DiffusionNFT），约 60M 监督样本、不依赖预训练生成器。
- **数字**：GenEval **0.87**、DPGBench **87.2**、ImgEdit **4.06**，理解任务与 Qwen3-VL-8B 持平。
- **定位**：ARLON 式松散级联的"端到端 + RL"成熟形态，代表统一理解/生成/编辑的工业级方向。

### 6.5 其他相关（背景，未逐一展开）
- **VideoPoet**（Google，ICML 2024，arXiv:2312.14125）：decoder-only Transformer，纯 AR 预测视频/音频/图像/文本 token，多模态生成目标混合预训练 + 任务适配；展示纯 AR 路线也能产生高保真运动，是 AR 视频路线的旗舰基线（本身不含扩散，作为对照）。
- **Streaming AR via Diagonal Distillation**（ICLR 2026 poster）：对角蒸馏实现流式 AR 视频，5 秒视频 2.61 秒生成（最高约 31 FPS）——属"蒸馏加速 AR"支线，与 FlowCache 互补。
- 任务上下文中提到的"ECCV2026 具身底座""Parallel Decoding Distillation"及"分离扩散"：**本次检索未能定位到可逐字核验的原始论文/原文**（详见 §8），不做数值与机制断言。

---

## 7. 苏剑林"分离式 / 多模态思路"相关博客（标注取证情况）

- 任务指向 `spaces.ac.cn/archives/10197`，实为苏剑林《"闭门造车"之多模态思路浅谈（二）：自回归》（2024-07-08，站点另有 kexue.fm 镜像，同编号）；同系列（一）为《无损输入》（archives/9984，2024-02-21）。
- **取证状态：本次调研中 spaces.ac.cn 与 kexue.fm 均拒绝抓取（extract 返回 upstream forbidden；本机直 curl 仅得 124 字节拦截页；浏览器后端 Camofox 未运行）。仅能从第三方索引确认标题、日期与主题（"围绕视觉自回归展开，讨论多模态本质难度、世界模型"）。**
- 因此，**不对该文具体公式与"分离扩散"命名下的技术细节做任何编造或断言**；可确认的仅是其讨论视觉 AR、多模态建模本质难度与世界模型。若需精确引用，建议在可访问 kexue.fm 的环境补抓，或由用户提供正文。
- 说明：中文社区所谓"分离式"思想（语义/规划与像素生成分离、各模态用各自最适合的生成目标）在学术论文中的对应物即 Show-o（按模态分目标）、ARLON / Mammoth2（AR 与扩散分工）与 Block Diffusion（块间 AR、块内扩散）。

---

## 8. 未能核实的条目（诚实标注）

1. 苏剑林 10197 正文公式与机制细节 —— 源站反爬，未取证（§7）。
2. "ECCV 2026 具身底座"具体论文 —— 检索未定位唯一对应原文，未展开。
3. "Parallel Decoding Distillation" 具体出处 —— 未定位到确切原始论文。
4. MAGI-1 VBench-I2V / Physics-IQ 的逐项数值 —— v1 HTML 数值多在图表/图片中，未取得可逐字引用文本表；仅引用论文明确文字结论。
5. 上下文将 FlowCache 描述为"厦大&字节" —— 与原文署名一致（一作厦大、字节实习，通讯厦大），已核实。

---

## 9. 结论：为什么是下一代底座候选

1. **能力互补、短板互消**：AR 提供 LLM 同构的长程语义规划、组合性、可变长度与推理/ agent 能力；扩散/流匹配提供高保真细节与高画质。融合模型可在同一参数空间内同时做"理解 + 生成 + 编辑 + 交互决策"。
2. **统一多模态的最短路径**：Show-o、BLIP3-o、Mammoth2 表明按模态/阶段分工（文本 AR、视觉扩散）能以一个模型覆盖 VQA、T2I/I2V、inpainting、视频续写，并保持理解不掉队——契合 GPT-4o 式产品形态（外界推测即 tokens→AR→diffusion→pixels）。
3. **复杂度可从二次降到线性/常数**：chunk/块级 AR + KV cache 使长视频推理复杂度对总长度近似线性、峰值显存常数（MAGI-1）；再叠加 FlowCache（最高 6.7×）、对角蒸馏（可达实时），让"无限长度 / 实时流式 / 世界模型 / 具身决策"首次工程可行。
4. **训练效率革命**：空间 + 时间金字塔把 10 秒视频 token 压缩约 7.8 倍，20.7k A100 小时即训出 768p 模型；多尺度单模型端到端替代级联多模型，知识共享、扩展性好。
5. **理论上可证明的统一**：Diffusion Forcing（独立噪声水平 = 部分掩码）与 MDLM（masked diffusion = MLM 损失混合）从似然/ELBO 层面证明 AR 与扩散是同一插值谱系的两端，而非对立路线，为规模定律与统一优化提供基础。

## 10. 主流融合范式分类

| 范式 | 结构 | 代表 | 特点 |
|---|---|---|---|
| **A. 松散级联（AR 规划 → 扩散精修）** | 两个模型，AR 出粗 token/语义，DiT 条件化出像素 | ARLON、BLIP3-o、（推测的）GPT-4o、Mammoth2(串行) | 工程简单、即插即用；有 exposure bias，需抗噪/对齐模块 |
| **B. 单模型按模态切换目标** | 一个 Transformer，文本 AR、视觉离散扩散 | Show-o | 参数统一、理解生成兼顾；保真度受离散 tokenizer 限制 |
| **C. 因果 chunk + 扩散去噪（diffusion forcing 系）** | chunk/块间单调/独立噪声、因果自回归，块内多步去噪 | Diffusion Forcing、MAGI-1、SkyReels-V2、Pyramid Flow(时间维)、Block Diffusion(文本) | 长度可扩展、流式、常数峰值显存；需专门注意力/调度 |
| **D. 多尺度/金字塔流统一** | 单 DiT 内跨分辨率分段流 + 压缩历史金字塔，端到端 FM | Pyramid Flow | 训练极省、单模型替代级联；语义对齐依赖 caption/renoising |
| **E. 训练免费缓存 / 蒸馏加速层** | 不改训练，chunkwise 缓存 + KV 压缩；或蒸馏并行/流式解码 | FlowCache、Diagonal Distillation、TeaCache 系 | 正交叠加，直接换取实时性；阈值需逐模型调 |
| （对照）纯 AR / 纯扩散 | 单范式 | VideoPoet（纯 AR token）；Sora 类全序列扩散 | 作为融合收益的基线 |

**总体判断**：范式 C（因果 chunk + 扩散）正在成为视频/世界模型主干，范式 A/B 在多模态理解-生成统一上收敛，范式 D、E 分别从训练侧与推理侧补足成本短板；三者叠加（因果 chunk 主干 + 金字塔省 token + cache/蒸馏实时化 + AR 语义规划）构成当前最清晰的"下一代多模态/视频底座"技术栈。
