# 高效注意力架构：线性 / 混合 / 稀疏注意力在视频生成中的演进

> 调研分支：混合（Hybrid）、线性（Linear）、稀疏（Sparse）注意力。
> 数据来源：各论文 arXiv 摘要页与 HTML 全文（访问时间 2026-10-06）。凡论文正文未明确给出的数字一律标注「论文未报告」，不做推测。
> 注意：本文涉及的部分论文编号（2607.xxxxx、2609.xxxxx）为 2026 年新论文，第三方 benchmark 横评数据较少，请以原文为准。

---

## 0. 问题背景与三条技术路线

视频扩散 Transformer（Video DiT）在去噪循环中反复处理数万到数十万量级的时空 token。以 MiniMax H3 的实测 profile 为例，**Softmax 注意力占去噪器运行时间的 85% 以上**（Video DeltaNet 论文数据）。降低注意力开销的路线可归纳为三类：

1. **线性注意力（Linear Attention）**：用特征映射把 Softmax 核分解为 $\phi(q)^\top \phi(k)$，借助矩阵乘法结合律将复杂度由 $O(N^2)$ 降到 $O(N)$，推理时退化为固定大小的循环状态（fast weight / RNN 视角）。代表：Gated DeltaNet、SANA-Video。
2. **混合注意力（Hybrid Attention）**：大部分层用线性注意力，周期性插入少量全 Softmax「锚点层」（典型比例 3:1，即 25% Softmax），恢复线性注意力缺失的高秩（full-rank）token 交互。代表：Kimi Linear（KDA）、SANA-Video 2.0、Video DeltaNet。
3. **稀疏注意力（Sparse Attention）**：保留 Softmax 语义但只计算部分 KV 块，分为训练后动态稀疏（training-free，如 Sol-Attn）与可训练稀疏/稀疏-线性融合（如 SLA）两类。低比特量化注意力（SageAttention 系列）则是正交的第四条加速维度，常与上述路线叠加。

---

## 1. Video DeltaNet（VDN）

- **论文**：Video DeltaNet: A Video-Native Hybrid Attention for Livestream Video Generation（arXiv:2609.20744，2026-09，v2；20 页）
- **机构**：作者含 Kurt Keutzer、Xiuyu Li、Zhaoyang Lv、Chenfeng Xu、Haiwen Feng 等；脚注注明「部分工作在 Impossible, Inc. 实习期间完成」。**摘要页未给出统一机构署名**，可大致归为 UC Berkeley 系作者群与 Impossible, Inc.（具体以论文全文为准）。开源：github.com/OpenVDN/vdn-minimax-h3，权重在 HuggingFace OpenVDN。
- **基座模型**：MiniMax H3（视频+文本+音频多模态直播生成模型），VDN 只替换 **video-to-video 交互**；涉及文本、音频的交互仍保留 Softmax。

### 1.1 核心机制

VDN = 局部 Softmax + 边界锚点 + 双向线性记忆，按时距角色切分注意力：

1. **滑窗 Softmax 注意力**：窗口对齐 VAE 的时间块（H3 VAE 一次解码 5 个连续 latent 帧），每个 query 块关注自身及前后相邻块 → 15 帧窗口，保留纹理、边界、短期运动所需的精确 token 匹配。
2. **边界锚点（Boundary Anchors，四连通）**：每帧都可直接 attend 到首、尾两个 latent 帧的全部 token；首尾帧反过来 attend 全序列。仅两行/两列获得稠密连接，代价小，且对首末帧条件生成（I2V、FL2VA）特别自然。
3. **双向线性注意力**：前向扫描状态总结窗口之前的帧、反向扫描状态总结窗口之后的帧；锚点帧不入状态（避免重复计数）。文本 prompt 先被压缩为一个 state，用它初始化两个方向的扫描，使线性分支文本感知，且 prompt 恰好计数一次。
4. **Video Delta Attention（VDA）——帧级 delta 规则（论文核心）**：
   - 标准 Gated DeltaNet / KDA 是**逐 token** 更新；视频扩散中一帧的所有空间 token 同时可得。简单做法（论文称 SANA-WM 采用的 batched additive 构造）是在同一衰减后状态上并行累加各 patch 的修正，但所有残差都基于更新前状态，相似 key 方向的写入会互相冲突或叠加。
   - VDA 把一帧定义为一个联合目标：新状态既要贴近衰减后的旧状态，又要同时拟合帧内全部 key–value 关联：
     $$S' = \arg\min_S \tfrac12\|S-\alpha S_{t-1}\|^2 + \tfrac12\sum_{i=1}^m \beta_i (k_i^\top S - v_i)^2$$
     一次求导得到闭式正规方程更新：
     $$S' = \alpha S_{t-1}(I + K^\top B K)^{-1} + K^\top B V (I + K^\top B K)^{-1}$$
     逆矩阵把帧内 key 相关性耦合进写入：重复 key 得到饱和权重，正交 key 各自独立写入。
   - **稳定性命题（Proposition 1）**：固定特征与门控时，继承状态的转移是非膨胀的（算子范数 $\le 1$），旧帧贡献不会被帧更新放大；而 SANA-WM 的加法构造需要额外乘 $1/\sqrt{\text{帧大小}}$ 才能稳定，VDA 无需帧尺寸缩放。
5. **分支融合**：两分支共用预训练 QKV 投影；线性分支沿用 Gated DeltaNet/KDA 风格特征处理（K/V 可分离短卷积→SiLU，Q/K 做 L2 归一化，无 RoPE）。Softmax 读出加内容相关 sigmoid 门；线性读出先 RMSNorm 再过独立 sigmoid 门；**两分支各有独立输出投影**后相加，允许它们向残差流的不同方向贡献。
6. **分阶段教师对齐训练（不改基础模型）**：A1 逐层对齐线性分支（200 步，骨干冻结）→ A2 端到端对齐（500 步）→ B 阶段对 Q/K/V/O 加 LoRA 联合适配（2000 步）→ DMD2 风格（去掉 GAN 项）蒸馏 50 步到 8 步（250 个 generator step）。

### 1.2 关键实测数字

工作负载：14.3 秒、768p（1344×768，345 帧 @24fps，102 个 latent 帧），训练集仅 10,015 个片段，评测 103 条固定第三方 prompt。

| 配置 | GPU | NFE | 端到端延迟 |
|---|---|---|---|
| Dense H3 | 1×B200 | 50 | 799.6 s |
| VDN-H3 | 1×B200 | 50 | 307.9 s |
| VDN-H3 few-step | 1×B200 | 8 | 49.3 s |
| VDN-H3 分布式 | **8×B200** | 8 | **6.70 s** |

- 官方口径：**8×B200 上 6.70 s 完成 DiT 去噪，相对同 GPU 数 50 步稠密 H3 提速 14.5×**（含步数蒸馏与系统优化的综合收益）。
- 纯架构收益（同 NFE 一次完整 transformer 评测）：H200 上 35.35 → 11.16 s（**3.2×**），B200 上 16.0 → 6.2 s（**2.6×**）。
- 随长度扩展：latent 帧 42→102 时 Softmax 注意力密度从 42.1% 降到 20.0%，骨干提速从约 1.8× 增到 3.2×（H200）/2.6×（B200）。
- 内核：小矩阵求逆融合内核相对 Cholesky 多内核方案 4.6–5.0×；VDA 四个融合 Triton 内核（Prep/Stats/Gather/Epilogue），Gather 提速 7.2–7.5×。
- **质量**：8 步 VDN-H3 在 5 个无参考质量指标（EvalCrafter VQAA/VQAT、Q-Align、FAST-VQA、DOVER++）上追平或略超 50 步 Dense H3（差异 +0.06~+1.00）；RAFT 运动量 11.71 vs 11.55 px；FIRM-Video：PQ 4.45 vs 4.40、IF 2.25 持平、WC 1.77 vs 1.84。首末帧条件（FL2VA）：PSNR 28.67 vs 28.85 dB、SSIM 0.826 vs 0.833、LPIPS 0.1156 vs 0.1044——损失很小但确实存在。

### 1.3 局限性

- 只替换 video-video 注意力，跨模态（文本、音频）交互仍是 Softmax；线性分支在首末帧条件保真度上仍有轻微但系统性的损失（LPIPS 略升）。
- 数据规模很小（1 万片段、103 条 prompt 评测），结论外推性有限；14.5× 是架构 + 8 步蒸馏 + 8 卡并行 + SGLang/MXFP8 的**全栈**数字，不能等同于注意力算法本身收益。
- 依赖预训练教师模型的对齐配方，不是从零训练。

### 1.4 与其他工作的关系

- 线性分支直接沿用 **Gated DeltaNet 与 KDA（Kimi）** 的门控/特征图设计，并把它们的 token 级规则升级为帧级规则；论文明确对比 **SANA-WM** 的 $1/\sqrt{\text{帧大小}}$ batched additive 方案，指出 VDA 用逆矩阵天然获得稳定性。
- 与 SANA-Video 2.0 同属「混合」阵营，但 VDN 是在 Softmax 基座上**后装**线性路径，SANA-Video 2.0 是**从零训练**；VDN 的混合粒度是「层内按时距切分」，SANA-Video 2.0 是「层间 3:1」。

---

## 2. SANA-Video（1.0）

- **论文**：SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer（arXiv:2509.24695，2025-09，v2；21 页）
- **机构**：MIT（Song Han、Han Cai）、NVIDIA、香港大学（Ping Luo）、University of Toronto / Vector Institute（Sanja Fidler）等多机构合作（Enze Xie 为通讯主导者之一；摘要页未列机构，按作者隶属关系归纳）。

### 2.1 核心机制

1. **纯线性 Linear DiT**：在 SANA 图像架构基础上扩展，使用 **ReLU 核线性注意力**并集成 RoPE。关键顺序是 **RoPE(ReLU(·))**：先 ReLU 再施加旋转位置编码，避免 ReLU 把位置信息滤掉。
   - 稳定性技巧：RoPE 会破坏 ReLU 输出的非负性，可能使线性注意力归一化分母趋零。做法是**分子中 q、k 都加 RoPE，分母中去掉其中一方的 RoPE**，保证分母为正。
   - 线性注意力基本形式：
     $$o_i = \frac{\sum_j \phi(q_i)^\top \phi(k_j)\, v_j}{\sum_j \phi(q_i)^\top \phi(k_j)},\qquad \phi=\mathrm{ReLU}$$
2. **带时空混合的 Mix-FFN**：SANA 用卷积补线性注意力的局部性，本文在 Mix-FFN 上追加时间 1D 卷积（带 shortcut），改善运动连续性（消融显示训练损失显著下降）。
3. **Constant-Memory KV Cache（LongSANA，块线性注意力）**——长视频生成的核心：
   - 线性注意力的递归状态 $S_i = S_{i-1} + \phi(k_i)v_i^\top$，$z_i = z_{i-1}+\phi(k_i)$。自回归生成新块时只需保存上一状态与键累积和，**缓存成本 $O(d^2)+O(d)$，与视频长度无关**，计算成本每 token $O(d^2)$；而因果全注意力的 KV cache 与计算都随长度线性增长，被迫退化为局部窗口、丢失全局上下文。
   - 配合块因果 Mix-FFN：训练时块尾 zero-padding 防泄漏，推理时缓存上一块最后一帧供因果时间卷积使用。
   - 块内全注意力（block 内并行生成）+ 块间线性状态递归，实现分钟级视频。
4. **深度压缩视频 VAE** 与数据过滤管线（论文另两大贡献，本文档不展开）。

### 2.2 关键实测数字

- 训练成本：**64×H100 训练 12 天**，论文称为 MovieGen 成本的约 1%。
- 480p、81 帧视频生成延迟约 **60 s**：吞吐比 MAGI-1 快 7.2×、比 Step-Video 快 4× 以上。
- VBench：T2V Total **83.71**（与 Open-Sora-2.0 14B 相当，优于 Wan2.1-1.3B）；I2V Total **88.02**（优于 Wan2.1-14B、HunyuanVideo-I2V 11B）。
- 与同级小模型（Wan 2.1-1.3B、SkyReel-V2-1.3B）相比：质量有竞争力，**实测延迟快 16×**。
- 消费级部署：RTX 5090 + NVFP4，5 秒 720p 生成 **71 s → 29 s（2.4×）**。
- 支持分辨率最高 720×1280、分钟级时长。

### 2.3 局限性

- 纯线性注意力（即使有卷积补偿）表达力受固定状态秩限制，注意力图相比 Softmax「更弥散、局部聚焦更弱」（论文自陈观察）；质量上限依赖小模型定位，未在数十亿参数规模证明。
- 分钟级长视频依赖块自回归，块间接缝/累积误差问题论文着墨不多；时间卷积 FFN 的开销随时长增长（这一点由 SANA-Video 2.0 明确指出并移除）。
- 许可证为 BY-NC-ND（非商业）。

### 2.4 与其他工作的关系

- 直接前身是 SANA（图像线性 DiT）；是**首个把纯线性 DiT 做成长视频（分钟级、恒定 KV cache）的代表作**，为 SANA-Video 2.0 的混合化和 SANA-WM 奠定基础。
- 与 Kimi/KDA 路线的区别：SANA-Video 用简单 ReLU 核 + 卷积，不使用 delta 规则与数据相关门控。

---

## 3. SANA-Video 2.0

- **论文**：SANA-Video 2.0: Hybrid Linear Attention with Attention Residuals for Efficient Video Generation（arXiv:2607.21553，2026-07；13 页）
- **机构**：与 1.0 同一团队：MIT（Song Han）、NVIDIA、HKU（Ping Luo）、Enze Xie 等。

### 3.1 核心机制

统一架构下的 **5B 与 14B（14.25B，40 层、宽度 4096，在 384 张 B200 上训练）**，从零训练（from scratch）而非线性化预训练模型。

1. **Hybrid Linear-Softmax Attention（3:1）**：
   - 75% 层使用**门控双线性（gated bilinear）线性注意力**（双向、非因果、**不含 delta 规则**）：写入门 $\alpha$ 控制状态写入、输出门控制读出，Q/K 归一化并加 RoPE：
     $$o_i = \frac{\sum_j \alpha_j\,\tilde q_i^\top \tilde k_j\, v_j}{\sum_j \alpha_j\,\tilde q_i^\top \tilde k_j}$$
   - 每第 4 层插入一个**门控 Softmax 锚点层**（双向 SDPA + QK Norm + RoPE + sigmoid 输出门，输出门据 Qiu et al. 2025），周期性恢复线性状态无法表达的高秩 token 交互。线性/Softmax 头使用不同 head 维度。
2. **Block Attention Residuals（AttnRes）——深度方向的信息路由（第二核心）**：
   - 普通加性残差迫使后面的层重新推导前面算过的信息。AttnRes 把已完成的块级特征摘要跨深度路由：层按组（默认 8 层一块）聚合，已完成块留一个 summary，进行中的块保留 running partial sum；路由源 = 初始 embedding + 各已完成块摘要 + 当前块部分和。
   - 视频适配：① 跨深度**共享路由 query**（注意力分支、FFN 分支各一个，而非每层独立），损失几乎不变（0.496 vs 0.495），显存开销从 +16% 降到 +4%；② 去掉显式时间步输入（消融发现 AdaLN 已使其冗余，去掉后 loss 0.962→0.920）。
   - 实测：同 checkpoint 开关 AttnRes，**深层状态的有效秩（effective rank）提升约 12%**；路由历史从 $O(L)$ 压到约 $O(L/8)$。
3. **架构搜索**：低分辨率（256p）代理实验扫 0%/14.3%/25%/50%/100% Softmax：全线性 val loss 0.955 最差、全 Softmax 0.945、50% 最低 0.897 但 1080p 延迟比 25% 高 1.29×，**25%（0.905）被选为 Pareto 拐点**——没有照搬语言模型的比例，而是在视频域重新实测。
4. **卷积-free SwiGLU FFN**：取代 1.0 的时间卷积 FFN（论文指出其开销随时长增长），便于内核融合。
5. 训练全流程：多源数据治理、分辨率/时长课程、结构化字幕、自蒸馏（Self-Flow）、DPO + ReFL 偏好后训练；token 数感知的 flow-shift、TQD 内容感知时间步路由等（详见原文）。

### 3.2 关键实测数字

- **40 步采样：480p 单卡 H100，VBench Total 84.30，延迟 13.2 s**（摘要数字）；主表中 5B 模型 81 帧 Total 84.30、Quality 85.61（表中最高）、Semantic 79.05。
- 编译后 DiT 前向在 720p/60s（1441 帧）形状下比匹配的全 Softmax 基线快 **3.2×**，且时长越长差距越大（全线性/混合的前向提速随时长从约 2× 上升）。
- **Sol-Engine 全栈优化（内核融合 + 缓存 + 稀疏注意力）再提速 3.58×**：5B 管线 720p/5s 达 **13.06 s**，论文称为单 H100 上比 Wan 2.2-A14B 快 **120×**。
- 另有 50 步 B200 部署的端到端提速（具体倍数见原文，摘要页未给单一数字）。
- 量化：QAT 下 MXFP4 权重 + MXFP8 激活，VBench Total 83.25 vs BF16 83.22（持平）；PTQ 82.87。
- 121 帧 / 193 帧长配置 Total 分别另有报告（附录，摘要未给数字）。

### 3.3 局限性

- 作者自述：架构选择依赖短程低分辨率代理研究；最长时长结果是张量形状层面的 profile，训练课程只到 8 秒，分钟级「真正长视频」生成尚未验证。
- 线性算子是双向、无 delta 规则的，不适合因果/流式/世界模型场景（论文自陈，留作未来工作：与因果 Gated DeltaNet 结合）。
- AttnRes 本身在成熟模型上对 MSE/质量的直接增益很小（0.48506 vs 0.48547，20 个噪声桶中 17 个略优），主要价值是跨层复用与有效秩，而非直接提分。
- 14B 训练需 384×B200，门槛高。

### 3.4 与其他工作的关系

- 直接把 **Qwen3-Next / Kimi Linear 在 LLM 中验证的 3:1 混合范式**与 **Kimi K3 的 Attention Residuals** 迁移并适配到双向视频扩散，但在视频域重新做了比例扫描（结论恰好也是 25%）。
- 相对 1.0：从「纯线性 + 卷积」升级为「门控线性 + Softmax 锚点 + 深度路由」，并把规模推到 14B。
- Sol-Attn/Sol-Engine 是其推理端的稀疏化组件（见下节），二者构成同团队「架构 + 引擎」组合拳；与 VDN 是同代竞品（从零训练 vs 后装对齐）。

---

## 4. Sol-Attn

- **论文**：Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention Sparsification（arXiv:2607.24027，2026-07；technical report）
- **机构**：MIT（Song Han）、NVIDIA、HKU 等；作者包括 SANA-Video 团队成员（Junsong Chen、Enze Xie 等），属 SANA/Sol 体系。**Training-free（无需训练、无需微调）**，当前仅支持前向推理。

### 4.1 核心机制

针对现有训练后动态稀疏注意力的两个痛点：① 路由僵化/不可控/有开销（固定 top-k 预算或 top-p 累积概率导致预算失衡，且要物化 proxy 分数）；② 非选中块被整体丢弃，激进稀疏时掉点。Sol-Attn 把**动态路由 + 稀疏计算 + 近似校正统一在单次 online-softmax 遍历**中：

1. **查询相关阈值路由（Query-Dependent Thresholding）**：
   - 实证发现聚合后的块级 proxy logits 在每个模型内近似高斯分布。对 query 块 $q$，设其 proxy 行的均值 $\mu_q$、标准差 $\sigma_q$，给定共享的标准化截断系数 $c$：
     $$\tau_q = \mu_q + c\,\sigma_q,\qquad \mathcal I_q = \{b : s_{qb} \ge \tau_q\}$$
     $c$ 控制模型级平均稀疏度（$\Phi(c)$ 对应选中密度），每个 query 按自身尺度映射阈值 → **预算动态且密度可控**，避免了 top-k 的固定预算与 top-p 的预算失衡。
   - 阈值无需物化整张 proxy 图：proxy 是 pooled-key 的线性投影，均值/方差可直接由 pooled-key 的一阶、二阶矩计算，$O(N)$ 时间与辅助存储。
2. **块级在线（on-the-fly）阈值化**：把 pooled-key 序列分 chunk 顺序扫描，边算 proxy 边按阈值选块，数学上等价于离线全局阈值，但不物化 proxy 图与路由索引；形成「外层稠密扫描判选、内层稀疏精确计算」的嵌套循环。
3. **Proxy 分数复用为近似校正（区别于 keep-or-drop）**：未选中块不丢弃，而对其块内指数分数矩阵在块内行均值处做 Taylor 展开，保留零阶项——用 pooled key 近似整个块对 Softmax 分子与分母的贡献。关键工程点：近似项的 token-to-block 分数 tile 的列均值恰好就是路由用的 chunk proxy 分数，**同一块 tile 同时用于路由、近似校正与精确计算**，三者在一次 online-softmax 通过中融合。

### 4.2 关键实测数字

- 摘要：视频生成端到端 **2.1× 提速**、视频编辑 **2.3× 提速**，视觉质量保持。
- T2V 主表（匹配约 85% 稀疏度，端到端相对 FA3）：
  - **Wan 2.1-14B（720p，81 帧）**：稀疏度 85.12%，提速 **2.02×**；VBench AVG 76.13（FA3 为 75.90，略高），与稠密输出相似度 LPIPS 0.2086（各方法中最好）。
  - **HunyuanVideo 13B（720p，129 帧）**：稀疏度 85.80%，提速 **2.12×**；VBench AVG 76.81 vs FA3 77.06（基本持平）。
  - **LTX 2.3-22B（1080p，361/721 帧）**：稀疏度约 89.8%，提速 **1.9×（2.4×，长设置）**；VBench AVG 74.69 vs 74.59。
- V2V：SANA-WM Refiner 上 **3.04×**（稠密相似度最好：LPIPS 0.343）；Bernini 编辑 **2.34×**（Bernini-Bench 质量最高）。
- 在消费级 GPU、2K 文生图（Ideogram 4）上也有提速（具体数字见原文表 4）。
- 对比基线：XAttn、Sparse VideoGen2（SVG2）、PISA——在同等稀疏度下 Sol-Attn 提速最高且质量/相似度最好。

### 4.3 局限性（论文自述）

- B200 内核尚未充分榨干 Blackwell 性能；只支持前向、不支持自回归视频生成（评测仅限双向扩散式视觉生成）。
- 训练后方法，不改变模型权重，因此无法获得架构级的渐近复杂度改进；稀疏度进一步拉高时零阶近似仍有误差（附录给出误差分析）。

### 4.4 与其他工作的关系

- 是 SANA-Video 2.0 论文中「Sol-Engine 再提速 3.58×」里稀疏注意力组件的独立技术报告；与 Sparse VideoGen、SpargeAttn、PISA 同属训练后稀疏注意力，核心差异化是**阈值路由（vs top-k/top-p）+ 丢弃块零阶近似校正（vs keep-or-drop）+ 单 pass 融合**。
- 与 SLA 的区别：Sol-Attn 无需训练、在线单 pass；SLA 是可训练的稀疏-线性融合。

---

## 5. Gated DeltaNet（含 DeltaNet 家族脉络）

- **论文**：Gated Delta Networks: Improving Mamba2 with Delta Rule（arXiv:2412.06464，2024-12；**ICLR 2025**）
- **机构**：**NVIDIA**（Songlin Yang 在 NVIDIA 实习期间工作；Jan Kautz、Ali Hatamizadeh）。代码 github.com/NVlabs/GatedDeltaNet。

### 5.1 核心机制

观察到两类记忆机制互补：**门控（gating）实现快速记忆擦除，delta 规则实现精准的定向更新**。

- 先回顾：Mamba2 用数据相关标量衰减 $\alpha_t$ 统一衰减全部键值关联，$S_t=\alpha_t S_{t-1}+k_t v_t^\top$——遗忘是无差别的；DeltaNet 用 delta 规则 $S_t = S_{t-1} + k_t(v_t - k_t^\top S_{t-1})$（软替换），能精准改写单个关联，但缺少快速清场能力，真实任务表现一般。
- **Gated Delta Rule（二者统一）**：
  $$S_t = \alpha_t S_{t-1} + \beta_t k_t\big(v_t - k_t^\top(\alpha_t S_{t-1})\big)$$
  $\alpha_t\to 0$ 时快速清空记忆（上下文切换）；$\alpha_t=1$ 时退化为纯 delta 规则做选择性更新。$\beta_t$ 为写入强度。
- 在线学习视角：该规则是带正则项的在线回归目标的（近似）闭式解；也可视为 fast weight 矩阵在测试时做一步 SGD。
- **硬件高效的 chunkwise 并行训练算法**：把序列切块，块内用带衰减感知因果掩码的矩阵形式（大量 matmul，可吃 tensor core），块间传递状态——线性时间训练。
- 混合架构：Gated DeltaNet 层 + 滑窗注意力或 Mamba2 层，进一步提效提分。

### 5.2 关键结果

- 论文为语言模型工作：在语言建模、常识推理、上下文检索（in-context retrieval）、长度外推、长上下文理解等基准上**一致优于 Mamba2 与 DeltaNet**。摘要页未给出单一速度数字（论文未在摘要报告具体倍数）。

### 5.3 局限性

- 标量（head 级）遗忘门：对同一头内所有通道一视同仁，记忆容量仍受状态维度限制（维度内正交键值对数量上限导致「记忆碰撞」）。这一局限直接催生 KDA 的通道级门控。
- 主要在语言任务验证，本身不是视频模型——但其规则被 Video DeltaNet、KDA、SANA-Video 2.0（作为初始化方向）大量引用，是视频混合注意力的「上游算子」。

### 5.4 家族后续

- **Gated DeltaNet-2（arXiv:2605.22791，NVIDIA，2026-05）**：指出一个标量门同时管「键侧擦除多少旧内容」和「值侧提交多少新内容」是耦合的；Gated Delta Rule-2 用**通道级擦除门 $b_t$ 与通道级写入门 $w_t$ 解耦**，并继承 KDA 的通道级衰减；在两个门塌缩为同一标量时退化为 KDA，衰减也塌缩时退化为 Gated DeltaNet。1.3B 参数 / 100B FineWeb-Edu token 上，在 Mamba-2、Gated DeltaNet、KDA、Mamba-3 中总体最强，RULER 多键检索（needle-in-a-haystack）优势最明显。

---

## 6. KDA：Kimi Delta Attention（Kimi Linear）

- **论文**：Kimi Linear: An Expressive, Efficient Attention Architecture（arXiv:2510.26692，2025-10；technical report）
- **机构**：**月之暗面 Kimi Team（Moonshot AI）**。开源 KDA 内核与 vLLM 实现，放出预训练/指令模型 checkpoint。
- 说明：任务书所列「KDA（kernel design agents）」实际应为 **Kimi Delta Attention**；未检索到名为 "Kernel Design Agents" 的注意力论文，此处按 KDA 原始出处撰写。

### 6.1 核心机制

- **KDA = Gated DeltaNet 的细粒度门控升级版**：GDN/Mamba2 用粗粒度的 **head 级标量遗忘门**；KDA 引入**通道级（channel-wise / 对角化）遗忘门 $\mathrm{Diag}(\alpha_i)$**，每个特征维度独立衰减（类似 GLA），更有效地利用有限的有限状态 RNN 记忆，并增强位置感知：
  $$S_t = \big(I - \beta_t k_t k_t^\top\big)\,\mathrm{Diag}(\alpha_t)\, S_{t-1} + \beta_t v_t^\top,\qquad o_t = S_t^\top q_t$$
- **专用 chunkwise 算法**：用 DPLR（对角 + 低秩）转移矩阵的特化变体，相比通用 DPLR（Mamba/Mamba2 式）显著减少计算，同时更贴近经典 delta 规则；通过 **WY 表示**把一串秩-1 更新压缩为紧凑表示（避免额外矩阵求逆），并用 **UT transform** 把非 matmul 运算转换掉以提升硬件利用率；块级状态更新形如：
  $$S_{t+1} = \mathrm{Diag}(\gamma_t) S_t + K_t^\top (U_t - W_t)$$
- **Kimi Linear 架构**：3B 激活 / 48B 总参数，KDA 与 MLA（Multi-Head Latent Attention）层做 **3:1 层间混合**（与 Qwen3-Next 同比例、不同算子）。

### 6.2 关键实测数字

- 相同训练配方下，Kimi Linear 在全部评测任务上以明显 margin **优于全 MLA 基线**；
- **KV cache 占用最多降低 75%**；
- **1M 上下文解码吞吐最高达 6×**。

### 6.3 局限性

- 语言模型与 agent 场景的结果，未直接验证视频扩散；擦除与写入仍共用标量门（该点由 NVIDIA Gated DeltaNet-2 批评并改进）。
- MoE/大总参数形态（48B 总参）部署仍重。

### 6.4 与其他工作的关系

- 上接 Gated DeltaNet（算子源头）与 GLA（通道级门思想），下被 **SANA-Video 2.0、Video DeltaNet 直接引用/继承**：VDN 线性分支的特征处理与门控明确标注「Following Gated DeltaNet and Kimi Delta Attention」。KDA + 3:1 混合 + Attention Residuals（Kimi K3）构成了 2025–2026 混合架构向视频领域迁移的完整模板。

---

## 7. SageAttention 系列（量化注意力，正交加速维度）

均出自**清华大学（thu-ml；张锦涛等）**，plug-and-play、无需训练。

### 7.1 SageAttention（arXiv:2410.02367，ICLR 2025）

- **机制**：首个系统验证「注意力可量化」的工作。Q、K 以 per-block（后融合 ROPE）粒度量化到 **INT8**；$\widetilde P$（softmax 概率，利用其行最大值为 1 的性质用静态尺度 1/127）与 V 量化到 INT8（per-block/per-channel）。基于 FlashAttention2 的 block tiling，并有量化融合技巧。
- **数字**：OPS 约为 **FlashAttention2 的 2.1×、xformers 的 2.7×**；精度优于 FlashAttention3；在语言、图像、视频生成模型上端到端指标几乎无损。

### 7.2 SageAttention2（arXiv:2411.10958，ICML 2025）

- **机制**：推向 4-bit——**Q、K 在 warp（thread）级粒度量化到 INT4**；$\widetilde P$、V 用 **FP8**；对 Q 做 smoothing 提升 INT4 $QK^\top$ 精度（异常值平滑）；$\widetilde PV$ 两级累加保精度；并按层/时间步自适应（难的层与时间步回退 INT8+FP8）。论文给出 CogVideoX 直接 INT4 全量化会完全糊片的反例。
- **数字**：RTX4090 上 OPS 约为 **FlashAttention2 的 3×、xformers 的 4.5×**；Hopper 上追平 FlashAttention3(FP8) 速度但精度更高；端到端指标可忽略损失。

### 7.3 SageAttention3（arXiv:2505.11594，2025）

- **机制/数字**：利用 Blackwell GPU 的 **FP4 Tensor Core，microscaling FP4 注意力**，RTX 5090 上达 **1038 TOPS、比该卡上最快的 FlashAttention 快 5×**，plug-and-play 加速多类模型推理；并首次探索前后向都用的 **8-bit 注意力训练**——微调任务无损，但预训练收敛更慢。
- 另有 SageAttention2++（arXiv:2505.21136）为 2 的更高效实现。

### 7.4 局限与定位

- 量化是与线性化/稀疏化**正交**的维度：不降低 $O(N^2)$ 渐近复杂度，靠低 bit 提吞吐；激进低 bit 在异常值分布特殊的层/时间步仍需回退；8-bit 训练尚不能无损预训练。
- 在视频注意力加速谱系中，SageAttention 是被 SANA、VDN 等系统工作叠加使用的底层内核选项（VDN 即用 MXFP8 加速宽 GEMM）。

---

## 8. SLA / SLA2（稀疏-线性融合，可训练路线补充）

- **SLA**（arXiv:2509.24006，2025-09，清华大学 thu-ml 等；含 Sparse VideoGen 作者 Haocheng Xi）：
  - **发现**：注意力权重可分成一小部分高秩的大权重与其余极低秩权重 → 前者稀疏加速、后者低秩（线性）加速。
  - **机制**：把注意力权重分为 critical / marginal / negligible 三类，分别走 $O(N^2)$ 精确注意力、$O(N)$ 线性注意力、直接跳过；单 GPU 内核融合，支持前向与反向，少量微调即可。
  - **数字**：注意力计算量减少 **95%（约 20×）而不损失端到端生成质量**；Wan2.1-1.3B 上注意力内核 **13.7× 提速**、视频生成端到端 **2.2×**。
- **SLA2**（arXiv:2602.12675，2026）：引入可学习路由（learnable routing）与 QAT，视频扩散模型上可达 **97% 稀疏度、注意力 18.6× 提速**（端到端数字见原文）。
- **定位**：SLA 把「稀疏」和「线性」在**同一层内按权重秩**融合，是介于纯稀疏（Sol-Attn）与层间混合（SANA-Video 2.0）之间的第三条混合粒度，且需要训练；与 SageAttention 同实验室，方法谱系相通。

---

## 9. 技术演进脉络总结

```
Softmax Attention  O(N²) ── FlashAttention 系列（IO 感知精确内核，不改复杂度）
                                   │
线性化（渐近复杂度）                │ 量化（正交维度）
Linear Attn(2020, Katharopoulos)   SageAttention 8bit → SA2 4bit → SA3 FP4 / 8bit 训练
   │  + 数据相关门控
GLA(通道门) / DeltaNet(delta 规则)
   └─ Mamba2(标量衰减) ──► Gated DeltaNet（NVIDIA, ICLR'25：门控×delta 统一）
                                   │
                    ┌──────────────┼───────────────────────────────┐
              通道级门控 KDA                        擦/写解耦 Gated DeltaNet-2
        (Kimi, 2025-10)                                     (NVIDIA, 2026-05)
              │ + 3:1 层间混合 + AttnRes
              ▼
  向视频扩散迁移（2025-09 → 2026-09）
  ┌──────────────────────┬──────────────────────────┬──────────────────────────┐
SANA-Video            SANA-Video 2.0              Video DeltaNet
纯线性 ReLU+RoPE      门控线性 75% + Softmax 锚 25% 局部Softmax+双向线性记忆
+常量 KV cache(分钟级) + Block AttnRes 深度路由       + VDA 帧级 delta 规则
16× 延迟 / 1% 成本    120×(全栈) / VDN 竞品           14.5×(8×B200,6.70s/14.3s 768p)
                      │ 推理端稀疏化
                      └─► Sol-Attn（training-free 在线阈值 + 零阶近似，2.1–3×）
  同层稀疏-线性融合：SLA（95% 计算削减，2.2× 端到端）→ SLA2（97%，可学习路由+QAT）
```

**脉络要点：**

1. **算子层面**：线性注意力经历了「外积累加 → 数据相关门控（GLA/Mamba2）→ delta 规则精准改写（DeltaNet）→ 门控+delta 统一（Gated DeltaNet）→ 通道级细粒度门控（KDA）→ 擦除/写入门解耦（Gated DeltaNet-2）」的清晰升级链，主线是**在固定大小状态内更精细地管理遗忘、擦除与写入**，缓解「记忆碰撞/秩瓶颈」。
2. **架构层面**：纯线性（SANA-Video，2025-09）证明了线性 DiT 在视频上可行且极快，但表达力受限；2026 年中起全面转向**混合**——LLM 验证的 3:1（25% Softmax）配比被 Kimi、Qwen 与 SANA 团队不约而同采用，并通过**跨深度路由（Attention Residuals）**让少量 Softmax 锚点的高秩信息服务全网络（SANA-Video 2.0）；Video DeltaNet 则给出**层内按时距角色切分 + 帧级 delta 规则**的另一种混合粒度，并解决预训练基座的后装对齐问题。
3. **稀疏路线分化**：训练后动态稀疏（Sparse VideoGen、SpargeAttn、PISA → **Sol-Attn**）追求即插即用，创新点从「选哪些块」转向「阈值路由 + 丢弃块近似校正 + 单 pass 融合」；可训练路线（**SLA/SLA2**）则把稀疏与线性按权重秩统一到同层。
4. **系统与算法协同**：所有极致数字（120×、14.5×）都是「高效架构 + 少步蒸馏 + 低比特（FP4/FP8/量化注意力）+ 多卡并行 + 融合内核/引擎（SGLang、Sol-Engine）」的全栈结果；单独看注意力算法本身的收益通常在 2–3.5×。
5. **待解决的开放问题**：① 混合比例与锚点位置目前靠低分辨率代理经验选择（25% 是经验拐点而非理论最优），内容自适应放置是明确方向；② 双向（扩散）与因果（流式/自回归/世界模型）算子尚未统一，delta 规则的因果混合是各方共同指出的下一步；③ 长时长（分钟级以上）训练与一致性仍缺充分验证；④ 量化注意力的 8-bit 预训练、4-bit 极端稀疏下的精度仍未完全解决。

---

### 附：论文索引

| 简称 | 标题 | arXiv | 时间 | 机构 |
|---|---|---|---|---|
| VDN | Video DeltaNet | 2609.20744 | 2026-09 | Berkeley 系 / Impossible, Inc.（摘要页未署名，以全文为准） |
| SANA-Video | SANA-Video: Block Linear DiT | 2509.24695 | 2025-09 | MIT / NVIDIA / HKU / U. Toronto |
| SANA-Video 2.0 | Hybrid Linear Attention w/ AttnRes | 2607.21553 | 2026-07 | MIT / NVIDIA / HKU |
| Sol-Attn | On-the-Fly Attention Sparsification | 2607.24027 | 2026-07 | MIT / NVIDIA / HKU |
| Gated DeltaNet | Gated Delta Networks | 2412.06464 | ICLR 2025 | NVIDIA |
| Gated DeltaNet-2 | Decoupling Erase and Write | 2605.22791 | 2026-05 | NVIDIA |
| KDA / Kimi Linear | Kimi Delta Attention | 2510.26692 | 2025-10 | 月之暗面 Kimi Team |
| SageAttention | Accurate 8-Bit Attention | 2410.02367 | ICLR 2025 | 清华大学 |
| SageAttention2 | Per-thread INT4 | 2411.10958 | ICML 2025 | 清华大学 |
| SageAttention3 | Microscaling FP4 / 8-bit 训练 | 2505.11594 | 2025 | 清华大学 |
| SLA | Fine-Tunable Sparse-Linear Attn | 2509.24006 | 2025-09 | 清华大学等 |
| SLA2 | Learnable Routing + QAT | 2602.12675 | 2026 | 清华大学等 |
