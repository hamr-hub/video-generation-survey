# 注意力机制的数学原理：从 Softmax 到线性 / Delta / 随机特征 / 视频帧级解法

> 记号约定（全篇尽量统一）：输入序列长度 $N$（视频里一帧有 $U$ 个空间 token）；注意力头维度 $d_k=d$（query/key）与 $d_v$（value）；$Q,K,V\in\mathbb R^{N\times d}$ 分别为 query/key/value，第 $t$ 行（列向量形式）记 $q_t,k_t,v_t$；$S_t\in\mathbb R^{d_k\times d_v}$ 为矩阵型循环记忆状态（fast weights）。下文对每式给出出处；无法逐字核对之处明确标注【待核】。

---

## 1. Softmax Attention：定义与二次复杂度

**出处**：Vaswani et al., *Attention Is All You Need*, NeurIPS 2017（Eq. 1）。

单头缩放点积注意力：

$$
\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,
\qquad
o_i=\frac{\sum_{j=1}^{N}\exp\!\bigl(q_i^\top k_j/\sqrt{d_k}\bigr)\,v_j}
{\sum_{j=1}^{N}\exp\!\bigl(q_i^\top k_j/\sqrt{d_k}\bigr)}.
$$

**变量含义**：$q_i,k_j\in\mathbb R^{d_k}$，$v_j\in\mathbb R^{d_v}$；$1/\sqrt{d_k}$ 是缩放因子，防止维数大时点积过大使 softmax 梯度消失；因果（causal）场景在 softmax 前对上三角置 $-\infty$，即 $j\le i$。

**复杂度**：注意力分数矩阵 $QK^\top\in\mathbb R^{N\times N}$ 显式存在，时间与显存均为 $O(N^2 d)$（分数矩阵本身占 $O(N^2)$）。这是长序列（尤其视频）的主要瓶颈——Video DeltaNet 报告在 MiniMax H3 负载中 softmax 注意力占去噪运行时间 85% 以上（arXiv:2609.20744, §1）。

**数学结构上的两个关键点**（后文反复用到）：
1. softmax 权重是**输入相关（data-dependent）且全秩**的：每个 query 对全部 key 做非线性归一化，注意力矩阵可以是满秩的，因此支持精确的内容寻址检索；
2. 分母把所有 key 耦合在一起，无法拆成“先对 key/value 聚合、再与 query 作用”的形式——这正是它不能利用矩阵乘法结合律变线性的原因。

---

## 2. 线性注意力：核特征映射 + 结合律，$O(N^2)\to O(N)$

**出处**：Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*, ICML 2020, arXiv:2006.16236（Eq. 4–6）。

### 2.1 核化与特征映射

若相似度可写成核函数 $\operatorname{sim}(q,k)=\phi(q)^\top\phi(k)$，其中特征映射 $\phi:\mathbb R^d\to\mathbb R^m$（$m$ 与 $N$ 无关），则去掉 softmax 的归一化耦合后：

$$
o_i=\frac{\sum_{j=1}^{N}\phi(q_i)^\top\phi(k_j)\,v_j}
{\sum_{j=1}^{N}\phi(q_i)^\top\phi(k_j)}
=\frac{\phi(q_i)^\top\displaystyle\sum_{j=1}^{N}\phi(k_j)v_j^\top}
{\phi(q_i)^\top\displaystyle\sum_{j=1}^{N}\phi(k_j)}.
$$

要求核非负（分母才有意义），原论文取 $\phi(x)=\operatorname{elu}(x)+1$（Eq. 7）；后续工作常用 $\operatorname{relu}$、sigmoid 等。

### 2.2 结合律：两种括号化顺序

写成矩阵形式（无归一化的“线性注意力”）：

$$
(\Phi(Q)\Phi(K)^\top)\,V \;=\; \Phi(Q)\,(\Phi(K)^\top V).
$$

- 左边：先算 $\Phi(Q)\Phi(K)^\top$ 得 $N\times N$ 矩阵，代价 $O(N^2 d)$；
- 右边：先算 $\Phi(K)^\top V\in\mathbb R^{m\times d_v}$（与 $N$ 无关大小的中间量），再被 $\Phi(Q)$ 左乘，代价 $O(N\,m d_v)$。

**对序列长度 $N$ 由二次降为线性**（固定头维度时）。这就是“核特征 + 矩阵乘法结合律”的全部要点：softmax 的非线性归一化不可结合，而内积核可以。

### 2.3 循环递推：linear attention 即 RNN / fast weight

**出处**：同上文 §3.4（Eq. 16–20）；标题即“Transformers are RNNs”。

因果情形下把 $S_i=\sum_{j\le i}\phi(k_j)v_j^\top$ 增量维护：

$$
S_0=0,\qquad
S_t=S_{t-1}+\phi(k_t)v_t^\top,\qquad
o_t=S_t\,\phi(q_t)
$$

（带归一化时另维护 $z_t=z_{t-1}+\phi(k_t)$，$o_t=S_t\phi(q_t)/(\phi(q_t)^\top z_t)$）。

**变量与视角**：$S_t$ 是一个随时间更新的小矩阵，读出 $o_t=S_t\phi(q_t)$ 是一次以 $S_t$ 为权重矩阵的线性变换——Schlag 等（2021）据此称线性注意力为“秘密的 fast weight programmer”（快速权重程序员，Schmidhuber 1991 年提出的概念：一个慢网络快速改写另一个网络的权重）。自回归推理时状态大小与 $N$ 无关，无需保存 KV cache，故长序列生成可快数千倍。

**代价**：$S_t$ 只是外积的累加，是**无擦除**的关联记忆；能存的正交 key–value 对数受维度 $d_k$ 限制，超长序列必然“记忆碰撞”，无法精确检索（Gated DeltaNet, arXiv:2412.06464, §1 引述 Schlag 2021）。

---

## 3. Delta Rule：把“只加不擦”变成“读后改写”

**出处**：Schlag, Irie & Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers*, ICML 2021, arXiv:2102.11174；更新规则的现代形式见 Yang et al., *Gated Delta Networks*, arXiv:2412.06464, §2.3。

### 3.1 更新方程

$$
\boxed{\;
S_t=S_{t-1}-\beta_t\bigl(S_{t-1}k_t\bigr)k_t^\top+\beta_t v_t k_t^\top
=S_{t-1}\bigl(I-\beta_t k_t k_t^\top\bigr)+\beta_t v_t k_t^\top
\;}
$$

等价地：

$$
S_t=S_{t-1}+\beta_t\bigl(v_t-S_{t-1}k_t\bigr)k_t^\top,\qquad o_t=S_t q_t.
$$

**变量含义**：$S_{t-1}k_t$ 是当前记忆对 key $k_t$ 关联值的“旧预测” $v_t^{\text{old}}$；$v_t-S_{t-1}k_t$ 是预测误差（delta）；$\beta_t\in[0,1]$ 是写入门（自适应学习率），由网络对 $x_t$ 计算（通常 sigmoid）。第一项沿 $k_t$ 方向**擦除**旧关联（rank-one 擦除），第二项沿同一方向**写入**新值——即“erase-then-write”。

### 3.2 与线性注意力的关系 / 梯度下降视角

对单样本目标 $\ell(S)=\tfrac12\|S k_t-v_t\|_2^2$，有 $\nabla_S\ell=(S k_t-v_t)k_t^\top$，于是：

$$
S_t=S_{t-1}-\beta_t\nabla_S\ell(S_{t-1})
$$

**delta rule 就是对“让 $S k_t\approx v_t$”做一步学习率 $\beta_t$ 的梯度下降**，源自 Widrow–Hoff 1960 的 least-mean-squares / delta rule。线性注意力 $S_t=S_{t-1}+\phi(k_t)v_t^\top$ 是其“无误差项、只写入”的特例（去掉 $-S_{t-1}k_t k_t^\top$ 擦除项）。DeltaNet 通过定向改写缓解关联记忆的容量碰撞问题，in-context 检索能力显著强于朴素线性注意力。

### 3.3 并行化（训练时）

逐 token 递推不利于 GPU。Yang et al. (2024) 注意到擦除矩阵 $I-\beta kk^\top$ 是（广义）Householder 矩阵的乘积，可用 **WY representation**（Bischof & Van Loan, 1985）压缩成低秩形式做 chunkwise 并行；Gated DeltaNet 在此基础上加入门控项（arXiv:2412.06464, §2.3–§3）。

---

## 4. Gated DeltaNet：衰减门 + 写入门

**出处**：Yang, Kautz & Hatamizadeh, *Gated Delta Networks: Improving Mamba2 with Delta Rule*, arXiv:2412.06464（2024-12；文中与项目页有时标注 2025）。

### 4.1 门控 delta rule

$$
\boxed{\;
\bar S_t=\alpha_t S_{t-1},\qquad
S_t=\bar S_t+\beta_t\bigl(v_t-\bar S_t k_t\bigr)k_t^\top
=\alpha_t S_{t-1}(I-\beta_t k_t k_t^\top)+\beta_t v_t k_t^\top,\qquad
o_t=S_t q_t.
\;}
$$

- **衰减/遗忘门 $\alpha_t\in(0,1)$**：Gated DeltaNet 原文为**逐头（head-wise）标量**，受 Mamba2 的 $S_t=\alpha_t S_{t-1}+v_t k_t^\top$ 启发（对比 Mamba2 无差别衰减所有关联）；
- **写入门 $\beta_t\in[0,1]$**：控制 delta 修正强度；
- 实践中对 $q,k$ 做 L2 归一化（QKNorm）以保证稳定性，输出经 RMSNorm + SiLU/Sigmoid 输出门。

**互补性（论文核心论点）**：$\alpha_t\to0$ 可快速整体擦除记忆；$\alpha_t\to1$ 退化为纯 delta rule，做定向更新而不影响其他内容——“gating enables rapid memory erasure while the delta rule facilitates targeted updates”。

### 4.2 Kimi Delta Attention (KDA)：更细粒度的门控

**出处**：Kimi Team, *Kimi Linear*, arXiv:2510.26692（2025-10-30）。

KDA 扩展 Gated DeltaNet：把 head-wise 衰减换成 **channel-wise（逐特征维）衰减**，思路同 Gated Linear Attention (GLA, Yang et al. 2024)：

$$
\bar S_t=S_{t-1}\operatorname{Diag}(\alpha_t),\qquad
S_t=\bar S_t+\beta_t(v_t-\bar S_t k_t)k_t^\top,
$$

其中 $\alpha_t\in\mathbb R^{d_k}$，每个记忆特征方向有独立遗忘率。KDA 还用专门的**对角加低秩（DPLR）转移矩阵**形式实现 chunkwise 并行核（论文 §1，代码在 fla-org/flash-linear-attention）。RetNet（Sun et al. 2023）与 GLA 则是“门控线性注意力”路线（$S_t=\operatorname{Diag}(\gamma_t)S_{t-1}+k_t^\top v_t$，无 delta 擦除项），可视为 GDN 在去掉 delta 项时的对应物【RetNet 具体式按原论文，本处只作关系性陈述】。

---

## 5. Performer：用随机特征近似 softmax 核

**出处**：Choromanski et al., *Rethinking Attention with Performers*, ICLR 2021, arXiv:2009.14794；方法名 **FAVOR+**（Fast Attention Via positive Orthogonal Random features）。

### 5.1 softmax 也是核

未缩放的 softmax 核为 $\mathsf k_{\mathrm{sm}}(x,y)=\exp(x^\top y)$。改写：

$$
\exp(x^\top y)=\exp\!\left(-\tfrac12\|x-y\|^2\right)\exp\!\left(\tfrac12\|x\|^2\right)\exp\!\left(\tfrac12\|y\|^2\right),
$$

即 softmax 核 = 高斯（RBF）平移核乘以两边的模长修正。高斯核有随机傅里叶特征（RFF, Rahimi & Recht 2007）：对 $\omega\sim\mathcal N(0,I_d)$，

$$
\exp\!\left(-\tfrac12\|x-y\|^2\right)=\mathbb E_{\omega}\!\left[\cos(\omega^\top x)\cos(\omega^\top y)+\sin(\omega^\top x)\sin(\omega^\top y)\right].
$$

### 5.2 随机特征映射

用 $m$ 个随机投影 $\omega_1,\dots,\omega_m$ 构造（无偏三角特征）：

$$
\tilde\phi(x)=\frac{e^{-\|x\|^2/2}}{\sqrt m}
\bigl(\cos(\omega_1^\top x),\sin(\omega_1^\top x),\dots,\cos(\omega_m^\top x),\sin(\omega_m^\top x)\bigr),
$$

满足 $\mathbb E[\tilde\phi(x)^\top\tilde\phi(y)]=\exp(x^\top y)$。FAVOR+ 的关键改进是使用**正随机特征（positive random features）**与**正交随机矩阵**（各 $\omega_i$ 正交化），使 $\phi(x)\ge0$（配合 softmax 注意力的概率解释、降低估计方差）：

$$
\phi_+(x)=\frac{e^{-\|x\|^2/2}}{\sqrt{2m}}\Bigl(\underbrace{e^{\omega_1^\top x},\dots,e^{\omega_m^\top x}}_{\text{positive features}},\ \underbrace{e^{-\omega_1^\top x},\dots,e^{-\omega_m^\top x}}_{\text{对偶项}}\Bigr)
\quad\text{【正特征构造的等价写法之一，以原文 Lemma/附录定义为准】}.
$$

近似注意力：$\widehat{\operatorname{Attn}}=\hat D^{-1}\,\phi_+(Q)^\top\bigl(\phi_+(K)^\top V\bigr)$，其中 $\hat D=\operatorname{diag}(\phi_+(Q)^\top\phi_+(K)\mathbf 1)$。因特征映射对所有输入固定（**不随 query–key 对改变**），右侧括号化使复杂度为 $O(N m d+N d^2)$，对 $N$ 线性；$m$ 越大方差越小。

**与前述方法的关系**：Performer 走的是“**用随机特征把 softmax 核本身线性化**”的路线（保留对 softmax 的无偏近似），而 Katharopoulos 线性注意力/DeltaNet 走的是“**换一个确定的核/改写规则**”的路线——前者近似 softmax 但理论上是随机估计，后者精确但不再是 softmax。

---

## 6. Video Delta Attention (VDA)：从逐 token 更新到一帧的联合正规方程

**出处**：Xi et al., *Video DeltaNet: A Video-Native Hybrid Attention for Livestream Video Generation*, arXiv:2609.20744（v1, 2026-09-17），§2 与项目页 openvdn.github.io。以下 $A,B$ 定义与正规方程与项目页“Video Delta Attention: A Frame-Wise Solution”逐式核对一致。

### 6.1 出发点：一帧 $U$ 个空间 token 应联合更新

设第 $t$ 帧有 $U$ 个空间 token，$K\in\mathbb R^{U\times d_k}$、$V\in\mathbb R^{U\times d_v}$（第 $u$ 行为 $k_u^\top, v_u^\top$，key 已单位归一化）；$\beta_u$ 为各 token 写入门，$\alpha$ 为 channel-wise 衰减。沿用 KDA 的衰减先态：

$$
\bar S=S_{t-1}\operatorname{Diag}(\alpha).
$$

定义两个帧统计量（**这就是要求写出的 $A,B$**）：

$$
\boxed{\;
A=K^\top\operatorname{Diag}(\beta)K=\sum_{u=1}^{U}\beta_u k_u k_u^\top,\qquad
B=V^\top\operatorname{Diag}(\beta)K=\sum_{u=1}^{U}\beta_u v_u k_u^\top.
\;}
$$

$A\in\mathbb R^{d_k\times d_k}$ 是加权 key Gram（键自相关）矩阵，$A\succeq0$；$B\in\mathbb R^{d_v\times d_k}$ 是加权 value–key 写入矩阵。

### 6.2 朴素并行（SANA-WM 式）的问题

若所有 token 都从**同一个冻结的** $\bar S$ 读、彼此不见面地批量写（SANA-WM 用 $\hat K=K/U$ 缩放保稳定性）：

$$
S_{\mathrm{SANA}}=\bar S+\sum_{u=1}^{U}\beta_u(v_u-\bar S\hat k_u)\hat k_u^\top
=\bar S\bigl(I-\tfrac{A}{U}\bigr)+\tfrac{B}{U}.
$$

各 token 的写入互不协商：同帧内相似 key、不同 value 的冲突只会叠加；且 $1/U$ 是与 token 几何无关的全局缩放（重复 key 时证据量未被利用，正交 key 时每条独立写入被无过罚）。

### 6.3 帧级正则化目标与正规方程

VDA 要求“一个共享状态同时拟合一整帧”，并加邻近项守住旧记忆：

$$
\min_{S}\ \frac12\|S-\bar S\|_F^2+\frac12\sum_{u=1}^{U}\beta_u\|S k_u-v_u\|_2^2.
$$

对 $S$ 求导并令其为零（利用 $A,B$）：

$$
0=S-\bar S+\sum_{u=1}^{U}\beta_u(Sk_u-v_u)k_u^\top
  =S-\bar S+SA-B.
$$

这就是**帧级正规方程（normal equation）**：

$$
\boxed{\;S(I+A)=\bar S+B\;}
\qquad\Longrightarrow\qquad
\boxed{\;S_t=(\bar S+B)(I+A)^{-1}.\;}
$$

（注意状态按 $S k$ 读出，故 $S$ 在左、$A$ 右乘；这是转置约定下的标准最小二乘正规方程。）

**与逐 token delta 的关系**：单 token delta rule 是目标 $\ell(S)=\tfrac12\|S k-v\|^2$ 的一阶梯度步 $S=\bar S-\beta\nabla\ell$；VDA 不是对每个 token 各走一步冻结梯度，而是把整帧 $U$ 个拟合条件联立、加上二次邻近项后**精确求解**。统计量仍是同一套 $A,B$，区别在“联合解正规方程” vs “冻结态批量梯度步”。

### 6.4 两个推论（论文给出）

1. **继承状态天然稳定（non-expansive）**：$A\succeq0$，谱分解 $A=Q\Lambda Q^\top$，则
$$
M_{\mathrm{VDA}}=(I+A)^{-1}=Q(I+\Lambda)^{-1}Q^\top,\qquad
\|M_{\mathrm{VDA}}\|_2=\max_i\frac{1}{1+\lambda_i}\le 1.
$$
加衰减后 $\operatorname{Diag}(\alpha)(I+A)^{-1}$ 范数仍 $\le1$。故 VDA **不需要 SANA-WM 的 $1/U$ 帧大小缩放**；对比加法式转移 $I-A$ 在 $\lambda_i>2$ 时范数突破 1（这正是 SANA-WM 必须除以 $U$ 的原因），而 $1/(1+\lambda)$ 只单调趋零、永不越过。
2. **相关性感知**：Neumann 级数 $(I+A)^{-1}=I-A+A^2-A^3+\cdots$ 表明一阶项是加法批更新，高阶项通过共享 key 方向刻画写入间的交互（重复 key 取共识、正交 key 各得满权重）；直接求逆对任意 $A\succeq0$ 都有定义，不受级数收敛域限制。

VDN 整体为 **hybrid**：局部（跨时间块）双向 sliding-window softmax + 首尾帧 4-way boundary anchors 保全局参照，其余远端上下文由双向（前向/后向）线性记忆 $S_t^\rightarrow,S_t^\leftarrow$ 处理，文本 prompt 在初始化时以 delta rule 各写一半入两个状态（$S_0^\rightarrow=S_0^\leftarrow=\tfrac12 S_T$），两分支各有 RMSNorm、sigmoid 门与独立输出投影（arXiv:2609.20744, Eq. 1）。

---

## 7. Hybrid：为什么必须插入 softmax 锚点层来恢复高秩交互

**出处与依据**：
- SLA, arXiv:2509.24006：对扩散模型注意力图的经验观察——注意力权重可分为“一小部分大权重、**高秩（high-rank）且稀疏**”与“剩余低秩、稠密”两部分；据此用稀疏精确注意力 + 线性注意力分别承担；
- Kimi Linear, arXiv:2510.26692, §1：纯线性结构受**有限状态容量**根本制约，长序列建模与 in-context 检索理论上困难；故按 **3:1（KDA : 全局 MLA）均匀比例**插入完整 softmax 注意力层，保持全局信息流，同时 KV cache 最多降 75%；
- Video DeltaNet, arXiv:2609.20744：固定大小状态无法随序列增长保留全部细粒度交互，直接替换会掉质量；故保留局部 softmax 与边界锚点。

### 7.1 秩的角度（数学直觉）

线性注意力输出可写为 $o_i=\phi(q_i)^\top S$，其对全体 token 的等效注意力/算子由固定大小的状态 $S\in\mathbb R^{m\times d_v}$ 中介：

- 任何经单一线性层的交互秩受状态内维度（$m$、头维度）限制，是**低秩瓶颈**；softmax 的 $N\times N$ 注意力矩阵则可随内容做到**满秩**，允许 query 与每个 key 直接、非线性、成对地精确交互；
- 多个纯线性层堆叠虽能提高整体表达，但难以在“需要精确比较两个特定位置”的任务（检索、复制、主体一致性/场景布局）中替代一次直接的高秩 softmax 交互（Jelassi et al. 2024、RALA 等“线性注意力低秩困境”的相关论述【具体定理按各原文】）。

### 7.2 锚点层的作用

在以线性层为主的栈中周期性插入（或在视频里以滑窗/边界锚点形式保留）softmax 注意力，等价于提供：

$$
\text{线性层：低秩、压缩、}O(N)\text{ 的长程记忆}
\;+\;
\text{softmax 锚点：高秩、精确、内容寻址的 pairwise 交互}.
$$

锚点层把分散的位置重新“拉到同一高秩空间”直接比对，恢复被有限状态压缩掉的交互通道（如 Kimi 的全局层、VDN 的首/尾帧锚点与 StreamingLLM 式 attention sink, Xiao et al. 2023）；线性层则负责其余大部分算力。这样整体复杂度接近线性，而质量逼近全 softmax——这是当前 SLA/SLA2、Kimi Linear、Video DeltaNet 共同采用的结构原则。

---

## 附：方法关系一览

| 方法 | 状态转移（核心） | 对 $N$ 复杂度 | 关键性质 |
|---|---|---|---|
| Softmax (2017) | $\operatorname{softmax}(QK^\top/\sqrt d)V$ | $O(N^2)$ | 满秩、精确内容寻址，不可结合 |
| Linear Attention (2020, 2006.16236) | $S_t=S_{t-1}+\phi(k_t)v_t^\top$ | $O(N)$ | 核特征+结合律；RNN/fast weight；只加不擦 |
| DeltaNet (2021, 2102.11174) | $S_t=S_{t-1}+\beta(v_t-S_{t-1}k_t)k_t^\top$ | $O(N)$ | 读后擦写；= 误差梯度步 |
| Gated DeltaNet (2412.06464) | $\bar S=\alpha S_{t-1}$；$S_t=\bar S+\beta(v_t-\bar S k_t)k_t^\top$ | $O(N)$ | head-wise 衰减 + 写入门 |
| KDA / Kimi Linear (2510.26692) | 衰减改 channel-wise；DPLR chunkwise | $O(N)$ | 细粒度遗忘；3:1 插 MLA |
| Performer/FAVOR+ (2009.14794) | $\phi_+(Q)^\top(\phi_+(K)^\top V)$ | $O(N)$（随机近似 softmax） | 正正交随机特征，无偏估计 |
| VDA / Video DeltaNet (2609.20744) | $S_t=(\bar S+B)(I+A)^{-1}$，$A=K^\top\!\operatorname{Diag}(\beta)K$，$B=V^\top\!\operatorname{Diag}(\beta)K$ | 对帧数线性、帧内精确求解 | 整帧联合正规方程；天然 non-expansive |

**不确定 / 待核项**：
- Performer 正随机特征的具体闭式以原文 Theorem/附录定义为准（文中给出的是等价构造之一）；
- RetNet/GLA 仅作关系性陈述，未逐式照抄；
- “hybrid 恢复高秩”在 SLA 是经验分解（稀疏高秩 + 稠密低秩），在 Kimi/VDN 主要是有限状态容量论证与架构实践，“秩”的严格定理请参考 Jelassi 2024、RALA 等专门工作；
- arXiv:2609.20744 与 arXiv:2510.26692 版本日期较新，本笔记依据 v1 HTML 与项目页，若后续版本修订请以最新原文为准。
