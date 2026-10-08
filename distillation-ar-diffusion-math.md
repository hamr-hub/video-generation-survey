# 蒸馏/加速 与 自回归–扩散统一的数学原理

> 核对说明：以下公式均对照原文 HTML/PDF 抽取；凡论文未给闭式或为本文整理的推导步骤，均标注【本文推导】。DMD 式 (2) 严格说需要正则条件（log-density 对 θ 可微、可交换期望与微分），原文未强调，见局限。

---

## 1. Progressive Distillation（渐进蒸馏, Salimans & Ho, ICLR 2022, arXiv:2202.00512）

### 1.1 目标：N 步 → N/2 步
教师以 **确定性 DDIM**（概率流 ODE 的离散格式）采样。学生被初始化自教师，在每个离散时间点 $t=i/N$ 上，用教师走 2 步 DDIM（$t\to t'\to t''$，其中 $t'=t-\tfrac{0.5}{N}$, $t''=t-\tfrac{1}{N}$），再反解一步 DDIM 得到教师目标 $\tilde{\mathbf x}$：

$$\mathbf z_{t'} = \alpha_{t'}\hat{\mathbf x}_\eta(\mathbf z_t) + \frac{\sigma_{t'}}{\sigma_t}\bigl(\mathbf z_t-\alpha_t\hat{\mathbf x}_\eta(\mathbf z_t)\bigr),$$

$$\mathbf z_{t''} = \alpha_{t''}\hat{\mathbf x}_\eta(\mathbf z_{t'}) + \frac{\sigma_{t''}}{\sigma_{t'}}\bigl(\mathbf z_{t'}-\alpha_{t'}\hat{\mathbf x}_\eta(\mathbf z_{t'})\bigr),$$

$$\boxed{\;\tilde{\mathbf x}=\frac{\mathbf z_{t''}-(\sigma_{t''}/\sigma_t)\,\mathbf z_t}{\alpha_{t''}-(\sigma_{t''}/\sigma_t)\,\alpha_t}\;}$$

学生损失（加权去噪 MSE）：

$$\mathcal L_\theta = w(\lambda_t)\,\bigl\|\tilde{\mathbf x}-\hat{\mathbf x}_\theta(\mathbf z_t)\bigr\|_2^2,\qquad \lambda_t=\log\frac{\alpha_t^2}{\sigma_t^2}.$$

### 1.2 推导逻辑
- DDIM 单步 $\mathbf z_t\mapsto\mathbf z_s$ 是关于 $\hat{\mathbf x}$ 的仿射可逆映射；因此"教师 2 步复合"可被解析地反解为"学生 1 步所需预测的 $\tilde{\mathbf x}$"（原文 Appendix G）。
- 目标 $\tilde{\mathbf x}$ 在给定 $\mathbf z_t$ 后是确定的，比用原始数据 $\mathbf x$（给定 $\mathbf z_t$ 仍多模）更"尖锐"，学生不必回归到条件均值（模糊）。
- 学生收敛后成为新教师，$N\leftarrow N/2$，重复：8192→…→4 步。

### 1.3 变量含义
$\alpha_t,\sigma_t$：前向边缘 $\mathbf z_t=\alpha_t\mathbf x+\sigma_t\boldsymbol\epsilon$ 的信号/噪声尺度；$\lambda_t$：log-SNR；$\eta,\theta$：教师/学生参数；$N$：学生采样步数。

### 1.4 少步稳定性的参数化（原文另一贡献）
$\epsilon$-预测权重 $w=e^{\lambda_t}=\alpha_t^2/\sigma_t^2$ 在 $\alpha_t\to0$（纯噪声端）给零权重且除以 $\alpha_t$ 放大误差，不适合蒸馏。采用：
- 直接预测 $\mathbf x$；
- $v$-预测：$\mathbf v=\alpha_t\boldsymbol\epsilon-\sigma_t\mathbf x$，$\hat{\mathbf x}=\alpha_t\mathbf z_t-\sigma_t\hat{\mathbf v}_\theta$；
- 权重用 truncated-SNR：$w=\max(1,\alpha_t^2/\sigma_t^2)$，或 SNR+1：$w=1+\alpha_t^2/\sigma_t^2$。

### 1.5 局限
- **只适用于确定性采样器**：随机 DDPM 两步复合一般非高斯，单步学生族无法精确表示（原文 App. G 明确）。
- 学生只在离散时间点训练，逐步丧失连续时间泛化性；质量仍随步数下降；极端 1 步受限。

---

## 2. Guided Diffusion 的两阶段蒸馏（Meng et al., CVPR 2023, arXiv:2210.03142）

### 2.1 为什么不能直接蒸
Classifier-free guidance（CFG）每步要评两个网络：条件 $\hat{\mathbf x}_{c,\theta}$ 与无条件 $\hat{\mathbf x}_\theta$，

$$\hat{\mathbf x}_\theta^{w}=(1+w)\,\hat{\mathbf x}_{c,\theta}-w\,\hat{\mathbf x}_\theta,$$

教师单步成本已是 2 次网络求值；直接对"2 个模型 × DDIM"做渐进蒸馏目标不明确。

### 2.2 Stage 1：先把 CFG 合并进单模型
训练一个以引导强度 $w$ 为条件的学生 $\hat{\mathbf x}_{\eta_1}(\mathbf z_t,w)$（$w$ 经 Fourier embedding 注入），拟合 CFG 合并输出：

$$\mathcal L_{\text{stage1}}=\mathbb E_{\substack{w\sim p_w,\,t\sim U[0,1]\\ \mathbf x\sim p_{\rm data}}}\Big[\omega(\lambda_t)\,\big\|\hat{\mathbf x}_{\eta_1}(\mathbf z_t,w)-\hat{\mathbf x}_{\theta}^{w}(\mathbf z_t)\big\|_2^2\Big].$$

一个模型覆盖区间 $w\in[w_{\min},w_{\max}]$（实验 $[0,4]$），保留质量–多样性可调性。

### 2.3 Stage 2：再做渐进蒸馏
对 Stage-1 模型按第 1 节机制逐轮减半步数。原文还给出 $N$ 步随机采样器：用 2 倍步长做一次确定性更新，再用原步长做一次随机后向（加噪），兼顾多样性。

### 2.4 局限
- Stage-1 输出本质是对**引导后隐式分布**的 score 拟合，CFG 本身改变分布（过强 $w$ 产生低概率伪结构），误差被一并蒸入。
- 单 $w$ 区间模型对区间外推差；像素空间需 4–16 步才匹配教师，1 步 FID 明显掉。

---

## 3. DMD：Distribution Matching Distillation（Yin et al., CVPR 2024, arXiv:2311.18828）

### 3.1 目标：最小化加噪后假/真分布的近似 KL
一步生成器 $G_\theta$，$x=G_\theta(z)$, $z\sim\mathcal N(0,I)$。最小化

$$D_{\rm KL}(p_{\rm fake}\Vert p_{\rm real})=\mathbb E_{z}\Big[-\big(\log p_{\rm real}(x)-\log p_{\rm fake}(x)\big)\Big].\tag{1}$$

密度不可解，但只需对 $\theta$ 的梯度。

### 3.2 梯度 = 两个 score 之差（推导）
【本文推导，对应原文式 (2)】设 $x=G_\theta(z)$，push-forward 密度满足 $\nabla_\theta\log p_{\rm fake}(x)$ 通过链式法则作用；对光滑可逆/参数化情形：

$$\nabla_\theta D_{\rm KL}=\mathbb E_{z}\Big[-\big(\underbrace{\nabla_x\log p_{\rm real}(x)}_{s_{\rm real}}-\underbrace{\nabla_x\log p_{\rm fake}(x)}_{s_{\rm fake}}\big)\frac{\partial G_\theta(z)}{\partial\theta}\Big].\tag{2}$$

推导要点：
1. $\nabla_\theta\big[-\log p_{\rm real}(G_\theta(z))\big]=-s_{\rm real}(x)\,\partial_\theta G$；
2. 对 $\log p_{\rm fake}$ 项，利用 score 恒等式 $\mathbb E_{x\sim p_{\rm fake}}[\nabla_x\log p_{\rm fake}(x)\,\partial_\theta G]=\mathbb E[\nabla_\theta\log p_{\rm fake}(x)\partial_\theta G]$（log-derivative / 链式），得到 $+s_{\rm fake}(x)\,\partial_\theta G$；
3. 合并即 $-(s_{\rm real}-s_{\rm fake})\partial_\theta G$。

直观：$s_{\rm real}$ 把样本拉向真实众数，$-s_{\rm fake}$ 把样本从假分布挤开；二者之差是"更真实、更不像假货"的方向。

### 3.3 为什么要加噪 + 两个扩散模型
问题：(a) 假样本处 $p_{\rm real}$ 可能为 0、score 发散；(b) 扩散模型只给**加噪后分布**的 score。解法（Score-SDE）：以 $q_t(x_t|x)=\mathcal N(\alpha_t x,\sigma_t^2 I)$ 扰动使两分布全支撑重叠，用两个去噪器分别逼近：

$$s_{\rm real}(x_t,t)=-\frac{x_t-\alpha_t\,\mu_{\rm base}(x_t,t)}{\sigma_t^2},\qquad s_{\rm fake}(x_t,t)=-\frac{x_t-\alpha_t\,\mu_{\rm fake}^{\phi}(x_t,t)}{\sigma_t^2},$$

其中 $\mu_{\rm base}$ 为固定教师；假 score 网络 $\mu_{\rm fake}^\phi$ 在线跟踪生成分布，用去噪损失训练：

$$\mathcal L_{\rm denoise}^{\phi}=\|\mu_{\rm fake}^{\phi}(x_t,t)-x_0\|_2^2.$$

最终**近似分布匹配梯度**（对 $t$ 求期望，权重 $w_t$ 归一化梯度量级）：

$$\boxed{\;\nabla_\theta D_{\rm KL}\simeq \mathbb E_{z,t,x,x_t}\Big[w_t\,\alpha_t\,\big(s_{\rm fake}(x_t,t)-s_{\rm real}(x_t,t)\big)\frac{\partial G}{\partial\theta}\Big]\;}\tag{7}$$

$$w_t=\frac{\sigma_t^2}{\alpha_t}\frac{CS}{\|\mu_{\rm base}(x_t,t)-x\|_1},$$

$C,S$ 为通道数、空间点数；$t\in[0.02T,0.98T]$。

### 3.4 回归损失（DMD 必需）
小噪声端真实 score 不可靠，且 score 对密度归一化不变 → 易 mode collapse。用预生成 (噪声, 教师 ODE 输出) 对 $\mathcal D=\{z,y\}$ 做 LPIPS 回归保底：

$$\mathcal L_{\rm reg}=\mathbb E_{(z,y)\sim\mathcal D}\,\ell(G_\theta(z),y),\qquad \mathcal L=D_{\rm KL}+\lambda_{\rm reg}\mathcal L_{\rm reg}.$$

### 3.5 局限
- 式 (2) 的严格成立需要 $G_\theta$ 足够正则、期望可交换（非Injective 映射下 $\log p_{\rm fake}$ 梯度用 score 表达需附加条件）；原文未讨论。
- 假 score 是**滞后估计**，训练早期不准；必须有回归损失 → 大规模 T2I 预构数据集昂贵，且把学生绑在教师采样路径上（DMD2 的动机）。

---

## 4. DMD2（Yin et al., arXiv:2405.14867, NeurIPS 2024）

### 4.1 去 regression：诊断与 TTUR
去掉回归损失后训练不稳，作者归因于**假 critic（$\mu_{\rm fake}^\phi$）未准确估计当前生成分布**。采用 two time-scale update rule（TTUR）：让 critic/假 score 更新更快、生成器更慢，使 critic 始终接近当前 $p_{\rm fake}$ 的平衡点（思想同 WGAN-GP 的 TTUR/Heidelberg 判据）。

### 4.2 GAN 损失：直接吃真实数据
加入判别器区分生成样本与真实图像：

$$\mathcal L_{\rm GAN}=\mathbb E_{z}\big[f(G_\theta(z))\big]-\mathbb E_{y\sim p_{\rm data}}\big[f(y)\big]\quad\text{（以 hinge/对偶形式实现）},$$

$$\nabla_\theta\big(\mathcal L_{\rm DMD}+\lambda_{\rm GAN}\mathcal L_{\rm GAN}\big).$$

意义：DMD 只通过教师（real score 是教师近似）间接获得真实信息；GAN 让学生**直接在真实数据上受监督**，弥补教师 real-score 误差，甚至可超过教师（ImageNet-64 FID 1.28，COCO 8.35）。

### 4.3 多步失配修复（training–inference input mismatch）
多步学生在推理时第 $k$ 步输入是**学生自己上一步的输出**，而原 DMD 训练时输入始终是教师/真值加噪 → 分布失配，步数越多误差越累积。修复：训练时**模拟推理轨迹**——先从 $z$ 用当前学生生成中间样本，再加噪/作为下一步输入：

$$\hat x_{k}=G_\theta^{(k)}(\cdot),\quad x_{\rm in}^{(k+1)}=\text{student rollout （而非教师/真值）},$$

即在 student's own samples 上计算 DMD/GAN 梯度。由此一个模型可在 1–4 步间自适应。

### 4.4 局限
- 引入 GAN → 继承 GAN 训练敏感（判别器架构、正则需精细设计），可能牺牲部分多样性/校准。
- 多步 rollout 训练增加显存/计算；TTUR 快慢比需调参；质量上限仍受教师分布与真实数据覆盖影响。

---

## 5. 一致性蒸馏目标（Consistency Models, Song et al., ICML 2023, arXiv:2303.01469；连续时间 sCM 见 SANA-Sprint, arXiv:2503.09641）

### 5.1 自一致性定义
PF ODE 轨迹 $\{\mathbf x_t\}$ 上，一致性函数 $f_\theta$ 把任意时刻点映回轨迹起点：$f_\theta(\mathbf x_t,t)\approx\mathbf x_0$，且边界条件 $f_\theta(\mathbf x_\epsilon,\epsilon)=\mathbf x_\epsilon$。同一条轨迹上输出必须一致。

### 5.2 Consistency Distillation（CD）目标
用教师 score $\phi$ 的一步 ODE（相邻时刻 $t_{n+1}\to t_n$）产生相邻对，要求学生在两端输出一致：

$$\boxed{\;\mathcal L_{\rm CD}^{N}(\theta,\theta^-,\phi)=\mathbb E\Big[\lambda(t_n)\,d\big(f_\theta(\mathbf x_{t_{n+1}},t_{n+1}),\,f_{\theta^-}(\hat{\mathbf x}_{t_n}^{\phi},t_n)\big)\Big]\;}$$

其中 $\hat{\mathbf x}_{t_n}^\phi=\mathbf x_{t_{n+1}}+(t_n-t_{n+1})\,\Phi(\hat{\mathbf x}_{t_{n+1}};\phi)$ 为教师 ODE 一步；$\theta^-$ 为停止梯度的 EMA 目标网络（避免坍缩平凡解）；$d$ 可取 $\ell_2,\ell_1$, LPIPS；$\lambda\equiv1$。

### 5.3 连续时间一致性蒸馏（sCM, Lu et al. 2023；SANA-Sprint 应用, arXiv:2503.09641）
离散 CD 依赖时间网格，sCM 把相邻对推广为连续时间 $\tau^-,\tau$（$\tau^-=\tau-\Delta\tau$, $\Delta\tau\to0$）的期望一致性损失，消除离散化偏差。SANA-Sprint 三点：
1. **免训练**把预训练 flow-matching 模型转成适于 sCM 的形式，再做连续时间一致性蒸馏；
2. 混合蒸馏：sCM（对齐教师 ODE）+ LADD 潜空间对抗蒸馏（提升单步保真）；
3. 统一 step-adaptive：一个模型 1–4 步，无需按步数分别训练。

### 5.4 局限
- CD 只在 ODE 轨迹局部对齐，无显式分布/似然保证；单步质量弱于多步，易过平滑。
- 依赖 EMA 目标与边界参数 $\epsilon$；sCM 对教师 score 精度敏感；LADD 引入 GAN 类不稳定性。

---

## 6. MDLM：Masked Diffusion Language Models（Sahoo et al., NeurIPS 2024, arXiv:2406.07524）

### 6.1 Mask 作为吸收态（absorbing state）
单 token $K$ 类，one-hot $\mathbf x\in\{0,1\}^K$，第 $K$ 类为特殊 $[mask]$，其 one-hot 记 $\mathbf m$。前向在干净数据与先验 $\pi$ 间插值：

$$q(\mathbf z_t|\mathbf x)=\mathrm{Cat}\big(\mathbf z_t;\,\alpha_t\mathbf x+(1-\alpha_t)\pi\big),\qquad \alpha_0\approx1,\ \alpha_1\approx0.$$

Masked diffusion 取 $\pi=\mathbf m$：一旦变 mask 就永久停留（吸收），$t=1$ 全 mask。其后验在 $\mathbf z_t=\mathbf m$ 时简化为

$$q(\mathbf z_s|\mathbf z_t,\mathbf x)=\mathrm{Cat}\!\left(\mathbf z_s;\frac{(1-\alpha_s)\mathbf m+(\alpha_s-\alpha_t)\mathbf x}{1-\alpha_t}\right)\quad(s<t).\tag{8}$$

### 6.2 SUBS 参数化
反向模型直接"代入"网络预测 $\mathbf x_\theta(\mathbf z_t,t)\in\Delta^K$ 到真实后验：

$$p_\theta(\mathbf z_s|\mathbf z_t)=q(\mathbf z_s|\mathbf z_t,\mathbf x=\mathbf x_\theta(\mathbf z_t,t)),$$

并用两个硬替换（SUBS）编码过程结构：
1. **Zero masking probability**：真实 token 永不等于 mask，$\langle\mathbf x,\mathbf m\rangle=0$ → 把 mask 类 logit 置 $-\infty$，强制 $\langle\mathbf x_\theta,\mathbf m\rangle=0$；
2. **Carry-over unmasking**：未被 mask 的 token 反向时保持不变 → 直接拷贝未 mask 输入。

### 6.3 Rao-Blackwellized ELBO → 加权 MLM（推导）
扩散 NELBO 的每段是 $D_{\rm KL}(q(\mathbf z_s|\mathbf z_t,\mathbf x)\Vert p_\theta(\mathbf z_s|\mathbf z_t))$。【原文 Suppl. B + 本文整理】在 SUBS 下：
- 未 mask 位置前后验恒等（carry-over），KL=0；
- mask 位置上，后验是"以概率 $(\alpha_s-\alpha_t)/(1-\alpha_t)$ 揭示真值、否则继续 mask"的两点分布；因 $\langle\mathbf x_\theta,\mathbf m\rangle=0$，学生在"揭示"分支给出的质量恰为 $\langle\mathbf x_\theta(\mathbf z_t,t),\mathbf x\rangle$；
- 该 Bernoulli/类别 KL 解析地只剩交叉熵项：

$$\mathcal L_{\rm diffusion}=\sum_i\mathbb E_q\left[\frac{\alpha_{t(i)}-\alpha_{s(i)}}{1-\alpha_{t(i)}}\log\big\langle\mathbf x_\theta(\mathbf z_{t(i)}),\mathbf x\big\rangle\right].\tag{10}$$

连续时间 $T\to\infty$：

$$\boxed{\;\mathcal L_{\rm NELBO}^{\infty}=\mathbb E_q\int_0^1\frac{\alpha_t'}{1-\alpha_t}\,\log\big\langle\mathbf x_\theta(\mathbf z_t,t),\mathbf x\big\rangle\,dt\;}\tag{12}$$

这就是**加权 masked-language-modeling 交叉熵损失的混合**：每个 mask 位置按权重 $-\alpha_t'/(1-\alpha_t)$ 惩罚 $\log p_\theta(x_i|\text{masked context})$。变量：$\langle\cdot,\cdot\rangle$ 为点积；$\alpha_t'=d\alpha_t/dt$。

**换元 $\gamma=\log(1-\alpha_t)$**（$\alpha$ 单调可逆）：

$$\mathcal L_{\rm NELBO}^{\infty}=-\mathbb E_q\int_{-\infty}^{0}\log\langle\mathbf x_\theta(\mathbf z_\gamma,\gamma),\mathbf x\rangle\,d\gamma,$$

可见目标**对噪声日程 $\alpha_t$ 的函数形式不变**（schedule invariance）。

**为什么叫 Rao-Blackwellization**：利用条件独立性（SUBS）解析地求出诸如 $\langle\mathbf x_\theta,\mathbf m\rangle=0$ 的期望，把原本需网络学出的项消掉，降低方差、收紧界（作者注明是对该术语的宽松借用）。

### 6.4 序列与 semi-AR
假设前向逐 token 独立、反向按 token 分解 $p_\theta(\mathbf z_s^{1:L}|\mathbf z_t^{1:L})=\prod_\ell p_\theta(\mathbf z_s^\ell|\mathbf z_t^{1:L})$，单个 encoder-only 网络对所有位置并行预测；采样可半自回归（分块揭示），支持任意长度。

### 6.5 局限
- PPL 仍比 AR 差约 15–25%（论文自报）；ELBO 是下界，式 (12) 与真实似然有间隙。
- mask 吸收过程信息只减不增，高 SNR（接近全 mask）端预测任务极难；采样器/步数对结果影响大。
- "Rao-Blackwellization"严格术语下是模型结构约束而非纯方差缩减，作者亦承认。

---

## 7. Diffusion Forcing：每 token 独立噪声（Boyuan Chen et al., 2024, arXiv:2407.01392）

### 7.1 核心机制
标准全序列扩散对所有 token 用**同一**噪声水平 $t$。Diffusion Forcing 令每个 token 有独立噪声水平 $t_i$：

$$q(\mathbf z_t^{1:L}|\mathbf x^{1:L})=\prod_{i=1}^{L}q(\mathbf z_{t_i}^{i}|\mathbf x^i),\qquad t_i\overset{iid}{\sim}U[0,1],$$

用一个 **causal next-token 架构**对任意每位置噪声水平去噪。极端情形连续插值：
- 历史 token $t_i=0$（干净）+ 未来 token $t_j=1$（全噪）→ 退化为标准**自回归 next-token 预测**；
- 所有 $t_i=t$ 同步 → 退化为标准**全序列扩散**。

### 7.2 似然意义
论文证明该训练对真实联合分布的**所有子序列**似然都优化一个变分下界（ELBO）：因为独立每 token 噪声等价于对子序列做前向扩散，causal 模型给出对应反向，故变长生成、超越训练时长 rollout、对未来做引导/采样均有原理依据。

### 7.3 变量与局限
$t_i$：位置 $i$ 的独立噪声时间；$\mathbf z_{t_i}^i=\alpha_{t_i}x^i+\sigma_{t_i}\epsilon_i$。
局限：causal + 每位置时间条件增加实现复杂度；论文主要在连续 token（视频/控制/机器人轨迹）验证，离散语言上的大规模优势未充分证明；ELBO 仍宽松，采样引导需额外设计。

---

## 8. 自回归似然 与 扩散 ELBO 如何统一

### 8.1 两者都是同一序列概率分解的特例
任何序列联合分布用链式法则：

$$\log p(\mathbf x^{1:L})=\sum_{i=1}^{L}\log p(x_i|\mathbf x^{<i}).$$

- **自回归**：以因果解码器直接建模每个条件项，是**精确**分解（模型类内无界间隙），但严格顺序。
- **扩散 ELBO**：在每个（或全部）位置引入隐变量 $\mathbf z_t$，

$$\log p(\mathbf x)\ge \mathbb E_q[-\mathcal L_{\rm recons}]-\sum_i D_{\rm KL}\big(q(\mathbf z_s|\mathbf z_t,\mathbf x)\Vert p_\theta(\mathbf z_s|\mathbf z_t)\big)-D_{\rm prior},$$

是**下界**，但位置可并行、噪声水平可任意。

### 8.2 统一视角一：Diffusion Forcing 的端点退化
如 7.1，当噪声水平向量取 $\mathbf t=(0,\dots,0,1)$（历史干净、下一 token 全噪），扩散去噪目标 $-\log\langle \mathbf x_\theta(\mathbf z_t,t),\mathbf x\rangle$ 恰好就是条件 next-token 交叉熵 $-\log p_\theta(x_i|x_{<i})$；$\mathbf t$ 全部相等则为全序列扩散。AR 与全序列扩散是"每 token 独立噪声多面体"的两个顶点，MDLM 式 (12) 是其加权 MLM 形式。

### 8.3 统一视角二：扩散目标 = 带数据增强的 ELBO（Kingma & Gao, arXiv:2303.00848）
常用扩散目标（$\epsilon$-MSE、$v$-MSE 等）都可写成对各噪声水平 **ELBO 的加权积分**：

$$\mathcal L_{\rm diff}=\mathbb E_{t}\big[\omega(t)\,\mathrm{ELBO}(t)\big].$$

当权重满足**单调条件**时，该目标等价于"对数据做高斯噪声扰动这一简单数据增强后的 ELBO"——即扩散训练与（增强）最大似然并非异类，$\epsilon$-MSE 只是 ELBO 的特定加权/截断。这把"感知质量目标"重新纳入似然框架。

### 8.4 统一视角三：工程上的混合谱系
- **ARLON**（arXiv:2410.20502）：AR Transformer 负责长程时间结构（先验/规划），DiT 在块内并行扩散去噪，二者级联，长视频上互补。
- **CausVid**（arXiv:2412.08725）与 **PDD**（Parallel Decoding Distillation）：把因果（自回归）模型通过**并行解码蒸馏**变成少步/分布式扩散式生成，显式让学生在同一条件下匹配多步/多 token 预测，修复因果–并行失配（思想同 DMD2 的 rollout 修复）。
- **Block diffusion / AR2D**（Runway 等）：块大小 = 1 时恢复逐 token 自回归，块增大趋向全序列扩散，提供连续插值旋钮。

### 8.5 统一表述（本文归纳）
令噪声水平为向量 $\mathbf t\in[0,1]^L$、损失为加权逐位置去噪交叉熵

$$\mathcal J(\mathbf t)=\mathbb E\left[\sum_{i=1}^{L}w(t_i)\,\big(-\log p_\theta(x_i\mid \{\mathbf z_{t_j}^{j}\}_{j\le i})\big)\right].$$

- $\mathbf t=(0,\dots,0,1)$ 且因果：$\mathcal J$ = AR 似然；
- $\mathbf t=t\mathbf 1$：$\mathcal J$ = 扩散 ELBO（连续 MDLM 形式 / Kingma-Gao 加权 ELBO）；
- 一般 $\mathbf t$：Diffusion Forcing；块级版本：ARLON/Block diffusion；蒸馏（PD/CD/DMD/PDD）只改变**推理时把这条期望路径压缩成几步**，不改变其作为似然下界的身份。

### 8.6 局限与不确定性
- "统一"是**目标族层面**的：AR 是精确似然而扩散是 ELBO，二者数值不可直接等同；间隙取决于扩散日程、权重与模型容量。
- CausVid/PDD 的具体失配修复形式本次仅依据摘要/二手核对，未逐式核原文，相关论断标注【不确定，需核原文】。
- Kingma–Gao 的单调权重条件并非对所有实际权重都成立（非单调时只能称加权 ELBO 积分）。

---

## 参考文献（含 arXiv）
1. Salimans & Ho, Progressive Distillation, arXiv:2202.00512 (ICLR 2022).
2. Meng et al., On Distillation of Guided Diffusion Models, arXiv:2210.03142 (CVPR 2023).
3. Yin et al., DMD, arXiv:2311.18828 (CVPR 2024).
4. Yin et al., DMD2, arXiv:2405.14867 (NeurIPS 2024).
5. Song et al., Consistency Models, arXiv:2303.01469 (ICML 2023).
6. SANA-Sprint, arXiv:2503.09641.
7. Sahoo et al., MDLM, arXiv:2406.07524 (NeurIPS 2024).
8. Boyuan Chen et al., Diffusion Forcing, arXiv:2407.01392.
9. Kingma & Gao, Diffusion Objectives as ELBO with Data Augmentation, arXiv:2303.00848.
10. ARLON, arXiv:2410.20502；CausVid, arXiv:2412.08725.
