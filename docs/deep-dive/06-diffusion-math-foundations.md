# 扩散模型 / Score Matching / Flow Matching 的数学原理

> 本文系统梳理 DDPM（离散变分扩散）、Score Matching（含去噪 score matching）、SDE 连续时间框架（VE/VP/sub-VP、Fokker–Planck、Anderson 反向 SDE、概率流 ODE）、DDIM、Flow Matching / Rectified Flow（最优传输条件流）以及 Consistency Models（自洽边界条件与连续时间一致性函数）的数学推导，并给出各理论之间的统一关系。
>
> 公式均核对自论文原文（arXiv HTML/PDF）。个别原文符号习惯已在文中注明；推导中需要自行补充的步骤与存疑处均以 **〔注〕** 标出。

---

## 0. 符号约定

| 符号 | 含义 |
|---|---|
| $\mathbf{x}_0\sim p_{\mathrm{data}}(\mathbf{x})=p_0(\mathbf{x})$ | 数据样本 / 初始（干净）分布 |
| $\mathbf{x}_t$（离散 $t=1,\dots,T$ 或连续 $t\in[0,T]$） | 加噪后的隐变量 |
| $p_t(\mathbf{x})$ | 时刻 $t$ 的边缘密度 |
| $\beta_t$、$\alpha_t=1-\beta_t$、$\bar\alpha_t=\prod_{s\le t}\alpha_s$ | DDPM 噪声调度（Ho et al. 2020 用 $\bar\alpha_t$；Song 2021 把累积量直接记作 $\alpha_t$） |
| $\boldsymbol\epsilon,\mathbf{z},\mathbf{w}$ | 标准高斯噪声 / Wiener 过程（布朗运动） |
| $\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$ | score function（对数密度梯度） |
| $\mathbf{s}_\theta(\mathbf{x},t)$、$\boldsymbol\epsilon_\theta(\mathbf{x},t)$ | 网络对 score / 噪声的预测 |
| $v_t(\mathbf{x})$、$u_t(\mathbf{x})$、$\phi_t$ | 向量场及其生成的流（flow，微分同胚） |

**一句话总览**：所有这些模型都在学习一条「数据分布 $\leftrightarrow$ 已知简单先验（通常是高斯）」之间的概率路径 $p_t$；DDPM 用离散马尔可夫链 + 变分下界，score-based 模型用 score 场 + Langevin/SDE，flow matching 用确定性 ODE 向量场，consistency model 则把整条 ODE 轨迹上的点一次性映射回轨迹起点。

---

## 1. DDPM：变分下界、前向闭式解、反向过程与噪声预测

**出处**：J. Ho, A. Jain, P. Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS 2020, arXiv:2006.11239。前向马尔可夫扩散的形式最早出自 Sohl-Dickstein et al., ICML 2015（*Deep Unsupervised Learning using Nonequilibrium Thermodynamics*）。

### 1.1 前向过程及其闭式解

前向（推理）过程是固定的、无参数的马尔可夫链：

$$
q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)=\prod_{t=1}^{T}q(\mathbf{x}_t\mid\mathbf{x}_{t-1}),\qquad
q(\mathbf{x}_t\mid\mathbf{x}_{t-1})=\mathcal{N}\!\left(\mathbf{x}_t;\sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\beta_t\mathbf{I}\right).
$$

令 $\alpha_t=1-\beta_t,\ \bar\alpha_t=\prod_{s=1}^{t}\alpha_s$。利用高斯分布可加性（重参数化连乘），直接对 $\mathbf{x}_0$ 加噪有**闭式解**（DDPM 论文 Eq. (4)）：

$$
\boxed{\;q(\mathbf{x}_t\mid\mathbf{x}_0)=\mathcal{N}\!\left(\mathbf{x}_t;\sqrt{\bar\alpha_t}\,\mathbf{x}_0,(1-\bar\alpha_t)\mathbf{I}\right)\;}
$$

即

$$
\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,\qquad \boldsymbol\epsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
$$

**推导要点**：每一步 $\mathbf{x}_t=\sqrt{\alpha_t}\mathbf{x}_{t-1}+\sqrt{\beta_t}\boldsymbol\epsilon_{t-1}$ 递归代入，噪声项 $\sum_{s=1}^{t}\big(\prod_{r=s+1}^{t}\alpha_r\big)\beta_s\boldsymbol\epsilon_{s-1}$ 相互独立，方差之和为 $\sum_s \bar\alpha_t\,\beta_s/\alpha_s=1-\bar\alpha_t$（等比望远镜求和）。当 $\bar\alpha_T\approx0$ 时 $\mathbf{x}_T\sim\mathcal{N}(\mathbf{0},\mathbf{I})$，故可取先验 $p(\mathbf{x}_T)=\mathcal{N}(\mathbf{0},\mathbf{I})$。

### 1.2 反向过程

真正的反向核 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ 一般不可解析，但当前向是高斯且噪声很小时它仍近似高斯（Sohl-Dickstein 2015；连续极限下即第 3 节的反向 SDE）。模型化为

$$
p_\theta(\mathbf{x}_{0:T})=p(\mathbf{x}_T)\prod_{t=1}^{T}p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t),\qquad
p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)=\mathcal{N}\!\left(\boldsymbol\mu_\theta(\mathbf{x}_t,t),\Sigma_\theta(\mathbf{x}_t,t)\right).
$$

### 1.3 ELBO 推导

对任意推理分布 $q$，由 Jensen 不等式（$\log p_\theta(\mathbf{x}_0)=\log\mathbb{E}_q[p_\theta(\mathbf{x}_{0:T})/q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)]\ge \mathbb{E}_q[\log p_\theta(\mathbf{x}_{0:T})-\log q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)]$）：

$$
\log p_\theta(\mathbf{x}_0)\ge \underbrace{\mathbb{E}_{q}\!\left[\log p_\theta(\mathbf{x}_{0:T})-\log q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)\right]}_{-L}
$$

展开乘积并整理，DDPM 论文 Eq. (5) 给出 ELBO 的三项分解：

$$
\mathbb{E}_{q}\!\left[\underbrace{-\log\frac{p_\theta(\mathbf{x}_0\mid\mathbf{x}_1)}{r_1}}_{L_0}
+\sum_{t>1}\underbrace{D_{\mathrm{KL}}\!\Big(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)\,\Big\|\,p_\theta(\mathbf{x}_{t-1}\mid\mathbf{x}_t)\Big)}_{L_{t-1}}
+\underbrace{D_{\mathrm{KL}}\!\big(q(\mathbf{x}_T\mid\mathbf{x}_0)\,\big\|\,p(\mathbf{x}_T)\big)}_{L_T}\right].
$$

其中 $L_T$ 与 $\theta$ 无关（前向已固定），$L_0$ 是离散数据的解码器项。

**关键技巧**：后验 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$ 可由贝叶斯公式得到闭式高斯（DDPM Eq. (6)–(7)）：

$$
q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)=\mathcal{N}\!\left(\mathbf{x}_{t-1};\tilde{\boldsymbol\mu}_t(\mathbf{x}_t,\mathbf{x}_0),\tilde\beta_t\mathbf{I}\right),
$$

$$
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t,\qquad
\tilde{\boldsymbol\mu}_t=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\mathbf{x}_0+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t.
$$

于是 KL 项变成两个高斯之间的距离，训练只需让 $\boldsymbol\mu_\theta$ 逼近 $\tilde{\boldsymbol\mu}_t$。

### 1.4 参数化：预测噪声与简化损失

DDPM 不直接预测均值，而是预测加入的噪声 $\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)$。由 $\tilde{\boldsymbol\mu}_t$ 的表达式代入 $\mathbf{x}_0=(\mathbf{x}_t-\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon)/\sqrt{\bar\alpha_t}$：

$$
\boldsymbol\mu_\theta(\mathbf{x}_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)\right).
$$

此时 $L_{t-1}$（去掉与 $\theta$ 无关的常数）是加权噪声预测误差（DDPM Eq. (11)–(12)，简化前）：

$$
L_{t-1}=\mathbb{E}_{\mathbf{x}_0,\boldsymbol\epsilon}\!\left[\frac{\beta_t^2}{2\alpha_t(1-\bar\alpha_t)\,\sigma_t^2}\big\|\boldsymbol\epsilon-\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)\big\|^2\right],
$$

其中 $\sigma_t^2=\beta_t$（或 $\tilde\beta_t$，取决于反向方差的选择）。Ho et al. 经验上**抛弃时间相关权重**，得到实际使用的简化目标（DDPM Eq. (14)）：

$$
\boxed{\;L_{\mathrm{simple}}(\theta)=\mathbb{E}_{t,\mathbf{x}_0,\boldsymbol\epsilon}\!\left[\big\|\boldsymbol\epsilon-\boldsymbol\epsilon_\theta(\sqrt{\bar\alpha_t}\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon,t)\big\|^2\right]\;}
$$

注意 $L_{\mathrm{simple}}$ 不再是严格的 ELBO（权重被改），但样本质量显著更好；论文还保留了 $L_0$ 的离散解码器（用 $\mathcal{N}(\mathbf{x}_0;\boldsymbol\mu_\theta,\Sigma_\theta)$ 建模逐像素对数似然）。反向方差固定为 $\sigma_t^2=\beta_t$ 或 $\tilde\beta_t$（DDPM Eq. (16)–(17)）。

**采样（ancestral，DDPM Eq. (15)）**：

$$
\mathbf{x}_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)\right)+\sigma_t\mathbf{z},\quad \mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})\ (\text{仅 }t>1).
$$

### 1.5 score 视角

Tweedie 公式（高斯加噪 $\mathbf{x}_t=\sqrt{\bar\alpha_t}\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon$）给出：

$$
\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t\mid\mathbf{x}_0)=-\frac{\mathbf{x}_t-\sqrt{\bar\alpha_t}\mathbf{x}_0}{1-\bar\alpha_t}=-\frac{\boldsymbol\epsilon}{\sqrt{1-\bar\alpha_t}}.
$$

去噪 score matching 保证最优时 $\boldsymbol\epsilon_\theta$ 预测的即真实噪声，于是

$$
\mathbf{s}_\theta(\mathbf{x}_t,t)=-\frac{\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)}{\sqrt{1-\bar\alpha_t}}\approx\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t).
$$

**这就是 DDPM 与 score matching 的等价桥梁**（Song 2021 把 $L_{\mathrm{simple}}$ 显式写成加权 DSM，见该文 Eq. (3)）。

---

## 2. Score Matching 与 Denoising Score Matching（DSM）

**出处**：
- A. Hyvärinen, *Estimation of Non-Normalized Statistical Models by Score Matching*, JMLR 2005。
- P. Vincent, *A Connection Between Score Matching and Denoising Autoencoders*, Neural Computation 2011（任务书所写 2008 系其技术报告年份；2011 为正式发表）。
- Y. Song, S. Ermon, *Generative Modeling by Estimating Gradients of the Data Distribution*（NCSN），NeurIPS 2019，arXiv:1907.05600。

### 2.1 显式 score matching

目标密度 $p_{\mathrm{data}}(\mathbf{x})$，模型 $p_\theta(\mathbf{x})=\frac{1}{Z_\theta}e^{-f_\theta(\mathbf{x})}$ 配分函数 $Z_\theta$ 难算，但其 score $\mathbf{s}_\theta(\mathbf{x})=\nabla_\mathbf{x}\log p_\theta(\mathbf{x})$ 不依赖 $Z_\theta$。Hyvärinen 提出最小化期望 Fisher 散度：

$$
J_{\mathrm{SM}}(\theta)=\mathbb{E}_{p_{\mathrm{data}}}\left[\tfrac12\big\|\mathbf{s}_\theta(\mathbf{x})-\nabla_\mathbf{x}\log p_{\mathrm{data}}(\mathbf{x})\big\|^2\right].
$$

经分部积分（要求边界项为零），去掉与 $\theta$ 无关项后得到可计算形式（Hyvärinen 2005 Eq. (4)）：

$$
J_{\mathrm{SM}}(\theta)=\mathbb{E}_{p_{\mathrm{data}}}\left[\,\mathrm{tr}\!\big(\nabla_\mathbf{x}\mathbf{s}_\theta(\mathbf{x})\big)+\tfrac12\|\mathbf{s}_\theta(\mathbf{x})\|^2\right]+\text{const}.
$$

实际困难：二阶导（Hessian 迹）计算贵；且要求 $p_{\mathrm{data}}$ 支集边界处模型密度趋零，对低维流形数据（图像）不成立。

### 2.2 去噪 score matching（Vincent 2011）

引入已知扰动核 $q_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})=\mathcal{N}(\tilde{\mathbf{x}};\mathbf{x},\sigma^2\mathbf{I})$，扰动后密度 $q_\sigma(\tilde{\mathbf{x}})=\int p_{\mathrm{data}}(\mathbf{x})q_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})\,d\mathbf{x}$。Vincent 的定理给出：**对扰动分布做显式 score matching，恒等于在条件核下做回归**：

$$
\boxed{\;\mathbb{E}_{q_\sigma(\tilde{\mathbf{x}})}\left[\tfrac12\big\|\mathbf{s}_\theta(\tilde{\mathbf{x}})-\nabla_{\tilde{\mathbf{x}}}\log q_\sigma(\tilde{\mathbf{x}})\big\|^2\right]
=\mathbb{E}_{p_{\mathrm{data}}(\mathbf{x})}\mathbb{E}_{q_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})}\left[\tfrac12\big\|\mathbf{s}_\theta(\tilde{\mathbf{x}})-\nabla_{\tilde{\mathbf{x}}}\log q_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})\big\|^2\right]+\text{const}\;}
$$

右端目标 $\nabla_{\tilde{\mathbf{x}}}\log q_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x})=(\mathbf{x}-\tilde{\mathbf{x}})/\sigma^2$ 完全已知，无需数据的真实 score，也无二阶导。这就是"去噪自编码器 ≡ score matching"的核心。**〔注〕** 等式证明依赖对 $\mathbf{x}$ 的分部积分；Vincent 2011 的结论是"目标值（objective value）相等"，其对 $\theta$ 的梯度自然相等；Lipman 2023（flow matching）借鉴的正是这一"条件目标替代边缘目标"的技巧。

### 2.3 NCSN：多尺度噪声 + Langevin 采样

Song & Ermon 2019 对几何递增噪声序列 $\sigma_1<\cdots<\sigma_N$ 训练一个噪声条件 score 网络（Song 2021 Eq. (1) 的形式）：

$$
\theta^*=\arg\min_\theta\sum_{i=1}^{N}\sigma_i^2\,\mathbb{E}_{p_{\mathrm{data}}(\mathbf{x})}\mathbb{E}_{p_{\sigma_i}(\tilde{\mathbf{x}}\mid\mathbf{x})}\left[\big\|\mathbf{s}_\theta(\tilde{\mathbf{x}},\sigma_i)-\nabla_{\tilde{\mathbf{x}}}\log p_{\sigma_i}(\tilde{\mathbf{x}}\mid\mathbf{x})\big\|^2\right].
$$

权重 $\sigma_i^2$ 把各噪声尺度的 Fisher 散度归一化（条件 score 的期望范数 $\propto1/\sigma_i^2$）。采样用 **Langevin MCMC**（Song 2021 Eq. (2)）：从 $\mathbf{x}_N^0\sim\mathcal{N}(\mathbf{0},\sigma_N^2\mathbf{I})$ 出发，在每个尺度上迭代

$$
\mathbf{x}_i^{m}=\mathbf{x}_i^{m-1}+\epsilon_i\,\mathbf{s}_{\theta^*}(\mathbf{x}_i^{m-1},\sigma_i)+\sqrt{2\epsilon_i}\,\mathbf{z}_i^m,
$$

步长 $\epsilon_i\to0$、步数 $M\to\infty$ 时收敛到该尺度的平稳分布 $p_{\sigma_i}$（Langevin 扩散 $d\mathbf{x}=\tfrac12\nabla\log p\,dt+d\mathbf{w}$ 的平稳分布即 $p$）。逐级降噪至 $\sigma_1\approx0$ 得到数据样本。

---

## 3. SDE 框架：前向/反向 SDE、Fokker–Planck、VE/VP

**出处**：Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, B. Poole, *Score-Based Generative Modeling through Stochastic Differential Equations*, ICLR 2021 (Oral), arXiv:2011.13456。反向 SDE 的一般理论出自 B. D. Anderson, *Reverse-time Stochastic Difference Equations*, 1982（文末参考文献；相关 STAP 论文亦引 Øksendal 教材）。

### 3.1 前向 SDE

$$
\boxed{\;d\mathbf{x}=\mathbf{f}(\mathbf{x},t)\,dt+g(t)\,d\mathbf{w}\;}\qquad(\text{Song 2021 Eq. (5)})
$$

$\mathbf{f}$：漂移系数（向量值），$g$：扩散系数（标量，与 $\mathbf{x}$ 无关；一般情形见其 Appendix A），$\mathbf{w}$：标准 Wiener 过程。$\mathbf{x}(0)\sim p_0$ 为数据，$\mathbf{x}(T)\sim p_T$ 为简单先验。

### 3.2 Fokker–Planck（Kolmogorov 前向）方程

边缘密度 $p_t(\mathbf{x})$ 的演化满足

$$
\frac{\partial p_t(\mathbf{x})}{\partial t}=-\sum_i\frac{\partial}{\partial x_i}\big[f_i(\mathbf{x},t)p_t(\mathbf{x})\big]+\frac12\sum_{i,j}\frac{\partial^2}{\partial x_i\partial x_j}\big[g(t)^2\,p_t(\mathbf{x})\big],
$$

即 $\partial_t p_t=-\nabla\!\cdot(\mathbf{f}p_t)+\tfrac12 g(t)^2\Delta p_t$（散度形式）。它与描述反向条件期望的 Kolmogorov 后向方程 $\partial_t u+\mathbf{f}\!\cdot\nabla u+\tfrac12g^2\Delta u=0$ 互为伴随。Fokker–Planck 是验证"向量场/SDE 生成某条概率路径"的根本工具，也是 3.5 概率流 ODE 的推导出发点。

### 3.3 Anderson 反向 SDE

Anderson (1982) 的结论：上述扩散过程的时间反演**仍是扩散过程**，其漂移只比正向多出一个 score 项（Song 2021 Eq. (6)）：

$$
\boxed{\;d\mathbf{x}=\big[\mathbf{f}(\mathbf{x},t)-g(t)^2\nabla_{\mathbf{x}}\log p_t(\mathbf{x})\big]dt+g(t)\,d\bar{\mathbf{w}}\;}
$$

$\bar{\mathbf{w}}$ 是时间从 $T$ 倒流到 $0$ 意义下的标准 Wiener 过程，$dt$ 为负的无穷小时间步。只要用 $\mathbf{s}_\theta$ 估计出所有时刻的 score，就能用任意 SDE 数值求解器从 $p_T$ 反向积到 $p_0$ 采样。

**〔注〕** 反向 SDE 可由反向时间的 Fokker–Planck 方程配方法/测度 Girsanov 变换推导；论文将完整推导放在 Appendix D.1 与引用 Anderson 1982。反向扩散系数与正向相同（$g^2$ 的项），这是标量扩散系数情形；一般矩阵情形反向扩散项与 $\mathbf{x}$ 相关（Appendix A）。

### 3.4 VE 与 VP SDE

NCSN（SMLD）与 DDPM 的离散扰动在 $N\to\infty$ 连续极限下分别给出两种 SDE（Song 2021 Eq. (9)、(11)）：

**(a) Variance Exploding（VE）**——SMLD 的连续化，无漂移，方差单调爆炸：

$$
d\mathbf{x}=\sqrt{\frac{d[\sigma^2(t)]}{dt}}\,d\mathbf{w},\qquad p_{0t}(\mathbf{x}(t)\mid\mathbf{x}(0))=\mathcal{N}\big(\mathbf{x}(t);\mathbf{x}(0),[\sigma^2(t)-\sigma^2(0)]\mathbf{I}\big).
$$

**(b) Variance Preserving（VP）**——DDPM 的连续化，均值被拉向 0，单位方差初值时方差恒为 1（故得名）：

$$
\boxed{\;d\mathbf{x}=-\tfrac12\beta(t)\mathbf{x}\,dt+\sqrt{\beta(t)}\,d\mathbf{w}\;}
$$

$$
p_{0t}(\mathbf{x}(t)\mid\mathbf{x}(0))=\mathcal{N}\!\left(\mathbf{x}(t);\mathbf{x}(0)e^{-\frac12\int_0^t\beta(s)ds},\ \left[1-e^{-\int_0^t\beta(s)ds}\right]\mathbf{I}\right).
$$

与 DDPM 记号对应：$\bar\alpha_t=\exp(-\int_0^t\beta(s)ds)$。论文还提出 **sub-VP SDE**（Eq. (12)）：

$$
d\mathbf{x}=-\tfrac12\beta(t)\mathbf{x}\,dt+\sqrt{\beta(t)\left(1-e^{-\int_0^t\beta(s)ds}\right)}\,d\mathbf{w},
$$

其方差 $\big(1-e^{-\int_0^t\beta}\big)^2\mathbf{I}$ 在每个时刻都被 VP 的方差上界控制，似然表现最好。

**连续时间训练目标**（Song 2021 Eq. (7)，统一 NCSN/DDPM 的加权 DSM）：

$$
\theta^*=\arg\min_\theta\mathbb{E}_t\Big\{\lambda(t)\,\mathbb{E}_{\mathbf{x}(0)}\mathbb{E}_{\mathbf{x}(t)\mid\mathbf{x}(0)}\big[\|\mathbf{s}_\theta(\mathbf{x}(t),t)-\nabla_{\mathbf{x}(t)}\log p_{0t}(\mathbf{x}(t)\mid\mathbf{x}(0))\|^2\big]\Big\}.
$$

通常取 $\lambda(t)\propto1/\mathbb{E}\|\nabla\log p_{0t}\|^2$。

### 3.5 概率流 ODE（Probability Flow ODE）

存在一条**确定性** ODE，其轨迹与 SDE 具有完全相同的边缘密度族 $\{p_t\}$（Song 2021 Eq. (13)）：

$$
\boxed{\;\frac{d\mathbf{x}}{dt}=\mathbf{f}(\mathbf{x},t)-\frac12 g(t)^2\nabla_{\mathbf{x}}\log p_t(\mathbf{x})\;}
$$

**推导要点**：该 ODE 对应的连续性方程（密度的 Liouville 方程）$\partial_t p_t=-\nabla\!\cdot\big[(\mathbf{f}-\tfrac12g^2\nabla\log p_t)p_t\big]$ 与 3.2 的 Fokker–Planck 方程相同（展开 $-\tfrac12g^2\nabla\!\cdot(p_t\nabla\log p_t)=-\tfrac12g^2\Delta p_t$），故边缘密度一致。

推论：
- 可用黑盒自适应 ODE 求解器（Dormand–Prince）快速、少步数采样，并显式权衡精度/NFE；
- 用瞬时变量替换公式（Chen et al. 2018 Neural ODE）$\frac{d\log p_t(\mathbf{x}(t))}{dt}=-\nabla\!\cdot v_t(\mathbf{x}(t))$ 可做**精确似然计算**；
- 编码唯一可识别（前向无训练参数），可做潜空间插值、编辑；
- 反向问题（inpainting、colorization）可用 $\nabla\log p_t(\mathbf{x}\mid\mathbf{y})=\nabla\log p_t(\mathbf{x})+\nabla\log p_t(\mathbf{y}\mid\mathbf{x})$ 无条件 score 组合求解。

---

## 4. DDIM：非马尔可夫前向与确定性加速采样

**出处**：J. Song, C. Meng, S. Ermon, *Denoising Diffusion Implicit Models*, ICLR 2021, arXiv:2010.02502。

### 4.1 核心观察

DDPM 的训练目标 $L_\gamma$（加权噪声预测，DDIM Eq. (5)）只依赖**边缘** $q(\mathbf{x}_t\mid\mathbf{x}_0)=\mathcal{N}(\sqrt{\alpha_t}\mathbf{x}_0,(1-\alpha_t)\mathbf{I})$（注意 DDIM 直接用 $\alpha_t$ 表示累积 $\bar\alpha_t$），不依赖具体的联合分布。因此可以构造一族**非马尔可夫**前向联合，保持所有边缘不变，从而得到同一训练目标下不同的（可短得多的）反向链。

### 4.2 非马尔可夫前向过程

以参数 $\boldsymbol\sigma\in\mathbb{R}_{\ge0}^{T}$ 索引（DDIM Eq. (6)–(7)）：

$$
q_\sigma(\mathbf{x}_{1:T}\mid\mathbf{x}_0)=q_\sigma(\mathbf{x}_T\mid\mathbf{x}_0)\prod_{t=2}^{T}q_\sigma(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0),
$$

$$
q_\sigma(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)=\mathcal{N}\!\left(\sqrt{\alpha_{t-1}}\mathbf{x}_0+\sqrt{1-\alpha_{t-1}-\sigma_t^2}\cdot\frac{\mathbf{x}_t-\sqrt{\alpha_t}\mathbf{x}_0}{\sqrt{1-\alpha_t}},\ \sigma_t^2\mathbf{I}\right).
$$

其构造保证 $q_\sigma(\mathbf{x}_t\mid\mathbf{x}_0)=\mathcal{N}(\sqrt{\alpha_t}\mathbf{x}_0,(1-\alpha_t)\mathbf{I})$ 对所有 $t$ 成立（DDIM Lemma 1）。该前向非马尔可夫（$\mathbf{x}_{t-1}$ 依赖 $\mathbf{x}_0$）；$\sigma_t\to0$ 时给定 $\mathbf{x}_0,\mathbf{x}_t$ 则 $\mathbf{x}_{t-1}$ 完全确定。

### 4.3 通用采样更新（DDIM Eq. (12) / Appendix C.3）

令模型先预测去噪结果 $f_\theta^{(t)}(\mathbf{x}_t)=(\mathbf{x}_t-\sqrt{1-\alpha_t}\,\boldsymbol\epsilon_\theta^{(t)}(\mathbf{x}_t))/\sqrt{\alpha_t}$（对 $\mathbf{x}_0$ 的预测），则

$$
\boxed{\;\mathbf{x}_{t-1}=\sqrt{\alpha_{t-1}}\underbrace{\left(\frac{\mathbf{x}_t-\sqrt{1-\alpha_t}\,\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)}{\sqrt{\alpha_t}}\right)}_{\text{预测的 }\mathbf{x}_0}
+\underbrace{\sqrt{1-\alpha_{t-1}-\sigma_t^2}\,\boldsymbol\epsilon_\theta(\mathbf{x}_t,t)}_{\text{指向 }\mathbf{x}_t\text{ 的方向项}}
+\underbrace{\sigma_t\mathbf{z}}_{\text{随机项}}\;}
$$

其中

$$
\sigma_t(\eta)=\eta\sqrt{\frac{1-\alpha_{t-1}}{1-\alpha_t}}\sqrt{1-\frac{\alpha_t}{\alpha_{t-1}}},\qquad \eta\in[0,1].
$$

- $\eta=1$：恢复 DDPM 的随机采样；
- $\eta=0$（$\sigma_t=0$）：**确定性 DDIM**——同一起点 $\mathbf{x}_T$ 无论用多少子步都得到一致（consistency）的样本，可在潜空间做有语义的插值；可只取 $\tau\subset\{1,\dots,T\}$ 的稀疏子序列（10–100 步），加速 $10\times$–$50\times$。

**与概率流 ODE 的关系**：$\eta=0$ 的确定性 DDIM 是概率流 ODE 的一种离散化（在特定 $\alpha_t$ 参数化与步长下近似一致；两者的精确连续/离散对应随参数化而异，**〔注〕** 这是常被混淆之处，严谨表述是"DDIM 与 probability-flow ODE 密切相关、共享确定性边缘"，而非逐数值步恒等）。

---

## 5. Flow Matching 与 Rectified Flow

**出处**：
- Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, M. Le, *Flow Matching for Generative Modeling*, ICLR 2023, arXiv:2210.02747。
- X. Liu, C. Gong, Q. Liu, *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*, ICLR 2023, arXiv:2209.03003（"StnLiu" 即 Xingchao Liu 等）。

### 5.1 连续正规流（CNF）

向量场 $v_t(\mathbf{x})$ 通过 ODE 生成流（微分同胚）$\phi_t$：

$$
\frac{d}{dt}\phi_t(\mathbf{x})=v_t(\phi_t(\mathbf{x})),\quad \phi_0(\mathbf{x})=\mathbf{x};\qquad p_t=[\phi_t]_\ast p_0,
$$

$[\phi_t]_\ast p_0(\mathbf{x})=p_0(\phi_t^{-1}(\mathbf{x}))\det[\partial\phi_t^{-1}/\partial\mathbf{x}]$ 为前推（变量替换）。向量场生成密度路径 $\Leftrightarrow$ 满足**连续性方程** $\partial_t p_t+\nabla\!\cdot(p_t v_t)=0$。

### 5.2 Flow Matching 与 Conditional Flow Matching

给定目标路径 $p_t$ 及其生成向量场 $u_t$，FM 目标（Lipman Eq. (5)）：

$$
L_{\mathrm{FM}}(\theta)=\mathbb{E}_{t,p_t(\mathbf{x})}\|v_t(\mathbf{x})-u_t(\mathbf{x})\|^2.
$$

但 $p_t,u_t$ 通常不可解析。仿照 DSM，引入条件路径 $p_t(\mathbf{x}\mid\mathbf{x}_1)$（$\mathbf{x}_1$ 为数据样本），边缘路径 $p_t(\mathbf{x})=\int p_t(\mathbf{x}\mid\mathbf{x}_1)q(\mathbf{x}_1)d\mathbf{x}_1$，**边缘向量场**（对条件场按后验权重加权，Lipman Eq. (8)）：

$$
u_t(\mathbf{x})=\int u_t(\mathbf{x}\mid\mathbf{x}_1)\frac{p_t(\mathbf{x}\mid\mathbf{x}_1)q(\mathbf{x}_1)}{p_t(\mathbf{x})}\,d\mathbf{x}_1.
$$

- **定理 1**：该边缘向量场生成边缘路径（满足连续性方程）。
- **Conditional Flow Matching 目标**（Eq. (9)）：

$$
\boxed{\;L_{\mathrm{CFM}}(\theta)=\mathbb{E}_{t,q(\mathbf{x}_1),p_t(\mathbf{x}\mid\mathbf{x}_1)}\|v_t(\mathbf{x})-u_t(\mathbf{x}\mid\mathbf{x}_1)\|^2\;}
$$

- **定理 2**：在 $p_t>0$ 条件下，$L_{\mathrm{CFM}}$ 与 $L_{\mathrm{FM}}$ 至多差一个与 $\theta$ 无关的常数，故二者对 $\theta$ 的梯度相同。于是无需模拟 ODE、无需知道边缘场，只要逐样本设计可采样的条件路径即可训练 CNF（"simulation-free"）。这与 DSM 的恒等结构完全平行：**条件回归目标 $\equiv$ 边缘匹配目标**。

### 5.3 高斯条件路径的条件向量场（定理 3）

取 $p_t(\mathbf{x}\mid\mathbf{x}_1)=\mathcal{N}(\boldsymbol\mu_t(\mathbf{x}_1),\sigma_t(\mathbf{x}_1)^2\mathbf{I})$，边界 $\boldsymbol\mu_0=0,\sigma_0=1$（标准高斯噪声端）与 $\boldsymbol\mu_1=\mathbf{x}_1,\sigma_1=\sigma_{\min}$（数据端）。令仿射流 $\psi_t(\mathbf{x})=\sigma_t\mathbf{x}+\boldsymbol\mu_t$，则唯一的规范向量场（Lipman Eq. (15)）：

$$
\boxed{\;u_t(\mathbf{x}\mid\mathbf{x}_1)=\frac{\sigma_t'}{\sigma_t}\big(\mathbf{x}-\boldsymbol\mu_t\big)+\boldsymbol\mu_t'\;}
$$

特例：
- **VE/VP 扩散路径**（Eq. (16)–(19)）：取对应 $\boldsymbol\mu_t,\sigma_t$ 即恢复概率流 ODE 的向量场。例如反向 VP：$u_t=-\tfrac12T'(1-t)\left[\tfrac{e^{-T(1-t)}\mathbf{x}-e^{-T(1-t)/2}\mathbf{x}_1}{1-e^{-T(1-t)}}\right]$，$T(t)=\int_0^t\beta(s)ds$。
- 局限：扩散路径在有限时间内不真正到达噪声（$p_0$ 只能近似高斯）。

### 5.4 最优传输（OT）路径与 Rectified Flow

高斯路径族中特别重要的一支是 Wasserstein-2 最优传输的位移插值（McCann 1997）。取数据样本 $\mathbf{x}_1\sim q$、噪声 $\mathbf{x}_0\sim p=\mathcal{N}(\mathbf{0},\mathbf{I})$，若 $(\mathbf{x}_0,\mathbf{x}_1)$ 按 OT 耦合 $\pi$（高斯 $p$ 对任意 $q$ 时即 Brenier 势映射；实践中用 mini-batch 近似的 OT 耦合 / 独立耦合）配对，则条件路径为直线：

$$
\mathbf{x}_t=(1-t)\mathbf{x}_0+t\mathbf{x}_1,\qquad p_t(\mathbf{x}\mid\mathbf{x}_1)=\mathcal{N}\big((1-t)\mathbf{x}_0+t\mathbf{x}_1,\sigma_t^2\mathbf{I}\big),
$$

条件向量场（期望意义下）：

$$
\boxed{\;u_t(\mathbf{x}\mid\mathbf{x}_1)=\mathbf{x}_1-\mathbf{x}_0\;}\qquad(\text{OT 直线；FM 论文 Eq. (20) 形式，}\sigma_{\min}\to0)
$$

**Rectified Flow（Liu et al. 2022/2023）** 直接用（通常独立的）配对 $(\mathbf{x}_0,\mathbf{x}_1)$ 构造"插值 + 直线速度"：

$$
\frac{d\mathbf{x}_t}{dt}=v_t(\mathbf{x}_t),\quad \mathcal{L}_{\mathrm{RF}}=\mathbb{E}_{t,\mathbf{x}_0,\mathbf{x}_1}\|v_t((1-t)\mathbf{x}_0+t\mathbf{x}_1)-(\mathbf{x}_1-\mathbf{x}_0)\|^2.
$$

它学到的边缘向量场（条件流期望）正是

$$
u_t(\mathbf{x})=\mathbb{E}\big[\mathbf{x}_1-\mathbf{x}_0\mid \mathbf{x}_t=\mathbf{x}\big],
$$

即对各插值直线速度按后验 $\pi(\mathbf{x}_0,\mathbf{x}_1\mid\mathbf{x}_t)$ 加权平均——不同直线在相交处被"整流"，边缘路径的轨迹被拉直（reflow：用学到的流重新配对再训，耦合趋于确定性、近 OT，速度场进一步平直），从而 ODE 求解步数大减。**〔注〕** Rectified Flow 的直线插值若用**独立耦合**，其边缘路径一般带交叉、并非 OT；FM-OT 路径强调先求 OT 再插值，二者在"使用 OT 耦合"时重合。Rectified Flow 的等价传输代价下降（期望动能 $\mathbb{E}\|v_t\|^2$）等性质见 Liu 论文定理；FM 论文则证明 OT 条件路径给出直线轨迹。

### 5.5 FM 与扩散的统一

Lipman 明确指出：**扩散的概率流 ODE 是 CNF/高斯概率路径族的一个特例**；用 FM 目标训练扩散路径比 score matching 更稳健（回归对象是 $O(1)$ 的向量场，而非随噪声尺度剧烈缩放的 score）。FM 把"先指定 SDE 再推 score"的范式换成"直接指定概率路径与向量场"，并允许非扩散（OT/直线）路径。

---

## 6. Consistency Models：自洽边界条件与连续时间一致性函数

**出处**：Y. Song, P. Dhariwal, M. Chen, I. Sutskever, *Consistency Models*, ICML 2023, arXiv:2303.01469。

### 6.1 概率流 ODE 基础

从第 3.5 节的 PF ODE 出发（CM 论文 Eq. (1)–(2)）：

$$
d\mathbf{x}_t=\boldsymbol\mu(\mathbf{x}_t,t)dt+\sigma(t)d\mathbf{w}_t,\qquad
\frac{d\mathbf{x}_t}{dt}=\boldsymbol\mu(\mathbf{x}_t,t)-\tfrac12\sigma(t)^2\nabla\log p_t(\mathbf{x}_t).
$$

论文采用 Karras (EDM) 参数化：$\boldsymbol\mu=\mathbf{0},\ \sigma(t)=\sqrt{2t}$，于是 $p_t=p_{\mathrm{data}}*\mathcal{N}(\mathbf{0},t^2\mathbf{I})$，PF ODE 为

$$
\frac{d\mathbf{x}_t}{dt}=-t\,\mathbf{s}_\phi(\mathbf{x}_t,t).
$$

$T=80,\epsilon=0.002$，实际在 $[\epsilon,T]$ 上考虑轨迹。

### 6.2 一致性函数与自洽性（定义）

给定 PF ODE 的一条解轨迹 $\{\mathbf{x}_t\}_{t\in[\epsilon,T]}$，定义**一致性函数** $f:(\mathbf{x}_t,t)\mapsto\mathbf{x}_\epsilon$（轨迹起点）。其核心性质——**自洽（self-consistency）**：

$$
\boxed{\;f(\mathbf{x}_t,t)=f(\mathbf{x}_{t'},t'),\qquad \forall t,t'\in[\epsilon,T]\ \text{（只要两点在同一 PF 轨迹上）}\;}
$$

即整条 ODE 轨迹上任意两点映射到同一个 $\mathbf{x}_\epsilon$。模型 $f_\theta$ 即对此函数的估计；不要求可逆（区别于 neural flows）。

### 6.3 边界条件（boundary condition）

在轨迹起点 $t=\epsilon$ 处，函数必须是恒等映射（CM 论文 Eq. (4) 前的约束）：

$$
\boxed{\;f(\mathbf{x}_\epsilon,\epsilon)=\mathbf{x}_\epsilon\;}
$$

这是最关键也最受限的架构约束，论文给出两种"近乎免费"的实现：

1. 分段参数化：$f_\theta(\mathbf{x},t)=\mathbf{x}$ 若 $t=\epsilon$；否则 $F_\theta(\mathbf{x},t)$。
2. 跳连参数化（Eq. (5)）：$f_\theta(\mathbf{x},t)=c_{\mathrm{skip}}(t)\mathbf{x}+c_{\mathrm{out}}(t)F_\theta(\mathbf{x},t)$，选取 $c_{\mathrm{skip}}(\epsilon)=1,c_{\mathrm{out}}(\epsilon)=0$（论文沿用 Karras 的连续权重选择）。

**连续时间视角**：对任意 $t\in[\epsilon,T]$，$f(\cdot,t)$ 都把该时刻的边缘点送回 $\mathbf{x}_\epsilon\approx\mathbf{x}_0$；因此一步网络求值 $f_\theta(\mathbf{x}_T,T)$ 即可从噪声生成数据。自洽性是连续时间（整条轨迹）上的约束，训练时通过相邻时间点对来逼近。

### 6.4 训练：蒸馏与独立训练

- **Consistency Distillation（CD，Eq. 在 §4）**：用预训练 score 模型 $\mathbf{s}_\phi$ 经一步 ODE 求解器（Euler/Heun）把 $\mathbf{x}_{t_{n+1}}$ 推到相邻的 $\hat{\mathbf{x}}_{t_n}$，构成同一轨迹上的相邻点对，最小化

$$
\mathcal{L}_{\mathrm{CD}}=\mathbb{E}\big[d\big(f_\theta(\mathbf{x}_{t_{n+1}},t_{n+1}),\,f_{\theta^-}(\hat{\mathbf{x}}_{t_n},t_n)\big)\big],
$$

$\theta^-$ 为 stop-gradient 的目标网络（EMA），$d$ 为 $\ell_2$ / LPIPS / Pseudo-Huber 度量。多步离散相邻对在极限下逼近连续自洽条件。

- **Consistency Training（CT，§5）**：不用预训练 score，直接用已知的边缘 $\mathbf{x}_t=\mathbf{x}_0+t\mathbf{z}$（EDM 参数化下）采样点对，估计 $\hat{\mathbf{x}}_{t_n}$（无需 $\mathbf{s}_\phi$），使 consistency model 成为独立的生成模型族。

多步采样：从 $\mathbf{x}_T$ 出发，交替 $f_\theta$ 输出与重新加噪到更粗时间步，可在计算量与质量间权衡。

---

## 7. 统一关系总图

1. **共同对象**：一条数据↔高斯先验的概率路径 $p_t$。区别只在路径是离散链还是连续、是随机 SDE 还是确定性 ODE，以及训练用什么等价目标。

2. **DDPM ↔ DSM ↔ score SDE**：
   - DDPM 的 ELBO 中真正依赖参数的 KL 项 = 加权噪声预测（$L_{\mathrm{simple}}$ 为其去权重简化版）；
   - Tweedie 公式把噪声预测与 score 一一对应：$\mathbf{s}_\theta=-\boldsymbol\epsilon_\theta/\sqrt{1-\bar\alpha_t}$；
   - Song 2021 Eq. (1)/(3) 直接表明 NCSN 与 DDPM 的目标都是**加权去噪 score matching**；
   - 离散 DDPM/NCSN 是 VP/VE SDE 的离散化；Anderson 反向 SDE 是 DDPM 反向高斯核的连续严格版本。

3. **SDE ↔ ODE**：概率流 ODE 与 SDE 共享边缘密度（连续性方程 = Fokker–Planck），把随机采样与确定性采样、精确似然统一起来。

4. **DDIM ↔ 概率流 ODE**：DDIM 通过非马尔可夫前向在**同一边缘/同一训练权重**下换取不同反向链；$\eta=0$ 是确定性的概率流式采样，$\eta=1$ 回到 DDPM。说明"训练目标只锁边缘，采样过程有巨大自由度"。

5. **DSM ↔ Flow Matching（同一数学技巧）**：
   - DSM：条件去噪回归目标 $\equiv$ 对扰动边缘分布的显式 score matching（Vincent 恒等式）；
   - CFM：条件向量场回归目标 $\equiv$ 边缘 FM 目标（Lipman 定理 2）。
   - 扩散的概率流向量场是 FM 高斯条件路径的特例；FM 额外允许 OT/直线路径。

6. **Rectified Flow / FM-OT**：在 OT（或独立）耦合上做直线插值，边缘场 = 条件直线速度的后验期望 $u_t(\mathbf{x})=\mathbb{E}[\mathbf{x}_1-\mathbf{x}_0\mid\mathbf{x}_t=\mathbf{x}]$；reflow 迭代把路径拉直、逼近 W2 最优传输，从而少步 ODE 采样。它给出了"为什么直线路径采样更快"的几何解释（W2 位移插值）。

7. **Consistency Models**：站在概率流 ODE 之上，把"沿轨迹多步积分"压缩成"任意轨迹点 $\to$ 轨迹起点"的单步映射；自洽性（同轨迹等输出）是连续时间约束，边界条件 $f(\mathbf{x}_\epsilon,\epsilon)=\mathbf{x}_\epsilon$ 锚定数据端。它与 DDIM 的"同一起点多步长一致"思想相通，但用函数恒等约束实现真正的一步生成，并可由扩散蒸馏或独立训练。

**演化逻辑链**：

$$
\text{DDPM（变分/马尔可夫）}
\xrightarrow[\text{Tweedie}]{\text{噪声}\leftrightarrow\text{score}}
\text{Score/DSM}
\xrightarrow{N\to\infty}
\text{SDE（VE/VP）}\ \&\ \text{概率流 ODE}
\begin{cases}
\xrightarrow{\text{非马尔可夫同边缘}} \text{DDIM（确定性少步）}\\
\xrightarrow{\text{条件场}\equiv\text{边缘场}} \text{Flow Matching} \xrightarrow{\text{OT 直线}} \text{Rectified Flow}\\
\xrightarrow{\text{轨迹}\to\text{起点的恒等映射}} \text{Consistency Models}
\end{cases}
$$

---

## 8. 参考文献

1. Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020. arXiv:2006.11239.
2. Sohl-Dickstein et al. *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. ICML 2015.
3. Hyvärinen. *Estimation of Non-Normalized Statistical Models by Score Matching*. JMLR 2005.
4. Vincent. *A Connection Between Score Matching and Denoising Autoencoders*. Neural Computation 2011.
5. Song, Ermon. *Generative Modeling by Estimating Gradients of the Data Distribution*（NCSN）. NeurIPS 2019. arXiv:1907.05600.
6. Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole. *Score-Based Generative Modeling through SDEs*. ICLR 2021. arXiv:2011.13456.
7. Song, Meng, Ermon. *Denoising Diffusion Implicit Models*. ICLR 2021. arXiv:2010.02502.
8. Lipman, Chen, Ben-Hamu, Nickel, Le. *Flow Matching for Generative Modeling*. ICLR 2023. arXiv:2210.02747.
9. Liu, Gong, Liu. *Flow Straight and Fast: Rectified Flow*. ICLR 2023. arXiv:2209.03003.
10. Song, Dhariwal, Chen, Sutskever. *Consistency Models*. ICML 2023. arXiv:2303.01469.
11. Anderson. *Reverse-time Stochastic Difference Equations*. 1982；McCann. *A Convexity Principle for Interacting Gases*（OT 位移插值）. 1997；Karras et al. *EDM*. 2022（CM 使用的参数化）。

> **存疑/易误标注汇总**：① Vincent 论文正式年份为 2011（2008 为早期技术报告）；② DDPM 的 $\bar\alpha_t$、Song 2021 的 $\alpha_t$、DDIM 的 $\alpha_t$ 符号含义不同，本文已逐处换算；③ $L_{\mathrm{simple}}$ 不是严格 ELBO；④ 确定性 DDIM 与概率流 ODE 是"近亲/共享边缘"而非逐数值步恒等；⑤ Rectified Flow 用独立耦合时其路径不是 OT，只有采用 OT 耦合或 reflow 极限才逼近 OT。
