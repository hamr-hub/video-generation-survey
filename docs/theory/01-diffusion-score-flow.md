# 扩散模型、Score-Based 模型与 Flow Matching 的数学原理

> 本文系统梳理 DDPM、score matching、SDE 框架、flow matching / rectified flow、DDIM 与 consistency models 的数学推导。所有关键公式均对照原文；个别原文未直接给出、由本文补出的推导会显式标注"（本文推导）"或"（约定相关，待核）"。

## 0. 总览：三条等价的线索

生成建模的目标：给定数据 $x_0\sim q_{\text{data}}$，构造一条从简单分布（高斯噪声）$p_{\text{prior}}\approx\mathcal N(0,I)$ 到数据分布的可逆映射，并学习其"方向信息"。三套范式学习的是同一对象的不同参数化：

| 范式 | 学习目标 | 学到的"方向" | 代表论文 |
|---|---|---|---|
| 扩散（DDPM） | 变分下界（ELBO） | 噪声 $\epsilon_\theta(x_t,t)\approx\epsilon$ | Ho et al., 2020 |
| Score-based | 得分函数 $\nabla_x\log p_t(x)$ | $\nabla_{x_t}\log p_t(x_t)$ | Song & Ermon, 2019/2020；Song et al., 2021 |
| Flow matching | 向量场 $v_t(x)$ | 条件流向量场的边缘期望 | Lipman et al., 2023；Liu et al., 2023 |

核心恒等式（噪声预测 ↔ score，见 §1.5）：

$$
\nabla_{x_t}\log p_t(x_t)=-\frac{1}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t,t).
$$

而 flow matching 的速度场 $v_t$ 在高斯插值路径下又与 $\epsilon_\theta$ 只差一个确定性线性变换（§5.3）。因此三者在数学上是统一的。

---

## 1. DDPM：去噪扩散概率模型

论文：Ho, Jain, Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS 2020，arXiv:2006.11239。理论框架源自 Sohl-Dickstein et al., 2015（arXiv:1503.03585）。

### 1.1 前向过程及其闭式解

给定方差表 $\beta_1,\dots,\beta_T$（线性 schedule，DDPM 中 $\beta_1=10^{-4},\beta_T=0.02$），定义马尔可夫前向链：

$$
q(x_t\mid x_{t-1})=\mathcal N\!\bigl(x_t;\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I\bigr).
$$

记 $\alpha_t=1-\beta_t$，$\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$。反复套用"缩放高斯 + 独立高斯"重参数化：

$$
x_t=\sqrt{\alpha_t}\,x_{t-1}+\sqrt{\beta_t}\,\epsilon_{t-1},\qquad \epsilon_{t-1}\sim\mathcal N(0,I),
$$

展开并合并独立噪声（$\sqrt{\alpha_t}\epsilon_{t-1}$ 与更早的噪声仍为独立高斯，方差相加），得到**任意时刻的闭式采样**：

$$
\boxed{\;q(x_t\mid x_0)=\mathcal N\!\bigl(x_t;\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I\bigr)\;}
$$

即

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal N(0,I).
$$

变量含义：$x_0$ 干净数据；$x_t$ 第 $t$ 步噪声样本；$\bar\alpha_t$ 信号保留比例，随 $t$ 单调降至接近 0，故 $q(x_T\mid x_0)\approx\mathcal N(0,I)$。

### 1.2 反向过程与 ELBO

反向链用可学习高斯参数化（DDPM 固定方差，见下）：

$$
p_\theta(x_{t-1}\mid x_t)=\mathcal N\!\bigl(x_{t-1};\mu_\theta(x_t,t),\,\sigma_t^2 I\bigr),\qquad p_\theta(x_T)=\mathcal N(0,I).
$$

对边缘对数似然用 Jensen / KL 拆解，得到**变分下界**（DDPM 式 (5)，Sohl-Dickstein 推导）：

$$
\mathbb E_q[-\log p_\theta(x_0)]\le \mathbb E_q\!\left[-\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\right]=L_{\text{VLB}},
$$

$$
L_{\text{VLB}}=
\underbrace{D_{\mathrm{KL}}(q(x_T\mid x_0)\|p(x_T))}_{L_T\ \text{（无参数，常数）}}
+\sum_{t>1}\underbrace{D_{\mathrm{KL}}(q(x_{t-1}\mid x_t,x_0)\|p_\theta(x_{t-1}\mid x_t))}_{L_{t-1}}
+\underbrace{(-\mathbb E_q\log p_\theta(x_0\mid x_1))}_{L_0}.
$$

推导要点：把 $\log p_\theta(x_0)=\log\int p_\theta(x_{0:T})dx_{1:T}$ 插入提议分布 $q(x_{1:T}\mid x_0)$，用 $\mathrm{KL}\ge 0$；再利用两条链的马尔可夫性把对数比值约分、按相邻时间点分组，即得各项 KL。

### 1.3 真实反向后验 $q(x_{t-1}\mid x_t,x_0)$

由贝叶斯公式与高斯乘积：

$$
q(x_{t-1}\mid x_t,x_0)=\mathcal N(x_{t-1};\tilde\mu_t(x_t,x_0),\tilde\beta_t I),
$$

$$
\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t,
\qquad
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t.
$$

（DDPM 式 (6)–(7)。）用 $x_0=\tfrac{1}{\sqrt{\bar\alpha_t}}(x_t-\sqrt{1-\bar\alpha_t}\,\epsilon)$ 代入，均值可写成 $x_t$ 与 $\epsilon$ 的线性式。

### 1.4 噪声预测参数化与简化损失

令网络预测噪声 $\epsilon_\theta(x_t,t)$，并设

$$
\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right).
$$

两个高斯之间的 KL（方差相同）只取决于均值差，代入后 $L_{t-1}$ 化简为：

$$
L_{t-1}=\mathbb E_{x_0,\epsilon}\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}
\|\epsilon-\epsilon_\theta(x_t,t)\|^2\right]+\text{常数}.
$$

Ho et al. 发现丢弃时间相关权重、均匀采样 $t$ 效果更好，得到**简化训练目标**（DDPM 式 (14) / Algorithm 1）：

$$
\boxed{\;L_{\text{simple}}=\mathbb E_{t,x_0,\epsilon}\left[\|\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\,t)\|^2\right]\;}
$$

采样时 $\sigma_t^2$ 可取 $\beta_t$ 或 $\tilde\beta_t$（DDPM 实验两者差异很小），反向更新：

$$
x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)+\sigma_t\mathbf z,\quad \mathbf z\sim\mathcal N(0,I),\ t>1.
$$

### 1.5 噪声预测即 score

由高斯密度直接求导：$q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$，故

$$
\nabla_{x_t}\log q(x_t\mid x_0)=-\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}
=-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}.
$$

Tweedie 引理给出 $\nabla_{x_t}\log p_t(x_t)=\mathbb E[\nabla_{x_t}\log q(x_t\mid x_0)\mid x_t]$，于是最优网络满足

$$
\boxed{\;\epsilon_\theta(x_t,t)=-\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log p_t(x_t)\;}
$$

这是连接 DDPM 与 score-based 模型的桥梁。

---

## 2. Score Matching 与去噪 Score Matching

出处：Hyvärinen, *Estimation of Non-Normalized Statistical Models by Score Matching*, JMLR 2005；Vincent, *A Connection Between Score Matching and Denoising Autoencoders*, Neural Computation 2011（技术报告 2008）；Song & Ermon, *Generative Modeling by Estimating Gradients of the Data Distribution*, NeurIPS 2019（arXiv:1907.05600），及其 *Improved Techniques for Training Score-Based Generative Models*, NeurIPS 2020（arXiv:2006.09011）。

### 2.1 Score matching 目标

数据密度 $p_{\text{data}}(x)$ 的得分定义为 $s(x):=\nabla_x\log p_{\text{data}}(x)$。用模型 $s_\theta(x)$ 拟合，期望 Fisher 散度目标为：

$$
\ell(\theta)=\mathbb E_{p_{\text{data}}}[\tfrac12\|s_\theta(x)-s(x)\|^2]
=\mathbb E_{p_{\text{data}}}[\tfrac12\|s_\theta(x)\|^2+\operatorname{tr}\nabla_x s_\theta(x)]+\text{常数}.
$$

右式由分部积分得到（Hyvärinen 2005），无需知道 $\nabla\log p_{\text{data}}$，但要求二阶导（迹项），且在数据流形外（图像等低维支撑）不稳定。

### 2.2 去噪 Score Matching（DSM）

Vincent 的关键结论：**对加噪分布做 score matching 等价于对"噪声条件得分"做显式回归**。设扰动核 $q(\tilde x\mid x)=\mathcal N(\tilde x;x,\sigma^2 I)$，扰动后分布 $q_\sigma(\tilde x)=\int q_\sigma(\tilde x\mid x)p_{\text{data}}(x)dx$，则

$$
\mathbb E_{q_\sigma(\tilde x)}\|s_\theta(\tilde x)-\nabla_{\tilde x}\log q_\sigma(\tilde x)\|^2
=
\mathbb E_{p_{\text{data}}(x)q_\sigma(\tilde x\mid x)}
\|s_\theta(\tilde x)-\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)\|^2+\text{常数}.
$$

而高斯核的条件得分有闭式：

$$
\nabla_{\tilde x}\log q_\sigma(\tilde x\mid x)=-\frac{\tilde x-x}{\sigma^2}.
$$

于是训练只需采样 $(x,\tilde x=x+\sigma\epsilon)$ 做回归。Song & Ermon（2019）把它推广到**噪声尺度序列** $\sigma_1<\cdots<\sigma_L$（连续情形用 $\sigma$ 的分布采样），联合目标：

$$
\boxed{\;\mathcal L_{\text{DSM}}(\theta)=
\mathbb E_{x\sim p_{\text{data}},\ \sigma\sim p(\sigma),\ \epsilon\sim\mathcal N(0,I)}
\left[\sigma^2\|s_\theta(x+\sigma\epsilon,\sigma)-(-\epsilon/\sigma)\|^2\right]\;}
$$

权重 $\sigma^2$ 使各尺度损失量纲一致（原文目标形式；缩放权重不改变最优解）。$s_\theta$ 用带噪声条件的网络（如 $\sigma$ 嵌入）实现，即 Noise Conditional Score Network（NCSN）。采样用 Langevin 动力学在每个噪声尺度上迭代：

$$
x_{i+1}=x_i+\eta\,s_\theta(x_i,\sigma)+\sqrt{2\eta}\,\mathbf z_i,
$$

定态分布正比于 $p_\sigma(x)\propto\exp(\int s_\theta dx)$；从大尺度到小尺度逐级退火（annealed Langevin / predictor-corrector 的雏形）。

---

## 3. 基于 SDE 的连续时间框架

论文：Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole, *Score-Based Generative Modeling through Stochastic Differential Equations*, ICLR 2021，arXiv:2011.13456。

### 3.1 前向 SDE 与 Fokker–Planck 方程

连续扩散过程 $x(t)$，$t\in[0,1]$，$x(0)\sim p_{\text{data}}$，满足 Itô SDE：

$$
\boxed{\;dx=f(x,t)dt+g(t)dw\;}
$$

$f$ 漂移向量（drift），$g$ 标量扩散系数，$w$ 标准布朗运动。对应密度 $p_t(x)$ 满足 **Fokker–Planck（正向 Kolmogorov）方程**：

$$
\frac{\partial p_t(x)}{\partial t}
=-\nabla\cdot(f(x,t)p_t(x))+\frac12 g(t)^2\sum_i\frac{\partial^2 p_t(x)}{\partial x_i^2}
=-\nabla\cdot\left(fp_t-\frac12 g^2\nabla p_t\right).
$$

要求 $p_1\approx\mathcal N(0,I)$（或其他已知先验）。

### 3.2 反向 SDE（Anderson 逆时扩散）

Anderson (1982) 给出随机过程的时间反转：给定前向 SDE，反向过程（时间从 $1$ 到 $0$）满足

$$
\boxed{\;dx=[f(x,t)-g(t)^2\nabla_x\log p_t(x)]dt+g(t)d\bar w\;}
$$

其中 $dt<0$（反向积分），$\bar w$ 是反向时间的标准布朗运动。该方程中唯一未知量是各时刻的 score $\nabla_x\log p_t$——恰好可用 §2 的 DSM 训练得到。论文还给出等价的**概率流 ODE**（确定性采样、可逆、可算精确对数似然）：

$$
dx=\left[f(x,t)-\frac12 g(t)^2\nabla_x\log p_t(x)\right]dt.
$$

ODE 与 SDE 具有相同的边缘密度 $p_t$（ODE 对应的概率流 $fp_t-\tfrac12g^2\nabla p_t$ 给出与 SDE 相同的 Fokker–Planck 方程）。

### 3.3 VE 与 VP 两个实例

**Variance Exploding（VE，对应 NCSN/退火 Langevin）**：无漂移 $f=0$，扩散系数随噪声尺度膨胀：

$$
dx=\sqrt{\frac{d[\sigma(t)^2]}{dt}}\,dw,
\qquad x(t)=x(0)+\sigma(t)\epsilon,\quad p_t=\text{数据}+\sigma(t)\text{ 高斯噪声},
$$

方差 $\sigma(t)^2$ 单调"爆炸"至很大，故名。其离散对应即 §2 的多尺度噪声。

**Variance Preserving（VP，对应 DDPM 的连续极限）**：

$$
dx=-\frac12\beta(t)x\,dt+\sqrt{\beta(t)}dw.
$$

该线性 SDE 有闭式解（Ornstein–Uhlenbeck）：

$$
x(t)=\sqrt{e^{-\int_0^t\beta(s)ds}}\,x(0)
+\sqrt{1-e^{-\int_0^t\beta(s)ds}}\,\epsilon,
$$

与 DDPM 的 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ 完全对应（$\bar\alpha_t=e^{-\int_0^t\beta(s)ds}$，离散时 $\bar\alpha_t=\prod(1-\beta_i)\approx e^{-\sum\beta_i}$）。边缘方差保持有界且 $p_1\approx\mathcal N(0,I)$，故名"方差保持"。VE/VP 加上 sub-VP（论文中随机部分方差更紧的变体，用于概率流 ODE 加速）构成连续扩散的主要参数化。

### 3.4 采样：Predictor–Corrector

论文给出统一采样器：数值 SDE/ODE 求解器（Euler–Maruyama、概率流 ODE 等）作 predictor，Langevin / HMC 作 corrector 把样本修正到 $p_t$，PC 采样器在 VE/VP 上都适用。

---

## 4. DDIM：确定性加速采样

论文：Song, Meng, Ermon, *Denoising Diffusion Implicit Models*, ICLR 2021，arXiv:2010.02502。

DDIM 定义**非马尔可夫**前向过程（其边缘 $q(x_t\mid x_0)$ 与 DDPM 完全相同，故同一训练好的 $\epsilon_\theta$ 可直接使用），其反向更新规则（论文式 (16)，记号换成 $\bar\alpha_t$）：

$$
x_{t-1}=\sqrt{\bar\alpha_{t-1}}\underbrace{\frac{x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}}_{\text{预测的 }x_0}
+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta(x_t,t)+\sigma_t\mathbf z.
$$

- $\sigma_t=0$：完全确定性更新，即 **DDIM**；映射可逆，是概率流 ODE 的一种离散格式（后来证明 DDIM 是概率流 ODE 的一阶数值解）。
- $\sigma_t=\sqrt{\tfrac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}}\sqrt{\beta_t}$：退化为 DDPM 随机采样。
- 因为只依赖 $\bar\alpha_t$，可取任意时间子序列做少步采样（20–50 步即可），是扩散加速的起点，也是 DDPM↔ODE↔flow 的纽带。

---

## 5. Flow Matching 与 Rectified Flow

论文：Lipman, Chen, Ben-Hamu, Nickel, Le, *Flow Matching for Generative Modeling*, ICLR 2023，arXiv:2210.02747；Liu, Gong, Liu, *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*, ICLR 2023（arXiv:2209.03003，任务中 "StnLiu" 即 rectified flow）。

### 5.1 连续归一化流与向量场

设时间依赖向量场 $v_t(x)$ 定义 ODE：

$$
\frac{d}{dt}\phi_t(x)=v_t(\phi_t(x)),\qquad \phi_0(x)=x.
$$

若 $x_0\sim p_0$（噪声），则 $x_t=\phi_t(x_0)$ 的密度 $p_t$ 满足连续性方程：

$$
\frac{\partial p_t}{\partial t}+\nabla\cdot(p_t v_t)=0.
$$

直接拟合向量场的传统 CNF 损失需要对 $\log p_t$ 求导，训练昂贵且不稳定。

### 5.2 Conditional Flow Matching：可回归的目标

给定数据样本 $x_1\sim p_{\text{data}}$，构造**条件概率路径** $q_t(x\mid x_1)$ 及对应条件向量场 $u_t(x\mid x_1)$，二者满足条件连续性方程。定义条件流匹配目标：

$$
\mathcal L_{\text{CFM}}(\theta)=
\mathbb E_{t,x_1,\,q_t(x\mid x_1)}\|v_\theta(t,x)-u_t(x\mid x_1)\|^2.
$$

关键定理（Lipman 2023, Prop. 1）：该二次目标的**总体最优解** $v_\theta=v_t^*$ 恰好是边缘向量场

$$
\boxed{\;v_t^*(x)=\mathbb E_{q(x_1\mid x)}[u_t(x\mid x_1)]
=\frac{\mathbb E_{p_{\text{data}}(x_1)}[q_t(x\mid x_1)u_t(x\mid x_1)]}{p_t(x)}\;}
$$

即"条件流的边缘期望"。证明：平方回归的最优解是目标在后验下的条件均值；展开 $p_t(x)=\int p_{\text{data}}(x_1)q_t(x\mid x_1)dx_1$ 即得。$u_t$ 有闭式、无需计算 $\nabla\log p_t$，故 CFM 是免仿真（simulation-free）的简单回归损失。

### 5.3 高斯插值路径（OT-CFM）

取 $x_0\sim p_0=\mathcal N(0,I)$，$x_1\sim p_{\text{data}}$，线性插值：

$$
x_t=(1-t)x_0+t x_1,\qquad u_t(x\mid x_0,x_1)=x_1-x_0.
$$

独立采样配对时路径可能交叉；用 mini-batch 最优传输配对（欧氏距离下求置换 $\min_\pi\sum\|x_0-x_1\|^2$，Sinkhorn）即 **OT-CFM**，路径更直、更少交叉：

$$
\mathcal L_{\text{OT-CFM}}=\mathbb E_{t,(x_0,x_1)\sim\pi}\|v_\theta(t,(1-t)x_0+t x_1)-(x_1-x_0)\|^2.
$$

与扩散的关系（本文对照推导）：DDPM 路径 $x_t=\sqrt{\bar\alpha_t}x_1+\sqrt{1-\bar\alpha_t}\,\epsilon$ 对 $t$ 求导即得该路径的目标速度场，它与 $\epsilon_\theta$ 成线性关系；取线性 schedule 的适当极限即得到 rectified flow 的直线路径。所以"预测噪声"与"预测速度"只是路径参数化不同。

### 5.4 Rectified Flow：拉直与最优传输

Rectified Flow（Liu et al. 2022/2023）从独立采样的 $X_0\sim p_0$、$X_1\sim p_1$ 出发，沿直线插值：

$$
X_t=(1-t)X_0+tX_1,\qquad \dot X_t=X_1-X_0.
$$

学习速度场 $v_t$ 拟合 $\mathbb E[X_1-X_0\mid X_t]$（与 CFM 相同的条件均值回归，即一次 "reflow"）。核心结论：

1. **直线化定理**：整流映射 $Z_1=z_0+\int_0^1 v_t(z_t)dt$ 诱导的耦合 $(Z_0,Z_1)$ 比原线性插值耦合在二阶意义下更直：

$$
\mathbb E\|Z_1-Z_0\|^2\le\mathbb E\|X_1-X_0\|^2.
$$

（位移方差下降；原文 Proposition，由 Jensen/凸性得到，精确常数建议回查原文。）
2. 反复 reflow（用上一轮模型的输入–输出配对再训）路径趋近直线，ODE 可用大步长少步求解，1 步即蒸馏为 GAN 式生成器。
3. 当两分布间最优传输映射唯一时，直线插值对应位移插值（displacement interpolation），整流极限趋向 OT 耦合——为"扩散蒸馏出少步/单步模型"提供理论依据。Stable Diffusion 3 即采用 rectified flow 训练。

---

## 6. Consistency Models

论文：Song, Dhariwal, Chen, Sutskever, *Consistency Models*, ICML 2023，arXiv:2303.01469。

### 6.1 PF-ODE 轨迹与自洽边界条件

从 §3 的概率流 ODE 出发：

$$
\frac{dx_t}{dt}=v(x,t)=f(x,t)-\frac12 g(t)^2 s_\phi(x,t).
$$

记从 $(x_t,t)$ 沿 ODE 积分到端点 $t=\epsilon$（避开 0 奇点）的解为 $\hat x_\epsilon$。一致性函数 $F_\theta(x,t)$ 要求：**同一条 ODE 轨迹上任意点的输出都相同**（自洽性），且在噪声端点等于输入：

$$
\boxed{\;F_\theta(x,t)=F_\theta(x',t')\ \text{若 }(x,t),(x',t')\text{ 在同一 PF-ODE 轨迹上},\qquad F_\theta(x,\epsilon)=x.\;}
$$

边界处恒等使轨迹公共输出落在数据端：对任意 $x_t$，$F_\theta(x_t,t)\approx\hat x_\epsilon$，即一步去噪。

### 6.2 连续时间一致性损失

沿轨迹取相邻两点（用反向 ODE 的 $\hat x_{t_{n+1}}\to\hat x_{t_n}$ 构造）。原文主算法以离散 $n$ 给出，其连续时间推广可写为：

$$
\mathcal L_{\text{CT}}(\theta,\theta^-)=
\mathbb E_{t\sim\mathcal U[\epsilon,T],\,x_t\sim p_t}
\left[\lambda(t)\,d(F_\theta(x_t,t),\,F_{\theta^-}(\hat x_{t'},t'))\right],
$$

其中 $t'=t+\Delta t$ 为沿 ODE 的邻近时刻，$\hat x_{t'}=x_t+\int_t^{t'}v(x,s)ds$（一步 ODE 估计，如 Heun），$\theta^-$ 为 EMA 目标网络（stop-gradient），$d$ 可取 $L_2$、LPIPS 等；边界条件通过参数化

$$
F_\theta(x,t)=c_{\text{skip}}(t)x+c_{\text{out}}(t)c_\theta(x,t),\qquad c_{\text{skip}}(\epsilon)=1,\ c_{\text{out}}(\epsilon)=0
$$

精确满足。训练方式有 consistency distillation（教师提供 score/ODE 步）与独立 consistency training（自估 $v$，无需预训练扩散模型）。采样为单步 $x_0\leftarrow F_\theta(x_T,T)$，也可多步提升质量。连续时间损失权重 $\lambda(t)$ 的具体形式在原文中以离散记号给出，本文连续化写法属标准推广，引用时建议回查原文定理编号。

---

## 7. 统一视角小结

1. **同一根 SDE，三种读法**：反向 SDE 需要 score（§3）；score 在 VP/DDPM 参数化下等价于噪声预测（§1.5、§2.2）；概率流 ODE 是确定性对偶，DDIM 是其离散格式（§4），flow matching 直接回归该 ODE 的速度场（§5）。
2. **目标函数统一为加权回归**：

$$
\mathbb E[w(t)\|\epsilon-\epsilon_\theta(x_t,t)\|^2]
\longleftrightarrow
\mathbb E[\sigma^2\|s_\theta+\epsilon/\sigma\|^2]
\longleftrightarrow
\mathbb E[\|v_\theta(x_t,t)-u_t\|^2].
$$

三者通过路径参数化（$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ 或直线插值）与链式求导互相线性变换。
3. **采样统一为 ODE/SDE 积分**：DDPM（随机 SDE）、DDIM/概率流 ODE（确定性）、rectified flow（被拉直的 ODE）、consistency models（ODE 轨迹上的自洽一步映射）是同一积分问题从多步到单步的不同解法。
4. **最优传输贯穿**：OT-CFM 用 OT 配对减少路径交叉；rectified flow 经 reflow 趋近 OT 耦合；这解释了扩散模型可被蒸馏为少步甚至单步生成器。

## 8. 参考文献（含 arXiv）

1. Ho, Jain, Abbeel. Denoising Diffusion Probabilistic Models. 2020. arXiv:2006.11239.
2. Sohl-Dickstein et al. Deep Unsupervised Learning using Nonequilibrium Thermodynamics. 2015. arXiv:1503.03585.
3. Hyvärinen. Estimation of Non-Normalized Statistical Models by Score Matching. JMLR 2005.
4. Vincent. A Connection Between Score Matching and Denoising Autoencoders. Neural Computation 2011.
5. Song & Ermon. Generative Modeling by Estimating Gradients of the Data Distribution. 2019. arXiv:1907.05600；改进版 arXiv:2006.09011.
6. Song et al. Score-Based Generative Modeling through SDEs. 2021. arXiv:2011.13456.
7. Song, Meng, Ermon. Denoising Diffusion Implicit Models. 2021. arXiv:2010.02502.
8. Lipman et al. Flow Matching for Generative Modeling. 2023. arXiv:2210.02747.
9. Liu, Gong, Liu. Flow Straight and Fast: Rectified Flow. 2023. arXiv:2209.03003.
10. Song et al. Consistency Models. 2023. arXiv:2303.01469.
11. Anderson. Reverse-time computation of diffusion processes. 1982（反向 SDE 的随机分析出处，转引自 2011.13456）。

> 核对说明：DDPM 式 (5)–(7)、(14)，DSM 回归等价式（Vincent 2011），SDE 论文的 VE/VP 参数化，CFM Prop. 1（条件期望最优解），rectified flow 拉直不等式，consistency models 的边界条件与参数化（Sec. 3–4）均按原文记号核对。Rectified flow 位移方差不等式的精确常数、consistency 连续时间损失权重 $\lambda(t)$ 在原文以离散记号给出，本文连续化写法属标准推广并已标注。