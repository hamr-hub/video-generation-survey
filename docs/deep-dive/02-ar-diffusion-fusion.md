# 扩散模型与自回归模型融合方向深度调研

> 直接检索 arXiv / OpenReview / 项目主页等一手资料；未能原文验证处明确标注。检索时间 2026-10。

## 0. 互补性概览
| 维度 | 自回归 AR | 扩散/流匹配 |
|---|---|---|
| 强项 | 长程语义规划、组合性、可变长度、KV cache、与LLM同构 | 高保真细节纹理、并行去噪、质量上限 |
| 弱项 | 误差累积、串行慢、受tokenizer瓶颈 | 全序列二次复杂度、固定长度、长时一致性弱 |

融合主线：AR 管"语义骨架/时间因果"，扩散管"像素血肉/局部细节"。

## 1. ARLON
- **机构**：Microsoft（合作KAIST/港中文/华中科大）；arXiv:2410.20502（v3 2025-04）。
- **机制**：①3D latent VQ-VAE（压缩4×8×8）把DiT连续latent量化为离散AR codes；②decoder-only AR按文本自回归生成全片粗粒度码；③语义注入比较MLP/adaptive-norm/ControlNet，采用adaptive norm分段送入DiT；④抗AR噪声：用压缩率更高的另一VQ-VAE的更粗token训练DiT + uncertainty sampling，解决teacher-forcing→AR推理的exposure bias。扩散式 $X^t=\sqrt{\alpha_t}X^0+\sqrt{1-\alpha_t}\epsilon$。
- **数字**：VBench 11项中**8项优于OpenSora-V1.2**，动态程度/美学显著提升；AR码初始化后DiT仅需**5–10步**（基线30步）；单A100 68帧：47s → 11–19s(DiT)+6s(AR)，**提速47–64%**，附录报推理速度+48–49%、FLOPs降约64%；支持渐进prompt，600帧长视频开源最优。
- **局限**：建在OpenSora-V1.2上画质上限受限；2K下AR码过长不可行；手部细节失真。
- **关系**："AR规划+扩散精修"松散级联代表，BLIP3-o/Mammoth2同动机。

## 2. Pyramid Flow（ICLR 2025）
- **机构**：北大、快手、NTU；arXiv:2410.05954；开源。
- **机制**：①**空间金字塔流**：去噪轨迹重解释为L级(=3)金字塔、仅末级全分辨率，跨分辨率分段流在"下采样+噪"与"上采样+干净"间插值，算力近1/L；②**单DiT单一FM目标端到端**统一生成与超分(2B MM-DiT)，噪声端点同向耦合；③**renoising(Eq.12–15)**：跨尺度上采样后缩放+校正高斯噪声匹配均值/协方差保概率路径连续；④**时间金字塔自回归**：逐段预测、历史用逐级压缩低分辨率金字塔并加腐蚀噪声；token 119,040→**15,360**；块级因果注意力，位置编码空间外推/时间插值。
- **数字**：仅**20.7k A100小时**训出768p/24fps/5s(最长10s)；对比Open-Sora1.2(4.8k Ascend+37.8k H100h仅97帧)省一半多算力且更优；VBench总分**81.72**、质量分**84.74**(超Gen-3 84.11)、平滑度99.12；EvalCrafter 244；5s/384p推理约**56s**；免微调可I2V。
- **局限**：语义分69.62偏低(粗合成caption)；384p推理仍偏慢。
- **关系**：同目标但比ARLON更紧(单模型连续流)；历史加噪呼应Diffusion Forcing，时间因果与MAGI-1同源而更早更轻。

## 3. MAGI-1
- **机构**：Sand.ai；arXiv:2505.13211；代码权重+MagiAttention开源。
- **机制**：视频切chunk(24帧≈1s)，chunk整体去噪、条件为全部前序chunk；**单调噪声时间表** $t_1\le t_2\le\cdots$ 强制因果(普通双向扩散为 $t_1=\cdots=t_n$)；**流水线并行**：chunk去噪到一定程度即启动下一个，最多**4 chunk并行**，峰值资源与片长无关；空间双向/时间因果主干；自研Transformer-VAE(空间×8/时间×4,PSNR36.55,解码12.28ms)；score三分解(无条件+时序+prompt)，时序引导权重1.5消接缝；Logit-Normal时间采样+prompt增强蒸馏(~7B)。
- **数字**：最大**24B**、上下文**4M token**；VBench-I2V/Physics-IQ较前代显著提升复杂运动/主体/物理交互(**逐项数值在图表中，未获可逐字引用文本表，标注待核**)；单一预训练统一T2V/续写/I2V；支持chunk-wise prompt与可控镜头转场(KV range调节)。
- **局限**：依赖MagiAttention等专用基础设施；蒸馏模型有随播放累积的饱和失真。
- **关系**：Diffusion Forcing思想规模化；FlowCache直接加速对象；与SkyReels-V2同路线。

## 4. FlowCache（ICLR 2026）
- **机构**：厦门大学(纪荣嵘组)+字节跳动(一作实习)；arXiv:2602.10825。
- **机制**：AR模型同一时间步各chunk处于异构去噪阶段。Theorem 1(幂律调度+最优速度场)证明相邻步相对L1 $d_i(t)=\|F_i(t+1)-F_i(t)\|_1/\|F_i(t)\|_1$ 随去噪**单调增大**，Corollary 1得各chunk相似度不等；**chunkwise独立缓存**：$d_i(t)<\tau$ 复用否则重算，前k步不缓存；阈值MAGI 0.01/0.015、SkyReels 0.1/0.15；**联合重要性–冗余KV压缩**(段/帧/像素三层级)保持显存固定上界。
- **数字（可逐字引用）**：MAGI-1 **2.38×**、SkyReels-V2 **6.7×**；VBench变化 **+0.87 / −0.79**。
- **局限**：阈值/预热需逐模型手配；依赖chunk异构性前提。
- **关系**：training-free、正交于量化/蒸馏；比TeaCache/ToCa/DiCache首次适配AR chunk时间轴。

## 5. Masked/离散扩散（理论支线）
- **MDLM (NeurIPS2024, Cornell Tech, arXiv:2406.07524)**：仅mask吸收态前向，SUBS参数化+连续时间Rao-Blackwellized ELBO化简为MLM损失加权混合；encoder-only即可生成、采样快3–4×并支持半自回归。LM1B扩散SOTA(≤23.00，SEEDD≤32.79、D3PM≤76.90)，与AR差距14–25%；SAR Gen.PPL 27.18/89.3s，比SSD-LM快25–30×。
- **Block Diffusion (arXiv:2503.09573)**：块间AR、块内离散扩散，块大小1=AR、全序列=扩散；支持变长/KV cache；梯度方差估计+数据驱动噪声调度解释块大小1时扩散仍劣于AR。扩散类新SOTA。后续Set Diffusion(可变位置/长度token集、每步更新KV)与Scaling Beyond MDLM(1.7B,uniform-state在GSM8K反超)继续泛化。
- **先驱**：D3PM(2021,吸收态最稳)、MaskGIT(mask-predict并行)、VQ-Diffusion(微软,VQ latent离散扩散免AR累积误差)。

## 6. AR↔Diffusion 统一架构
- **Show-o (ICLR2025, NUS, 1.3B, arXiv:2408.12528)**：单Transformer按模态切换——文本AR、图像离散扩散(MaskGIT式)，统一prompting+omni-attention。COCO zero-shot FID **9.24**(优于GLIDE 5B/DALL·E2 6.5B)，GenEval **0.68**(比同级LDM高约0.24)，约50步出图。局限：离散token限制保真度/分辨率。
- **Diffusion Forcing (NeurIPS2024, MIT/IBM, arXiv:2407.01392)**：每token独立噪声(=部分掩码)，因果next-token模型按任意per-token调度，优化所有子序列似然ELBO；可超训练horizon稳定rollout，fruit-swap记忆任务约**80%**(基线近0)。是MAGI-1/Pyramid时间条件/SkyReels的直接理论源头；v2=History Guidance(DiT+latent)。
- **BLIP3-o (Salesforce, arXiv:2505.09568)**：DiT生成CLIP语义特征、"先理解后生成"顺序预训练，引用GPT-4o式混合架构猜测。
- **MammothModa2 (字节, 约13B, arXiv:2511.18262)**：串行AR(生成专家+MammothTok)→单流DiT，三层特征对齐，**端到端联合NTP+Flow Matching**+SFT/RL(DiffusionNFT)，约60M样本；GenEval **0.87**、DPGBench **87.2**、ImgEdit **4.06**，理解持平Qwen3-VL-8B。是ARLON级联的端到端成熟形态。
- **VideoPoet (Google, ICML2024, arXiv:2312.14125)**：纯AR多模态token混合目标预训练+适配，作为纯AR对照(不含扩散)。
- **Diagonal Distillation (ICLR2026 poster)**：5秒视频2.61s生成(最高约31FPS)，蒸馏加速AR支线，与FlowCache互补。

## 7. 苏剑林"分离式"相关（取证标注）
spaces.ac.cn/archives/10197 实为《"闭门造车"多模态思路浅谈（二）：自回归》(2024-07-08；kexue.fm镜像同编号)，同系列(一)《无损输入》(9984)。**本次spaces/kexue均反爬(forbidden；直curl仅124字节；Camofox未运行)，仅能从第三方索引确认标题/日期/主题(视觉自回归、多模态本质难度、世界模型)，不对其公式与"分离扩散"具体机制作任何断言**。"分离"思想的论文对应物即Show-o(按模态分目标)、ARLON/Mammoth2(AR与扩散分工)、Block Diffusion(块间AR块内扩散)。

## 8. 未能核实条目
①苏剑林10197正文细节；②"ECCV2026具身底座"具体论文；③"Parallel Decoding Distillation"确切出处；④MAGI-1逐项VBench/Physics数值；以上均未定位可逐字核验原文，不做断言。FlowCache"厦大&字节"已与署名核实一致。

## 9. 为何是下一代底座候选
1. **短板互消**：AR给长程规划/组合/变长/agent能力，扩散给高保真；同参数内做理解+生成+编辑+决策。
2. **统一多模态最短路径**：Show-o/BLIP3-o/Mammoth2证"文本AR、视觉扩散"可一模型覆盖VQA/T2I/I2V/续写而理解不掉队，契合GPT-4o式产品形态。
3. **复杂度二次→线性/常数**：chunk AR+KV cache使长视频峰值显存常数(MAGI-1)，叠加FlowCache(6.7×)与对角蒸馏(实时)，让无限长度/流式/世界模型/具身决策工程可行。
4. **训练效率**：金字塔token压缩约7.8×，20.7k A100小时出768p，单模型替代级联、知识共享。
5. **可证明的统一**：Diffusion Forcing(噪声=部分掩码)与MDLM(masked diffusion=MLM混合)从似然层证明二者是同一插值谱系两端。

## 10. 主流融合范式
| 范式 | 结构 | 代表 |
|---|---|---|
| A 松散级联(AR规划→扩散精修) | 两模型 | ARLON、BLIP3-o、Mammoth2(串行)、(推测)GPT-4o |
| B 单模型按模态切换目标 | 一Transformer | Show-o |
| C 因果chunk+扩散去噪(diffusion forcing系) | 块间单调/独立噪声、块内去噪 | Diffusion Forcing、MAGI-1、SkyReels-V2、Block Diffusion |
| D 多尺度/金字塔流统一 | 单DiT跨分辨率分段流+压缩历史 | Pyramid Flow |
| E 训练免费缓存/蒸馏加速层 | 不改训练 | FlowCache、Diagonal Distillation、TeaCache系 |
| 对照 | 单范式 | VideoPoet(纯AR)、Sora类全序列扩散 |

**判断**：范式C正成为视频/世界模型主干，A/B在多模态统一上收敛，D/E从训练与推理侧补成本；"因果chunk主干+金字塔省token+cache/蒸馏实时化+AR语义规划"构成最清晰的下一代底座技术栈。