# 推理加速深度调研：少步蒸馏 / 量化 / 缓存 / RSI 自动 kernel 优化

> 调研时间：2026-10。数字均来自论文/项目页检索；未核实者标注"查不到"。
> Sol-Engine(2606.23743)、Sol-H3(2609.35110)、HyperFlow、PDD(2607.26004)、AccelOpt(2511.15915)为2026年新稿。

## 1. Sol-Engine（arXiv:2606.23743）
- **机构**：NVIDIA Research, Efficient AI Team & Singapore Lab（含港大Ping Luo、Song Han）。
- **核心机制**：不新增单一算法，把5类技术——跨步cache、稀疏注意力、token剪枝、量化、kernel fusion——组织成agent-native加速栈。论点：最优配方高度依赖(model, hardware, serving config)三元组。流程：并行skill agents局部搜索（cache搜skip/补偿；sparse-attn搜层与pattern；quant搜分层/分时间步精度；fusion做GEMM+GELU、RoPE+norm）→ integrator全局组合（增益不可直接相乘、误差累积）→ human validator视觉反馈。全程training-free。
- **实测（B200端到端）**：Cosmos3-Super 64B 99.6→43.9s(**2.27x**)；LTX-2.3 22B 97.8→41.0s(**2.38x**)；SANA-Video 2B 29.4→10.6s(**2.77x**)；项目页标称2–3x。B300单卡Cosmos 2.56x。
- **质量代价**：VBench均分变化 +0.21% / −0.54% / −0.21%，近无损；LTX美学−2.09%。
- **局限**：终判仍靠人（PSNR奖励模糊、惩罚细节偏移）；搜索/ profiling成本高；training-free上限低于重训；配方不可迁移。
- 附录配方：Cosmos=TeaCache式残差重放(阈值1.15,start10,最多连中3)+首尾3步高精+中段NVFP4；LTX=固定skip(8of15)+stage2 PISA(密度0.1)+保留50%token+NVFP4仅FFN；SANA-Video=EasyCache(阈值0.1,50步跳约16)+BF16线性注意力+QKV合并+torch.compile。

## 2. Sol-H3（arXiv:2609.35110）
- **机构**：NVIDIA Research。
- **核心机制**：算法侧跨分辨率两阶段——低分辨率早期步定布局、高分辨率晚期步精修，阶段间可学习latent-to-latent映射，免VAE decode/reencode（作者承认草稿+精修是prior art，贡献在serving）。系统侧RSI循环搜fusion/布局/collective/传输精度，latency+数值一致双评，区分bit-exact/算术融合/近似三档，**fail-closed契约**禁止偷改步数。
- **实测**：8×GB200 5s含音频**1.434s**(3.5x实时)；单GB200 **6.23s,22.2x**；DGX Spark **56.2s**全程HBM、省约20%显存；摘要最高≈30x。B300项目页(50步vs4步)：5s 11.04x、10s 13.57x、15s 15.05x。跨VAE映射PSNR **30.618dB**/SSIM0.8962。
- **质量代价**：结构/身份/运动连续，仅轻微差异；完整VBench总分**查不到**。
- **局限**：两阶段改契约非同产物；映射需训练且有损；音频须全程存活。

## 3. SANA-Sprint（arXiv:2503.09641）
- **机构**：NVIDIA+MIT+清华+HuggingFace。
- **机制**：20步flow模型压到1–4步，混合双目标：sCM连续时间一致性蒸馏（参数化 f=cos(t)x−sin(t)σF）保教师对齐 + LADD latent对抗蒸馏提单步保真；统一step-adaptive。消融 sCM 8.93 / LADD 12.20 / 结合8.11 FID。
- **数字**：1步**FID7.59/GenEval0.74**，优于FLUX-schnell(7.94/0.71)，H100 0.1s vs 1.1s（约10x）。
- **代价/局限**：1步弱于4步/教师；仅T2I无视频验证；对抗训练不稳。

## 4. DMD / DMD2
- **DMD（CVPR2024，MIT+Adobe）**：分布级匹配，最小化加噪p_real/p_fake的近似KL，梯度=教师score−在线fake score；需regression损失。单步SD1.5 0.09s FID**11.49**。局限：数据构建贵、学生绑教师轨迹。
- **DMD2（arXiv:2405.14867，NeurIPS2024 Oral）**：①去regression+TTUR稳fake critic；②加GAN对真实数据；③多步训练消除train/inference失配。ImageNet64 FID**1.28**、COCO FID**8.35**（改善3.14、超50步教师），成本约降**500x**，SDXL可出megapixel。局限：训练复杂、极端单步多样性受损、视频需改造。

## 5. CFG蒸馏
- **GD（2210.03142，2022）**：学单模型匹配CFG组合输出再渐进蒸馏；全量微调不稳。
- **AGD（2503.07274，ETH Zürich）**：冻结基座，仅训约2% adapter（cross-attn/offset残差，条件含guidance scale），在**CFG轨迹**上训练，2前向→1，NFE减半；FID持平CFG且略优于GD，SDXL可单24GB卡蒸馏；具体FID表**查不到**。局限：图像域、需逐guidance区间学。
- 视频侧CFG由CausVid/PDD直接蒸入（guidance=1.0），Sol-H3另有block-0共享/guidance-prefix复用。

## 6. 4步代表：CausVid / FastH3 / H3-Turbo
- **CausVid（2412.07772，CVPR2025，MIT+Adobe+Cornell）**：双向DiT改因果自回归+视频DMD 4步；ODE初始化（缺则Frame 48.1不稳）+**非对称蒸馏**（双向教师→因果学生）抑误差累积+KV cache。首帧**1.3s**、**9.4FPS**，VBench-Long **84.27**；对比CogVideoX延迟约160x、吞吐16x。局限：架构改造、流式帧质量略逊。
- **FastH3（FastVideo）**：4步VSA/data-free稀疏蒸馏（V2为8步全量）；独立基准**查不到**，Sol-H3横评非4NFE首选。
- **MiniMax-H3-Turbo（ModelTC）/Larry Turbo LoRA**：4步联合音视频（约5x采样）。Sol-H3 3000对A/B（120 T2VA+120 Ref2VA，人评+VLM双盲）：**4NFE T2VA首选Larry Turbo v4与LightX2V FL2VA；4NFE Ref2VA首选阿里PAI PDD与LightX2V**。统一VBench数字**查不到**。

## 7. 8步代表：HyperFlow / PDD / FastHunyuan / Hyper-SD
- **HyperFlow（Video Rebirth，2026-09）**：data-free flow自蒸馏（基座自产自筛自蒸）+双时间(t,r)条件；49→**8 NFE**。LoRA rank256/316模块/2.8GB，覆盖t2va/fl2va/ref2va含音频。4×H200 约60s vs175s（**≈2.9–3x**）；Sol-H3评为**8NFE Ref2VA首选**。局限：自定义loader、许可排除美欧英韩、VBench总分**查不到**。
- **PDD（2607.26004）+阿里PAI Acc LoRA**：时间域分N区间按L块，一次前向预测块内全部mean velocity（教师RK目标），免JVP；变块长支持可变NFE。4–8NFE在LTX-2.3/Wan14B/Qwen-Image称SOTA且多样性更优（具体表**查不到**）。PAI：rank64+32个per-interval末层head，8/4步无CFG、音视频联合；**8NFE T2VA首选PAI PDD Acc8与官方VDN-H3 stage-dmd-250**。局限：契约严格(Euler/shift12,3)、需专用加载。
- **FastHunyuan/FastMochi（FastVideo v1.0,2024-12）**：PCM一致性蒸馏，约8x加速，首开视频DiT开源配方，近线性扩64GPU；统一对照数字**查不到**。
- **Hyper-SD（2404.13686，NeurIPS2024，ByteDance）**：轨迹分段一致性，1–8步SDXL/SD1.5 SOTA；视频迁移**查不到**。

## 8. TeaCache与缓存家族（2411.19108，CVPR2025，阿里）
- **机制**：training-free，时间步嵌入调制噪声输入做低成本差异信号+多项式拟合rescale，差异小则重放，自适应非均匀变化。
- **数字**：Open-Sora-Plan **4.41x、VBench−0.07%**（slow），fast 6.83x；Latte slow1.86x（VBench同分无损）/fast3.28x；OpenSora1.55–2.25x；HunyuanVideo 1.6x/2.1x、Mochi1.5/2.1x。
- **局限**：阈值逐模型调、非Euler(如res_2s)适配差。家族：PAB(均匀步)、TaylorSeer(Taylor预测,ICCV2025)、EasyCache/Less is Enough(2507.02860)、First-Block Cache、BWCache。

## 9. FP8/FP4量化
- **MXFP8/NVFP4（PyTorch博客2026-04，Meta+HF）**：16/32元素microscaling+选择性量化+CUDA Graphs。B200端到端最高**MXFP8 1.26x、NVFP4 1.68x**；Flux selective使MXFP8 LPIPS0.138→0.108、NVFP4 0.480→0.438；NVFP4全量化CLIP降至约21.25。局限：必须selective，收益被VAE/文本编码器稀释。
- **SVDQuant（2411.05007，MIT Han Lab+NVIDIA）**：smoothing迁移离群值→SVD分16bit低秩支吃离群+4bit残差支，Nunchaku融合两支kernel（不融合则rank32增57%开销）。FLUX12B：DiT显存22.7→6.5GiB(**3.5x**)，笔记本4090端到端111.7→**12.9s(8.7x)**，LPIPS**0.223**优于NF4(0.272)/naive INT4(0.322)；5090 NVFP4 3.1x；免offload总10.1x。局限：图像为主、低秩有0.3GiB开销。
- **其他**：Hopper FP8 E4M3/E5M2 TensorRT（博客级）；PTQ4DiT/Q-DiT/ViDiT-Q（细数字**查不到**）；DiRotQ(2605.16732) PCA旋转W4A4，PixArt-ΣFID15.9/PSNR19.1dB称超SVDQuant，FLUX 4090 2.3x/显存2.1x（单一新来源）；视频FP8+稀疏co-design(2506.04648)数字**查不到**。

## 10. RSI与agent化kernel搜索
- **AlphaEvolve（2506.13131，Google DeepMind）**：Gemini Flash广探+Pro精修，自动evaluator+进化库。成果：4×4复矩阵**48次标量乘法**（56年来首次）；Borg回收**0.7%全球算力**；Pallas matmul **≈23%**（降Gemini训练约1%）；FlashAttention **≈32/32.5%**；TPU Verilog位宽削减过形式验证。75%复现SOTA/20%更优。局限：需可自动验证评价器；曾出现奖励黑客（超长输入搞崩评分器）。
- **AccelOpt（2511.15915）**：beam search+optimization memory，planner(12计划)/executor(2尝试,最多144核)/summarizer(slow-fast对提策略)；正1.04x/负1.15x阈值，正负改写皆存。NKIBench(14个Trainium核)：Trainium1峰值**49→61%**、Trainium2 **45→59%**，开源模型匹敌Sonnet4且**便宜26x**。局限：仅Trainium、峰值仍约60%、预算大。
- 其他：CUDA-LLM/CudaForge（编译+正确+profile闭环）、KernelBench（无反馈时模型写核差）、RSI research agents(2609.26457)。

## 11. 分类体系与组合方式
### 11.1 分类（按是否改生成契约）
| 层 | 族 | 改契约 | 代表 | 增益 |
|---|---|---|---|---|
| 算法 | 轨迹步数蒸馏 | 是 | PD/CM/sCM/LCM/PCM/Hyper-SD/PDD | 4–8步,5–10x |
| 算法 | 分布匹配 | 是 | DMD/DMD2/CausVid | 1–4步,可超教师 |
| 算法 | 对抗蒸馏 | 是 | ADD/LADD/DMD2-GAN | 单步保真 |
| 算法 | CFG蒸馏 | 是 | GD/AGD/内置 | NFE≈2x |
| 算法 | 跨步缓存 | 近似不改调度 | TeaCache/PAB/TaylorSeer/EasyCache/FBC | 1.5–4.4x |
| 模型 | 稀疏注意力 | 近似/可训练 | STA/VSA/SparseVideoGen/SpargeAttn/Sage/Sol-Attn/PISA | attn2.8–17x |
| 模型 | token剪枝 | 近似 | ToMeSD/Astraea/TAPE/CoReDiT | 视序列 |
| 模型 | 原生高效 | 训练期 | 线性注意力/金字塔 | 设计级 |
| 系统 | 量化 | 近似 | FP8/MXFP8/NVFP4/SVDQuant/AWQ | 1.26–3.6x |
| 系统 | fusion/布局/通信 | 否(可分档) | CUTLASS/CODA/compile/RSI | launch/带宽 |

### 11.2 组合规律
1. **等契约与改契约分开记账**（Sol-H3）：fusion like-for-like；步数/分辨率改产物。
2. **增益不可直接相乘**：integrator处理cache×sparse×quant误差累积与瓶颈迁移。
3. **典型栈顺序**：步数/CFG蒸馏 → 稀疏attn或cache → token剪枝(高分辨率晚阶段) → 选择性量化(首尾敏感层高精) → RSI fusion收尾。
4. **量化须selective且与融合协同**（SVDQuant不融合被57%搬运吞掉；NVFP4全量LPIPS差）。
5. **agent/RSI是元层**：有效性上限=评价函数质量（正确性+profile+人/VLM闸）；AlphaEvolve奖励黑客、Sol-Engine保留人工为证。
6. **数据光谱**：data-free最省但受教师上限；轨迹蒸馏需教师推理；分布匹配+GAN最强但最复杂。

## 附：主要"查不到"
- Sol-H3完整VBench总分；AGD/PDD/FastHunyuan统一对照数；FastH3/HyperFlow/Larry/LightX2V独立基准；PTQ4DiT/Q-DiT/ViDiT-Q及视频FP8 co-design端到端数字；PDD(2607.26004)完整作者单位。