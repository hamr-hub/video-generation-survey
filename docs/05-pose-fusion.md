# 05 · 姿态骨架融合

骨架（pose/skeleton）是舞蹈生成里比音频更标准的中间表示，动作与外观解耦。三个融合层级。

## Level 1：Pose 当条件喂进 UNet（最轻、最常用）

- **表示**：DWPose/OpenPose 提取 2D 关键点 → 渲染成骨架热图（stick figure）或坐标 token。DWPose 覆盖身体+手+脸，优于 OpenPose。
- **融合方式**：
  - **ControlNet**（主流，SD1.5 级，~12GB）：冻结视频 UNet + pose ControlNet 分支，特征逐层注入。
  - 通道拼接：骨架图与噪声 latent concat。
  - pose encoder → cross-attention 软条件。
- **音频+骨架双条件**：两路各一 encoder，UNet 内相加/交叉；骨架管动作踩点、音频管节奏风格。骨架已隐含节拍，两路天然对齐。
- **代表**：MagicAnimate、Animate Anyone、Moore-AnimateAnyone、Champ（spline pose）、MimicMotion。

## Level 2：骨架当中间潜空间（最利于音画同步 / 边缘友好）

两段式，不直接生成像素：

```
音乐 → 节拍/动作生成 → 骨架序列 → pose2video → 画面
       Bailando/FLOAT/EDGE    ControlNet/AnimateAnyone
```

- 第一段：音乐特征（beat/chroma）→ 动作骨架，模型极小（甚至 LSTM/Transformer），**踩点在此精确保证**。
- 第二段：骨架驱动像素，外观/动作解耦。
- **代表**：Bailando（VQ 动作码）、EDGE（SMPL 扩散）、FLOAT（flow matching 运动 latent）。
- **优点**：同步在小模型段锁死，大模型只管好看，重活轻活拆开，对 7GB 设备最友好。

## Level 3：三位一体联合扩散（最重，研究向）

- 视频 UNet + 音频 UNet + **姿态支路**三路联合去噪，两两 cross-attention，共享时间网格。
- 骨架既是条件又是被生成对象：给定音乐，骨架与画面一起涌现。
- 姿态支路本身极轻（几十关键点，几乎不占显存），增量主要在 attention。
- 可由 MM-Diffusion 双 UNet 直接加一路姿态，或用 SMPL 参数当第三模态。

## 工程要点

1. **时间网格对齐**：音频帧、骨架帧、视频帧映射到同一物理时间轴，按帧位置而非张量下标对齐（与 Sol-H3 跨模态衔接同理）。
2. 姿态来源统一用 **DWPose**，MimicPose/MimicMotion 已验证。
3. Orin 7GB 建议：**Level 2 两段式**，第一段极轻先验证踩点，第二段可 CPU offload。

## Thin-Slice 落地建议

1. 先只跑第一段"音乐→骨架"（Bailando/FLOAT，几百 MB），立刻验证踩点准不准。
2. 再接 pose2video（Moore-AnimateAnyone/MimicMotion + ControlNet）。
3. 用最小代价验证整条链通不通，再谈画质。
