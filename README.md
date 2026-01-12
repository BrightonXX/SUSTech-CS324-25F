<div align="center">

# SUSTech CS324: Deep Learning

![Semester](https://img.shields.io/badge/Semester-2025_Fall-blue?style=flat-square)
![Language](https://img.shields.io/badge/Language-Python_(PyTorch)-orange?style=flat-square&logo=python)
![Score](https://img.shields.io/badge/Assignment_Score-100%2F100-success?style=flat-square)

📖 Language: **中文** | [English](./README_en.md)

<p align="center">
  <strong>南方科技大学 CS324 深度学习作业仓库</strong><br>
  包含三次 Assignment 的完整源码与报告。
</p>

</div>

## 📚简介 

**Semester:** 2025 Fall
**Lecturer:** Prof. Jianguo Zhang (张建国)

本仓库归档了我在这门课程中的三次作业（Assignment 1-3）的源代码和详细报告。

> [!NOTE]
> **关于评分标准的个人理解：**
> 在学期初，老师曾提到作业评分看重 **“探索精神”**。
> 我将其理解为：**不满足于仅仅完成 Task 要求，而是根据实验结果做进一步的消融实验（Ablation Study）、对比分析或额外功能的实现。**
> 践行这一思路后，我的三次作业均获得了满分，希望这个思路能给你提供参考。

## 📂 作业列表 (Assignments)

|  #  |           主题 (Topic)           |     核心内容 (Key Contents)      | 分数 |                                               亮点 / 额外探索                                                |
|:---:|:--------------------------------|----------------------------------|:---:|:----------------------------------------------------------------------------------------------------------:|
|  1  | **Perceptron & MLP (NumPy)**    | 手写反向传播、SGD/BGD 收敛分析      | 100 | 额外探究了 **Momentum** 对收敛速度的影响，以及 **Leaky ReLU** 解决 Dead Neurons 的效果。                         |
|  2  | **CNN & RNN (PyTorch)**         | VGG on CIFAR-10, RNN 梯度消失验证 | 100 | 详细推导了 Vanishing Gradient 的数学原理；对比了 Data Augmentation 与 Learning Rate Scheduler 的组合效果。 |
|  3  | **LSTM & GAN**                  | LSTM 解决长序列依赖, GAN 生成手写数字 | 100 | 实现了 GAN 的 **Latent Space 插值** 与 **向量算术** (e.g., $7 - 1 + 6 \approx 9$)。                          |

[![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FBrightonXX%2FSUSTech-CS324-25F&label=Visitors&countColor=%23263759)](https://visitorbadge.io/status?path=https%3A%2F%2Fgithub.com%2FBrightonXX%2FSUSTech-CS324-25F)

## 🧪 实验报告预览 (Report Highlights)

为了体现“探索精神”，我在报告中包含了一些有趣的实验现象分析。

### Assignment 3: GAN Latent Space Arithmetic
验证 GAN 是否学习到了数字的语义特征，而非单纯记忆像素。
*   **插值 (Interpolation):** 观察数字 `7` 渐变为 `6` 的过程，推测 `1` 位于两者之间。
*   **算术 (Arithmetic):** 提取“顶部横线”特征向量 ($z_{roof} = z_7 - z_1$)，并将其注入到 `6` 中，得到了类似 `9` 的结果。

<div align="center">
  <img src="./assets/gan_interpolation.png" width="80%" alt="GAN Interpolation">
  <p><em>From Report 3: Latent Space Interpolation & Vector Arithmetic</em></p>
</div>

### Assignment 1: Convergence Analysis
在手写 MLP 阶段，对比了不同 Batch Size 对 Loss 收敛曲线的影响，并额外实现了 Momentum 优化器与 Leaky ReLU，证明了它们在特定初始化下的鲁棒性。

## 🧠 建议 (Suggestions)

1.  **关于 Report**：
    *   助教和老师更喜欢看到**图表**和**数据分析**，而不是大段的代码粘贴。
    *   如果 Task 让你对比 A 和 B，不妨顺手把 C 也跑一下（比如 Ass 1 中 Task 没要求 Leaky ReLU，但我顺手做了对比实验），这通常是加分项。

2.  **关于课程**：
    *   目前的计系培养方案中，CS324 的地位略显尴尬，但内容非常硬核且前沿。
    *   建议在做作业时多思考 "Why"，比如“为什么 Loss 在第8轮开始上升？”（过拟合），并尝试用代码去验证你的猜想。

## 🛠️ 环境 (Environment)
*   Python 3.8+
*   PyTorch (CUDA Recommended)
*   NumPy, Matplotlib

---

> [!CAUTION]
> **Honor Code**
>
> 南方科技大学计算机系严查代码抄袭。
> 本仓库代码仅供**思路参考 (Reference Only)**，请勿直接复制用于作业提交。
> 请保持学术诚信，享受 Deep Learning 的乐趣。

<div align="center">
  <p>如果这个仓库对你有启发，给个 ⭐️ <strong>Star</strong> 呗！</p>
</div>
