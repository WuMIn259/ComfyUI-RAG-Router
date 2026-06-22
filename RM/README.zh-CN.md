# 🧠 ComfyUI RAG Router (Z-Image RAG Router)

![ComfyUI](https://img.shields.io/badge/ComfyUI-Extension-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10+-blue)

<p align="right">
  <a href="./README.md">English</a> | <strong>简体中文</strong>
</p>

**ComfyUI RAG Router** 是一款为 ComfyUI 设计的工业级“检索增强生成 (RAG)”图像路由插件。
![从检索→Flux Klein4B图像尺寸预处理→Zimage Turbo丰富细节和画面](./example_workflows/P1.png)

它不仅是一个图库检索工具，更是下一代 **"粗排到精抛 (Coarse-to-Fine)"** 与 **"特征注入 (Vision Injection)"** 混合工作流的**智能中枢大脑**。
它能根据用户的自然语言 Prompt，从本地海量素材库中匹配最合适的参考图与提示词，并自动接管下游采样器的 Denoise（降噪）参数，实现 Img2Img 与 Txt2Img 的无缝全自动切换。
![生成实机展示](./example_workflows/P2.gif)

---

## ✨ 核心特性 (Key Features)

* **🧠 完全离线的语义检索 (Offline Multilingual Semantic Search)**
    * 内置 `paraphrase-multilingual-MiniLM-L12-v2` 向量检索，支持中英等多语言混合检索。
    * **懂语义，不仅是找标签**：“湛蓝的海洋”和“深邃的大海”也能被精准匹配，彻底告别死板的关键词正则匹配。
* **⚡ 闪电般的矩阵缓存 (Lightning Matrix Cache)**
    * 拒绝每次生图时的低效 For 循环。首次运行自动构建 `rag_index.pt` 张量矩阵缓存。
    * 哪怕本地图库有成千上万张素材，检索效率依然迅速！

* 🔀 **智能引擎切换 (Smart Denoise Routing)**
    * **命中图库时**：输出参考图、扩充的完整图像提示词，以及你设定的 `base_denoise`，完美引导图生图或风格迁移。
    * **未命中时（低于阈值）**：输出纯黑兜底图，并**自动强制输出 Denoise 1.0**。下游采样器瞬间化身纯文生图 (Txt2Img) 引擎，无需任何复杂的 Switch 旁路节点！

* **🌫️ 原生低频信息引导 (Low-Frequency Guiding)**
    * 内置高斯模糊 (`blur_radius`) 控制。一键抹杀原图高频细节，只提取“构图与色彩光影”作为底稿，规避抄袭风险。
* **🎲 “同分盲盒”抽卡机制 (Tied-Score Gacha)**
    * 面对图库中多个语义完全相同的素材，系统会基于 `seed` 在高分池（0.01 容差内）进行随机抽取，让每次点击生成都保持随机化！

---

## 🛠️ 安装指南 (Installation)

1. **克隆仓库：**
   将本仓库克隆到你的 ComfyUI `custom_nodes` 目录下：
   ```bash
   cd ComfyUI/custom_nodes
   git clone [https://github.com/你的用户名/ComfyUI-RAG-Router.git](https://github.com/你的用户名/ComfyUI-RAG-Router.git)

> [!IMPORTANT]
> **需要手动离线模型设置：**
> 为了确保最大程度的隐私保护并实现即时初始化，该扩展不会自动从 Hugging Face 在线下载嵌入模型（embedding model）。在运行工作流之前，您**必须**在本地配置好该模型。
> 
> 1. 从官方存储库下载所有文件: [Hugging Face - paraphrase-multilingual-MiniLM-L12-v2](https://hugging face.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/main)
> 2. 创建文件夹名: `models/paraphrase-multilingual-MiniLM-L12-v2` 在此扩展程序的目录内。
> 3. 将所有下载的配置文件和权重文件 (including the `1_Pooling` folder) 放入其中。
## 📂 资产库目录结构

您的本地图库目录中应包含成对的图像文件与 `.txt` 提示词文件，且两者的文件名必须相同：
```文本
your_gallery_folder/
├── asset_001.png
├── asset_001.txt   <-- (包含高度详尽的描述性提示词-建议用自然语言描述)
├── asset_002.jpg
└── asset_002.txt