# 🧠 ComfyUI RAG Router

![ComfyUI](https://img.shields.io/badge/ComfyUI-Extension-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10+-blue)

<p align="right">
  <strong>English</strong> | <a href="./README.zh-CN.md">简体中文</a>
</p>

**ComfyUI RAG Router**  is an industrial-grade Retrieval-Augmented Generation (RAG) image routing extension built exclusively for ComfyUI.
![Retrieval → Flux Klein4B image dimension preprocessing → Zimage Turbo detail & scene enhancement](./example_workflows/P1.png)

Far more than a simple gallery lookup tool, it acts as the intelligent core hub for next-generation hybrid workflows combining Coarse-to-Fine ranking and Vision Feature Injection.
Given user natural language prompts, it matches the most suitable reference images and complementary prompts from your local massive asset library. 
It also automatically manages the downstream sampler's denoising parameters, enabling seamless, fully automatic switching between Img2Img and Txt2Img generation modes.
![Live generation demo](./example_workflows/P2.gif)

---

## ✨ Key Features

* **🧠 Fully Offline Multilingual Semantic Search**
  * Embeds the `paraphrase-multilingual-MiniLM-L12-v2` embedding model for vector search, supporting mixed queries in Chinese, English and other languages.
  * Semantic-aware matching instead of rigid tag filtering: phrases like "crystal blue ocean" and "deep dark sea" return accurate matches, eliminating inflexible keyword regex matching entirely.
* **⚡ Lightning-Fast Matrix Cache System**
  * Avoids inefficient iterative loops on every generation run. Automatically builds a tensor matrix cache file `rag_index.pt` on the first execution.
  * Delivers near-instant retrieval performance under 0.05s even with local libraries containing tens of thousands of asset images.

* 🔀 ** Intelligent Generation Engine Routing via Denoise Control
    * **When matching assets are found: Outputs matched reference images, expanded full descriptive prompts, 
and your predefined base_denoise value, perfectly driving image-to-image generation or style transfer pipelines.
    * **When no matches meet the similarity threshold: Returns a solid black fallback image and forces denoise strength to 1.0 automatically. 
Downstream samplers instantly switch to pure text-to-image mode without complicated switch bypass node setups.
* **🌫️ Native Low-Frequency Composition Guidance
    * Built-in Gaussian blur adjustable via the blur_radius parameter. 
Instantly strip high-frequency fine details from reference images, retaining only composition, color palette and lighting as base drafts to mitigate copyright replication risks.
* **🎲 Tied-Score Random Draw Gacha Mechanism
    * For multiple assets with nearly identical semantic similarity scores, the node performs seeded random sampling from the high-score pool (within a 0.01 similarity tolerance range),
 ensuring unique, varied outputs every generation run.
---

## 🛠️ Installation Guide

Clone the repository
Navigate to your ComfyUI custom_nodes folder and clone this repo:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/WuMIn259/ComfyUI-RAG-Router.git

> [!IMPORTANT]
> **Manual Offline Model Setup Required:**
> To ensure maximum privacy and instant initialization, this extension does not auto-download the embedding model from Hugging Face online. You **must** set up the model locally before running your workflows.
> 
> 1. Download all files from the official repository: [Hugging Face - paraphrase-multilingual-MiniLM-L12-v2](https://hugging face.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/tree/main)
> 2. Create a folder named `models/paraphrase-multilingual-MiniLM-L12-v2` inside this extension's directory.
> 3. Place all the downloaded config and weight files (including the `1_Pooling` folder) into it.

## 📂 Asset Library Directory Structure

Your local gallery directory should contain matching pairs of images and `.txt` prompt files with identical filenames:
```text
your_gallery_folder/
├── asset_001.png
├── asset_001.txt   <-- (Contains highly-detailed descriptive prompts- It is recommended to use natural language for the description.)
├── asset_002.jpg
└── asset_002.txt
