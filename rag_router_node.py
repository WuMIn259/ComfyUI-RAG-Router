# rag_router_node.py
import os
import torch
import numpy as np
import random
from PIL import Image, ImageFilter

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
    print("[Z-Image RAG] Vector Engine (Multilingual) Ready.")
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    import difflib
    print("[Z-Image RAG] WARNING: SentenceTransformers not found. Fallback to String Matcher.")

class ZImage_RAG_Router:
    def __init__(self):
        self.model = None
        if HAS_SENTENCE_TRANSFORMERS:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            local_model_path = os.path.join(current_dir, "models", "paraphrase-multilingual-MiniLM-L12-v2")

            if os.path.exists(local_model_path):
                print(f"[Z-Image RAG] 🟢 成功定位本地离线向量模型: {local_model_path}")
                self.model = SentenceTransformer(local_model_path, device='cpu')
            else:
                print("[Z-Image RAG] 🟡 未找到本地模型，尝试通过网络下载 (可能需要特定网络环境)...")
                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "gallery_path": ("STRING", {"default": "./gallery_assets"}), 
                "blur_radius": ("INT", {"default": 0, "min": 0, "max": 200, "step": 1}),
                
                # 【新增功能】一键去色开关！彻底消灭原图色彩污染
                "grayscale_mode": ("BOOLEAN", {"default": False}),
                
                "match_threshold": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "base_denoise": ("FLOAT", {"default": 0.67, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rebuild_index": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), 
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("matched_image", "matched_prompt", "denoise", "debug_info")
    FUNCTION = "route_prompt"
    CATEGORY = "Z-Image/Routing"

    def load_image(self, img_path, blur_radius, grayscale_mode):
        i = Image.open(img_path).convert("RGB")
        
        # 1. 模糊降频
        if blur_radius > 0:
            i = i.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
        # 2. 【核心】去色处理，并转换回 RGB 通道以防 ComfyUI 报错
        if grayscale_mode:
            i = i.convert('L').convert('RGB')
            
        img_tensor = torch.from_numpy(np.array(i).astype(np.float32) / 255.0).unsqueeze(0)
        return img_tensor

    def get_gallery_pairs(self, gallery_path):
        pairs = []
        if os.path.exists(gallery_path):
            for file in os.listdir(gallery_path):
                if file.endswith(".txt"):
                    txt_path = os.path.join(gallery_path, file)
                    img_path = os.path.splitext(txt_path)[0] + ".png"
                    if not os.path.exists(img_path):
                        img_path = os.path.splitext(txt_path)[0] + ".jpg"
                    if os.path.exists(img_path):
                        pairs.append((txt_path, img_path))
        return pairs

    def route_prompt(self, prompt, gallery_path, blur_radius, grayscale_mode, match_threshold, base_denoise, rebuild_index, seed):
        random.seed(seed) 
        
        pairs = self.get_gallery_pairs(gallery_path)
        fallback_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        
        if not pairs:
            return (fallback_image, prompt, 1.0, "Gallery is empty or path invalid.")

        best_score = 0.0
        best_img_path = None

        if HAS_SENTENCE_TRANSFORMERS:
            index_file = os.path.join(gallery_path, "rag_index.pt")
            embeddings = None
            img_paths = []

            if not rebuild_index and os.path.exists(index_file):
                try:
                    data = torch.load(index_file)
                    if len(data['img_paths']) == len(pairs):
                        embeddings = data['embeddings']
                        img_paths = data['img_paths']
                except:
                    pass

            if embeddings is None:
                print(f"[Z-Image RAG] Building Semantic Index for {len(pairs)} items...")
                texts = []
                for txt_p, img_p in pairs:
                    with open(txt_p, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                    img_paths.append(img_p)
                
                embeddings = self.model.encode(texts, convert_to_tensor=True)
                torch.save({'embeddings': embeddings, 'img_paths': img_paths}, index_file)
                print("[Z-Image RAG] Index built and saved successfully!")

            prompt_emb = self.model.encode(prompt, convert_to_tensor=True)
            scores = util.cos_sim(prompt_emb, embeddings)[0] 
            
            max_score_tensor = torch.max(scores)
            max_score = max_score_tensor.item()
            
            tolerance = 0.01 
            top_indices = torch.where(scores >= max_score - tolerance)[0].tolist()
            
            chosen_idx = random.choice(top_indices)
            best_score = scores[chosen_idx].item()
            best_img_path = img_paths[chosen_idx]
            
            debug_str_extra = f" (Selected 1 from {len(top_indices)} top matches)"

        else:
            top_candidates = []
            tolerance = 0.01
            for txt_path, img_path in pairs:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    tag_text = f.read()
                score = difflib.SequenceMatcher(None, prompt.strip().lower(), tag_text.strip().lower()).ratio()
                
                if score > best_score + tolerance:
                    best_score = score
                    top_candidates = [img_path]
                elif abs(score - best_score) <= tolerance:
                    top_candidates.append(img_path)
            
            best_img_path = random.choice(top_candidates)
            debug_str_extra = f" (Selected 1 from {len(top_candidates)} top matches)"

        matched_prompt = ""
        output_denoise = 1.0  
        
        if best_img_path is None or best_score < match_threshold:
            debug_str = f"Score: {best_score:.2f} | Failed Threshold ({match_threshold:.2f}). No Match."
            matched_image = fallback_image
            matched_prompt = prompt 
            output_denoise = 1.0
        else:
            debug_str = f"Score: {best_score:.2f}{debug_str_extra} | Match: {os.path.basename(best_img_path)}"
            # 把 grayscale_mode 传给 load_image
            matched_image = self.load_image(best_img_path, blur_radius, grayscale_mode)
            output_denoise = base_denoise
            
            best_txt_path = os.path.splitext(best_img_path)[0] + ".txt"
            if os.path.exists(best_txt_path):
                with open(best_txt_path, 'r', encoding='utf-8') as f:
                    matched_prompt = f.read()

        return (matched_image, matched_prompt, output_denoise, debug_str)

NODE_CLASS_MAPPINGS = {
    "ZImage_RAG_Router": ZImage_RAG_Router
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImage_RAG_Router": "Z-Image RAG Router 🧠"
}
