# llm_purifier_node.py
import os
import torch
import gc
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

# 【核心升级】：建立一个全局缓存池，模型加载一次后永久保留在内存/显存中
_ZIMAGE_LLM_CACHE = {
    "model_name": None,
    "device": None,
    "tokenizer": None,
    "model": None
}

class ZImage_Prompt_Purifier:
    @classmethod
    def INPUT_TYPES(cls):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        llm_dir = os.path.join(current_dir, "llm_models")
        
        if not os.path.exists(llm_dir):
            os.makedirs(llm_dir)

        model_folders = [f for f in os.listdir(llm_dir) if os.path.isdir(os.path.join(llm_dir, f))]
        if not model_folders:
            model_folders = ["Please put models in llm_models folder"]

        return {
            "required": {
                "llm_model": (model_folders,),
                "system_prompt": ("STRING", {
                    "multiline": True, 
                    # 【优化系统提示词】明确禁止前缀
                    "default": "你是一个专业的AI图像提示词重构专家。结合【原图参考】与【用户新需求】，输出一段完美的英文提示词。要求：\n1. 颜色、材质必须完全听从用户的【新需求】。\n2. 构图、光影保留【原图参考】。\n3. 【绝对警告】只输出纯英文提示词内容本身！绝对不准使用\"prompt:\", \"image:\", 引号等任何前缀或修饰符！"
                }),
                "rag_reference_prompt": ("STRING", {"multiline": True}),
                "user_target_prompt": ("STRING", {"multiline": True}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "max_new_tokens": ("INT", {"default": 256, "min": 64, "max": 1024}),
                # 【新增开关】默认打勾。打勾时，秒级响应；取消打勾时，阅后即焚。
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("purified_prompt", "debug_info")
    FUNCTION = "purify"
    CATEGORY = "Z-Image/LLM"

    def purify(self, llm_model, system_prompt, rag_reference_prompt, user_target_prompt, device, max_new_tokens, keep_model_loaded):
        global _ZIMAGE_LLM_CACHE

        if llm_model == "Please put models in llm_models folder":
            return ("", "Error: No LLM models found.")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "llm_models", llm_model)

        print(f"\n[Z-Image LLM] 🤖 启动提示词净化中枢...")

        try:
            # 【缓存机制拦截】：如果模型没变且设备没变，直接秒用缓存！
            if (_ZIMAGE_LLM_CACHE["model"] is not None and 
                _ZIMAGE_LLM_CACHE["model_name"] == llm_model and 
                _ZIMAGE_LLM_CACHE["device"] == device):
                print(f"[Z-Image LLM] ⚡ 命中全局缓存！跳过加载过程。")
                tokenizer = _ZIMAGE_LLM_CACHE["tokenizer"]
                model = _ZIMAGE_LLM_CACHE["model"]
            else:
                # 如果缓存里的模型不对，先清理掉旧的
                if _ZIMAGE_LLM_CACHE["model"] is not None:
                    print(f"[Z-Image LLM] 🔄 正在释放旧模型缓存...")
                    del _ZIMAGE_LLM_CACHE["model"]
                    del _ZIMAGE_LLM_CACHE["tokenizer"]
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                print(f"[Z-Image LLM] 📂 全新加载模型: {llm_model} (挂载至: {device.upper()})")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                
                # 【终极原生性能版配置】
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device if device == "cpu" else None,
                    trust_remote_code=True,
                    torch_dtype=torch.float16, # 👈 注意：必须是全小写的 float16
                    attn_implementation="sdpa", # 👈 【隐藏涡轮】开启原生融合注意力加速！
                    low_cpu_mem_usage=True
                )
                
                if device == "cuda":
                    model = model.to("cuda")

                # 存入全局缓存
                _ZIMAGE_LLM_CACHE["model_name"] = llm_model
                _ZIMAGE_LLM_CACHE["device"] = device
                _ZIMAGE_LLM_CACHE["tokenizer"] = tokenizer
                _ZIMAGE_LLM_CACHE["model"] = model

            user_content = f"【原图参考】\n{rag_reference_prompt}\n\n【用户新需求】\n{user_target_prompt}"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

            # 【核心修改区：加入官方物理开关并做兼容容错】
            print("[Z-Image LLM] ⚙️ 正在应用 Chat Template，并尝试关闭底层思维链 (Thinking Mode)...")
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True, 
                    enable_thinking=False # 👈 官方文档给出的终极硬开关！
                )
            except Exception:
                # 兜底：如果用户换回了老模型（不支持这个参数），就走正常流程
                prompt_text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )

            inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

            print("[Z-Image LLM] 🧠 正在执行极速解码...")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False, # 👈 因为已经在源头关闭了 Thinking，这里可以极其安全地使用贪心解码！
                    pad_token_id=tokenizer.eos_token_id
                )

            generated_ids = outputs[0][len(inputs.input_ids[0]):]
            raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # 【暴力清洗】：剪掉思维链，抹杀大模型喜欢乱加的前缀
            purified_result = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
            if "</think>" in purified_result:
                purified_result = purified_result.split("</think>")[-1].strip()
                
            # 物理级清洗前缀
            prefixes_to_strip = ["prompt:", "image:", "color:", "output:", "result:", '"']
            for prefix in prefixes_to_strip:
                if purified_result.lower().startswith(prefix):
                    purified_result = purified_result[len(prefix):].strip()
            if purified_result.endswith('"'):
                 purified_result = purified_result[:-1].strip()

            debug_str = f"Success. Device: {device.upper()}."
            print(f"[Z-Image LLM] ✨ 净化完成:\n{purified_result}\n")

        except Exception as e:
            purified_result = user_target_prompt
            debug_str = f"LLM Error: {str(e)}"
            print(f"[Z-Image LLM] ❌ 错误: {debug_str}")

        finally:
            if 'inputs' in locals():
                del inputs
            if 'outputs' in locals():
                del outputs
            
            # 【控制流】：根据用户开关决定是否清理大模型
            if not keep_model_loaded:
                print("[Z-Image LLM] 🧹 阅后即焚模式触发，正在卸载模型...")
                _ZIMAGE_LLM_CACHE["model"] = None
                _ZIMAGE_LLM_CACHE["tokenizer"] = None
                if 'model' in locals(): del model
                if 'tokenizer' in locals(): del tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        return (purified_result, debug_str)

NODE_CLASS_MAPPINGS = {"ZImage_Prompt_Purifier": ZImage_Prompt_Purifier}
NODE_DISPLAY_NAME_MAPPINGS = {"ZImage_Prompt_Purifier": "Z-Image LLM Prompt Purifier 🤖"}
