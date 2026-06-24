# __init__.py
import sys
import os

# 确保当前目录被加入到系统路径，方便内部文件互相导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从你的业务代码文件中导入节点类
from .rag_router_node import ZImage_RAG_Router
from .llm_purifier_node import ZImage_Prompt_Purifier

# 注册节点类映射
NODE_CLASS_MAPPINGS = {
    "ZImage_RAG_Router": ZImage_RAG_Router,
    "ZImage_Prompt_Purifier": ZImage_Prompt_Purifier
}

# 注册节点在 ComfyUI UI 面板上的显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImage_RAG_Router": "Z-Image RAG Router 🧠",
    "ZImage_Prompt_Purifier": "Z-Image LLM Prompt Purifier 🤖"
}

# 必须包含这个 __all__ 列表，ComfyUI 靠它来加载节点
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("------------------------------------------")
print(" Loaded: Z-Image RAG Router (Architecture Edition)")
print("------------------------------------------")
