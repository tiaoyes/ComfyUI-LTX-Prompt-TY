import torch
import numpy as np
from PIL import Image
import io
import base64
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ComfyUI-LTX-Prompt-TY")

class LTXPromptTiaoYe:
    @classmethod
    def INPUT_TYPES(s):
        # 根据用户最新需求定制的模型列表
        OLLAMA_MODELS = [
            "qwen3.5:9b", 
            "qwen3.5:4b", 
            "qwen3.5:27b", 
            "qwen3.5-abliterated:4B",
            "qwen3.5-abliterated:9b",
            "qwen3.5-abliterated:27b",
            "qwen3-vl:4b",
            "qwen3-vl:8b"
        ]
        
        return {
            "required": {
                "图片_1": ("IMAGE",),
                "Ollama地址": ("STRING", {"default": "http://localhost:11434"}),
                "模型选择": (OLLAMA_MODELS, {"default": "qwen3.5:9b"}),
                "角色特征库": ("STRING", {
                    "multiline": True, 
                    "default": (
                        "跳爷: 银色背头, 深蓝唐装(银丝云纹), 右手墨翠扳指。\n"
                        "徒弟: 黑色短发, 灰色粗布汗衫, 左臂淡红伤疤。"
                    )
                }),
                "跳爷导演剧本": ("STRING", {
                    "multiline": True, 
                    "default": "跳爷 严厉地叮嘱: '练拳先练心。'"
                }),
                "对话语言": (["对话保留中文", "全篇翻译英文"], {"default": "对话保留中文"}),
            },
            "optional": {
                f"图片_{i}": ("IMAGE",) for i in range(2, 11)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("LTX最终提示词",)
    FUNCTION = "execute_ty_local_logic"
    CATEGORY = "🎬 LTX 提示词 跳爷"

    def tensor_to_base64(self, tensor):
        try:
            t = tensor.detach().cpu()
            if len(t.shape) == 4: t = t[0]
            nparr = (t.numpy() * 255).astype(np.uint8)
            img = Image.fromarray(nparr)
            img.thumbnail((768, 768), Image.Resampling.LANCZOS)
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e: return None

    def execute_ty_local_logic(self, 图片_1, Ollama地址, 模型选择, 角色特征库, 跳爷导演剧本, 对话语言, **kwargs):
        try:
            # 1. 序列帧处理
            all_tensors = [图片_1] if 图片_1.shape[0] == 1 else [图片_1[i].unsqueeze(0) for i in range(图片_1.shape[0])]
            for i in range(2, 11):
                img = kwargs.get(f"图片_{i}")
                if img is not None: all_tensors.append(img)
            b64_list = [self.tensor_to_base64(t) for t in all_tensors if self.tensor_to_base64(t) is not None][:10]

            # 2. 核心 Prompt 逻辑：对话对齐 + LTX 2.3 物理特性
            lang_mode = "Keep dialogue in Chinese" if 对话语言 == "对话保留中文" else "Translate all to English"
            
            system_prompt = f"""
            Task: Local Master Cinematic Director for LTX-Video 2.3.
            [STRICT IDENTITY & DIALOGUE LOCK]
            Characters: {角色特征库}
            Script: {跳爷导演剧本}
            
            [DIRECTIVES]
            1. SPEAKER ALIGNMENT: Dialogue belongs ONLY to the character named before the colon in the script. 
            2. VISUAL SYNC: Describe mouth movement of the CORRECT speaker defined in the fingerprints. 
            3. LTX 2.3 PHYSICS: Detail sweat/fluid absorption, cloth tension, and volumetric lighting.
            4. MOTION: Ensure fluid motion transition across these {len(b64_list)} frames.

            FORMAT: One immersive natural language paragraph in English (except dialogue). 
            START WITH: 'Photorealistic cinematic video,'
            Sync: {lang_mode}. Quotes "" mandatory for dialogue.
            """

            # 3. Ollama 本地请求
            endpoint = f"{Ollama地址.rstrip('/')}/api/chat"
            payload = {
                "model": 模型选择, 
                "messages": [{"role": "user", "content": system_prompt, "images": b64_list}], 
                "stream": False
            }
            res = requests.post(endpoint, json=payload, timeout=300).json()
            return (res['message']['content'],)

        except Exception as e:
            return (f"本地反推执行失败: {str(e)}",)

NODE_CLASS_MAPPINGS = {"LTXPromptTiaoYe": LTXPromptTiaoYe}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXPromptTiaoYe": "🎬 LTX 提示词 跳爷"}