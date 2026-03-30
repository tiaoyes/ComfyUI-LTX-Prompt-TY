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
        OLLAMA_MODELS = ["qwen3-vl:4b", "qwen3.5:4b", "qwen3.5:9b", "qwen3.5:27b", "qwen3.5-abliterated:4b", "qwen3.5-abliterated:9b", "qwen3.5-abliterated:27b", "qwen3-vl:8b"]
        return {
            "required": {
                "图片_1": ("IMAGE",),
                "Ollama地址": ("STRING", {"default": "http://localhost:11434"}),
                "模型选择": (OLLAMA_MODELS, {"default": "qwen3-vl:4b"}),
                "显存策略": (["立即卸载", "5分钟驻留", "始终驻留"], {"default": "立即卸载"}),
                "角色与剧本分镜": ("STRING", {"multiline": True, "default": "【角色】跳爷: 银发, 深蓝唐装. 徒弟: 黑色汗衫.\n【剧本】跳爷对徒弟说：'这就是真功夫。'"}),
                "对话语言": (["对话保留中文", "全篇翻译英文"], {"default": "对话保留中文"}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": { f"图片_{i}": ("IMAGE",) for i in range(2, 11) }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("LTX最终提示词",)
    FUNCTION = "execute_ty_prompt_logic"
    CATEGORY = "🎬 LTX 提示词 跳爷"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def tensor_to_base64(self, tensor):
        try:
            t = tensor.detach().cpu()
            if len(t.shape) == 4: t = t[0]
            nparr = (t.numpy() * 255).astype(np.uint8)
            img = Image.fromarray(nparr)
            img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=80)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e: return None

    def execute_ty_prompt_logic(self, 图片_1, Ollama地址, 模型选择, 显存策略, 角色与剧本分镜, 对话语言, 随机种子, **kwargs):
        keep_alive_val = {"立即卸载": 0, "5分钟驻留": "5m", "始终驻留": -1}.get(显存策略, 0)
        
        try:
            all_tensors = [图片_1] if 图片_1.shape[0] == 1 else [图片_1[i].unsqueeze(0) for i in range(图片_1.shape[0])]
            for i in range(2, 11):
                img = kwargs.get(f"图片_{i}")
                if img is not None:
                    if img.shape[0] == 1:
                        all_tensors.append(img)
                    else:
                        all_tensors.extend([img[j].unsqueeze(0) for j in range(img.shape[0])])
            b64_list = [self.tensor_to_base64(t) for t in all_tensors if self.tensor_to_base64(t) is not None]

            # 语言逻辑硬编码
            if 对话语言 == "对话保留中文":
                lang_instruction = "CRITICAL: Character dialogue (text inside double quotes) MUST be in CHINESE. All other descriptive text must be in professional English."
            else:
                lang_instruction = "CRITICAL: Translate the entire response, including character dialogue, into professional English."

            system_instruction = f"""
            You are a professional LTX-Video 2.3 prompt architect.
            [STRICT RULES]
            1. Output ONLY the prompt paragraph.
            2. NO preamble, NO explanations, NO intro/outro.
            3. START with 'Photorealistic cinematic video,'.
            
            [STRUCTURE]
            - Start with main action from Frame 1.
            - Detail kinetics, cloth physics, and character appearance from the provided character info.
            - Describe environment and volumetric lighting.
            - Use professional camera terms (Dolly, Track, Rack Focus).
            - {lang_instruction}
            - Context: {角色与剧本分镜}
            """

            endpoint = f"{Ollama地址.rstrip('/')}/api/chat"
            payload = {
                "model": 模型选择, 
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"Create a prompt for {len(b64_list)} frames. Context: {角色与剧本分镜}", "images": b64_list}
                ], 
                "stream": False,
                "keep_alive": keep_alive_val,
                "options": {"seed": 随机种子, "temperature": 0.3}
            }
            
            res = requests.post(endpoint, json=payload, timeout=300).json()
            output = res['message']['content'].strip()
            
            # 清洗开头和结尾的废话
            if "Photorealistic" in output:
                output = output[output.find("Photorealistic"):]
            
            # 找到最后一个标点符号截断
            last_punc = max(output.rfind('.'), output.rfind('"'), output.rfind('”'), output.rfind('!'), output.rfind('?'))
            if last_punc != -1:
                output = output[:last_punc+1]

            return (output,)

        except Exception as e: return (f"导演逻辑错误: {str(e)}",)

NODE_CLASS_MAPPINGS = {"LTXPromptTiaoYe": LTXPromptTiaoYe}
NODE_DISPLAY_NAME_MAPPINGS = {"LTXPromptTiaoYe": "🎬 LTX 提示词 跳爷"}
