from PIL import Image
from pathlib import Path
import os
import json
import numpy as np
import folder_paths
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
import mlx.core as mx
import torch
import re

def tensor_to_pil(image_tensor, batch_index=0) -> Image:
    # Convert tensor of shape [batch, height, width, channels] at the batch_index to PIL Image
    image_tensor = image_tensor[batch_index].unsqueeze(0)
    i = 255.0 * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img

class ModelsInfo:
    def __init__(self):
        current_dir = Path(__file__).parent.resolve()
        models_info_file = os.path.join(current_dir, "models.json")
        with open(models_info_file, "r", encoding="utf-8") as f:
            self.models_info = json.load(f)

class Qwen25VL(ModelsInfo):
    def __init__(self):
        super().__init__()
        self.model_checkpoint = None
        self.model = None
        self.processor = None
        self.config = None
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"default": "", "multiline": True}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
                "seed": ("INT", {"default": -1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL"

    def inference(
        self,
        image,
        text: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        seed: int = -1,
    ):
        model_info = self.models_info["mps"][1]
        self.model_checkpoint = os.path.join(folder_paths.base_path, model_info['local_path'])
        # 如果模型不存在就下载
        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download
            # 使用 huggingface 下载
            file_path = snapshot_download(repo_id=model_info['repo_id'], local_dir=self.model_checkpoint,local_dir_use_symlinks=False)
            print(f"Model downloaded to: {file_path}")
        # 加载模型
        from mlx_vlm import load,generate
        import mlx.core as mx
        mx.random.seed(seed=seed)
        self.model, self.processor = load(self.model_checkpoint)
        self.config = load_config(self.model_checkpoint)
        formatted_prompt = apply_chat_template(
            self.processor, self.config, text, num_images=1
        )
        # 推理
        image = tensor_to_pil(image)
        print(f"image type: {type(image)}")
        output = generate(self.model, self.processor, formatted_prompt, [image], verbose=False, temperature=temperature, max_new_tokens=max_new_tokens, seed=seed)
        del self.model, self.processor, self.config  # 清理显存
        self.model = None
        self.processor = None
        self.config = None
        torch.mps.empty_cache()  # 清理 MPS 显存
        print(f"Output: {output}")
        return (output.text,)
    
class Qwen3(ModelsInfo):
    def __init__(self):
        super().__init__()
        self.model_checkpoint = None
        self.model = None
        self.tokenizer = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {"default": "你是一个智能助理","multiline": True}),
                "user_prompt": ("STRING", {"multiline": True}),
                "direct": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
                "seed": ("INT", {
                    "default": 0,  # 默认值
                    "min": 0,      # 最小值
                    "max": 0xffffffffffffffff,  # 最大值（64位整数）
                    "step": 1      # 步长
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenCUDA"

    def inference(
        self,
        system_prompt: str,
        user_prompt: str,
        direct: bool = False,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        seed: int = -1,
    ):
        if direct:
            return (user_prompt,)
        model_info = self.models_info["mps"][0]
        self.model_checkpoint = os.path.join(folder_paths.base_path, model_info['local_path'])
        # 如果模型不存在就下载
        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download
            # 使用 huggingface 下载
            file_path = snapshot_download(repo_id=model_info['repo_id'], local_dir=self.model_chnaeckpoint,local_dir_use_symlinks=False)
            print(f"Model downloaded to: {file_path}")
        # 加载模型
        from mlx_lm import load, generate
        import mlx.core as mx
        mx.random.seed(seed=seed)
        self.model, self.tokenizer = load(self.model_checkpoint)
        prompt = f"{system_prompt}\n\n{user_prompt} /no_think"  # 添加 /no_think 标签
        if self.tokenizer.chat_template is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt} /no_think"}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )

        # 5. 生成响应
        def remove_think_tags(text):
            """
            删除文本中所有的<think>标签及其内容，并移除标签后的换行符
            
            参数:
            text (str): 包含XML标签的文本
            
            返回:
            str: 移除<think>标签及后续换行符后的文本
            """
            # 正则表达式模式：匹配<think>标签及其内容，以及紧随其后的换行符
            pattern = r'<think>.*?</think>\s*'
            # 使用re.DOTALL标志使.可以匹配换行符
            # 使用非贪婪匹配(.*?)确保只匹配到最近的</think>
            return re.sub(pattern, '', text, flags=re.DOTALL)
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            verbose=True,
        )
        del self.model, self.tokenizer  # 清理显存
        self.model = None
        self.tokenizer = None
        torch.mps.empty_cache()  # 清理 MPS 显存
        print(f"Response: {response}")
        return (remove_think_tags(response),)