import json
from typing import List
import requests


class TextCleaningEngine:
    
    def __init__(self, base_url: str = 'http://localhost:11434', model: str = 'qwen2.5:7b'):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self._setup_system_prompts()
    
    def _setup_system_prompts(self) -> None:
        self.cleaning_prompts = {
            "sentence_repair": """你是一个专业的文本修复专家。请对输入文本执行以下操作：

            1. **删除不连贯的短句片段** - 移除无意义的断句、不完整的表达
            2. **修复语法错误** - 修正主谓不一致、时态错误等
            3. **优化句子流畅度** - 调整语序，确保逻辑连贯
            4. **保持语义完整性** - 不改变原文的核心含义
            5. **处理特殊符号** - 合理保留必要的标点，移除干扰符号

            输出要求：
            - 直接返回清洗后的一段完整正文文本
            - 不要添加任何解释或标记
            - 保持段落结构和重要格式
            - 确保文本自然通顺""",

                        "paragraph_reconstruction": """你是一个文本重构专家。请对以下文本进行深度修复：

            处理重点：
            🔹 识别并删除不连续的短句碎片小标题
            🔹 合并语义相关的短句
            🔹 重建逻辑连接关系
            🔹 优化文本流畅性和可读性
            🔹 保持专业术语和关键信息
            - 删除与作者信息相关的内容、以及关键词
            - 删除类似'中国发作性睡病诊断与治疗指南(2022版),中华医学会神经病学分会睡眠障碍学组'等背景信息

            重构原则：
            - 语义连贯性优先
            - 保持原文风格不变
            - 修复并不重写内容
            - 确保技术准确性
            - 不要添加任何解释或标记"""
        }
    
    def _call_model_api(self, prompt: str, text_chunk: str, system_prompt_key: str) -> str:
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": self.cleaning_prompts[system_prompt_key],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.8,
                "num_predict": min(len(text_chunk) * 2, 4000),
                "repeat_penalty": 1.2
            }
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
            
        except requests.exceptions.RequestException as e:
            print(f"API调用失败: {e}")
            return text_chunk
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            return text_chunk
        except Exception as e:
            print(f"未知错误: {e}")
            return text_chunk
    
    def _create_cleaning_prompt(self, text_chunk: str) -> str:
        return f"需要处理的文本：\n{text_chunk}\n\n请返回清洗后的文本："
    
    def _create_reconstruction_prompt(self, text_chunk: str) -> str:
        return f"需要处理的文本：\n{text_chunk}\n\n请返回重构后的文本："
    
    def clean_text_chunk(self, text_chunk: str) -> str:
        if not text_chunk or not text_chunk.strip():
            return text_chunk
            
        cleaning_prompt = self._create_cleaning_prompt(text_chunk)
        cleaned_text = self._call_model_api(cleaning_prompt, text_chunk, "sentence_repair")
        
        reconstruction_prompt = self._create_reconstruction_prompt(cleaned_text)
        final_text = self._call_model_api(reconstruction_prompt, cleaned_text, "paragraph_reconstruction")
        
        return final_text

    def batch_clean_text(self, text_chunks: List[str]) -> List[str]:
        return [self.clean_text_chunk(chunk) for chunk in text_chunks]


def main():
    engine = TextCleaningEngine()
    
    sample_texts = [
        "这是一个测试文本。它包含一些不的句子，和一些语法误。另一个例子！这里有更多的问题，比如标,.k符号的使不当？",
        "另一个测试段落。句子不完整。标点符号错误，还有重复的内容重复的内容。"
    ]
    
    print("开始文本清洗...")
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n--- 示例 {i} ---")
        print(f"原始文本: {text}")
        
        cleaned_text = engine.clean_text_chunk(text)
        print(f"清洗后: {cleaned_text}")


if __name__ == "__main__":
    main()
