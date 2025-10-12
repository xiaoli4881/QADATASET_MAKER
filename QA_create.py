

import json
from typing import List
import requests


class QAcreate_Engine:
    def __init__(self, base_url='http://localhost:11434', model='qwen2.5:7b'):
        self.base_url = base_url
        self.model = model
        self.setup_system_prompts()
    
    def setup_system_prompts(self):
        self.cleaning_prompts = {
            "qa_create": """角色:
            您是一位专业的医疗专家。

            任务：
            基于给定文本进行总结归纳并生成20对必须是针对具体疾病的医疗问诊类、或者专业医学名词解释问答对。

            规则：

            问题和答案必须与针对具体疾病医疗问诊相关、或者专业医学名词解释，全部翻译成专业中文。

            答案可以直接引用原文完整句子全部翻译成专业中文。
            
            禁止生成类似本指南、本文这类问答对。
            如无明确答案，则跳过该问题。

            每问必答，不可生成无对应答案的问题。

            问题与答案须严格限定在医疗疾病问诊相关或者专业医学名词解释问答对。检查无误后输出

            输出格式：

            json
            [
                {"human": "问题1", "assistant": "完整答案句"},
                {"human": "问题2", "assistant": "完整答案句"}
            ]
            如无可生成内容，则输出空数组 []。"""
        }
    
    def call_model_api(self, prompt: str, text_chunk: str) -> str:
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": self.cleaning_prompts["qa_create"],
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
            return ""
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            return ""
    
    def validate_qa_pairs(self, qa_pairs: str, original_text: str) -> bool:
        try:
            qa_list = json.loads(qa_pairs)
            for qa in qa_list:
                if 'human' not in qa or 'assistant' not in qa:
                    return False
                if not qa['human'] or not qa['assistant']:
                    return False
                if qa['assistant'] not in original_text:
                    return False
            return True
        except:
            return False

    def create_qa_pairs(self, text_chunk: str, mode="qa_create") -> str:
        prompt = f"""
        文本内容：
        {text_chunk}

        请基于以上文本生成问题和答案，确保每个问题都能在原文中找到对应的明确答案："""   
        
        qa_pairs = self.call_model_api(prompt, text_chunk)
        
        if not qa_pairs:
            print("生成问答对失败")
        
        if not self.validate_qa_pairs(qa_pairs, text_chunk):
            print("生成的问答对验证失败，可能存在没有答案的问题")
        
        try:
            qa_list = json.loads(qa_pairs)
            if len(qa_list) < 3:
                print("生成的有效问答对数量不足")
        except:
            print( "生成的格式不是有效的JSON")
        
        if len(qa_pairs) < len(text_chunk) * 0.3:
            print("生成的问答对内容过少")
        # elif len(qa_pairs) > len(text_chunk) * 3:
        #     return qa_pairs[:len(text_chunk) * 2]
        
        if not qa_pairs:
            return ""
        else:
            return qa_pairs

    def batch_clean_text(self, text_chunks: List[str], mode="qa_create") -> List[str]:
        cleaned_chunks = []
        for chunk in text_chunks:
            cleaned_chunk = self.create_qa_pairs(chunk, mode)
            cleaned_chunks.append(cleaned_chunk)
        return cleaned_chunks


def test_qa_engine():
    engine = QAcreate_Engine()
    
    test_texts = [
        "机器学习是人工智能的一个重要分支。它通过算法让计算机从数据中学习规律，并做出预测或决策。深度学习是机器学习的一个子领域，使用神经网络模型处理复杂任务。自然语言处理是人工智能的另一个重要方向，专注于让计算机理解和生成人类语言。监督学习需要标注数据，无监督学习不需要标注数据。强化学习通过奖励机制让智能体学习最优策略。",
        "气候变化是当今世界面临的重大挑战。全球变暖导致极端天气事件频发，海平面上升威胁沿海城市。减少碳排放是应对气候变化的关键措施，需要各国共同努力。可再生能源如太阳能和风能是替代化石燃料的重要选择。植树造林可以吸收二氧化碳，缓解温室效应。国际社会通过巴黎协定来协调全球气候行动。"
    ]
    
    print("开始测试问答生成引擎...")
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n=== 测试文本 {i} ===")
        print(f"原文: {text}")
        print(f"原文长度: {len(text)} 字符")

        result = engine.create_qa_pairs(text)

        print(f"\n生成的问答对:")
        print(result)
        print(f"结果长度: {len(result)} 字符")
        
        try:
            qa_data = json.loads(result)
            print(f"成功生成 {len(qa_data)} 对问答")
            for j, qa in enumerate(qa_data, 1):
                print(f"{j}. Q: {qa['human']}")
                print(f"   A: {qa['assistant']}")
                if qa['assistant'] in text:
                    print("   ✓ 答案在原文中验证通过")
                else:
                    print("   ✗ 答案在原文中未找到")
        except:
            print("生成的格式不是有效的JSON")
        
        print("-" * 50)


if __name__ == "__main__":
    test_qa_engine()
