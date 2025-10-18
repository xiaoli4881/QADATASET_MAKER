# import json
# from typing import List
# import requests


# class TextCheckEngine:
    
#     def __init__(self, base_url: str = 'http://localhost:11434', model: str = 'qwen2.5:7b'):
#         self.base_url = base_url.rstrip('/')
#         self.model = model
#         self._setup_system_prompts()
    
#     def _setup_system_prompts(self) -> None:
#         self.cleaning_prompts = {
#             "qa_check": """你是一个医学方面的专家。请对输入文本执行以下操作：

#             1. 判断是否是专业医学疾病相关的问答知识，如果不是就删除，输出空，如果是就保留
#             2. 判断问答对是否是中文，如果是其他语言请翻译成专业中文，输出问答对
#             3. 判断问题的专业性，如果不专业就改写成专业问题，输出问答对
#             4. 删除关于某一类期刊或指南等类似问答对，输出空
            
#             输出要求：
#             - 直接返回处理后的问答对
#             - 不要添加任何解释或标记
#               请严格按照以下JSON数组格式输出，不要包含任何其他内容：
#             {'human': '阻塞性睡眠呼吸暂停的患病率是多少？', 'assistant': '男性中阻塞性睡眠呼吸暂停的患病率为33.99%，女性为17.49%；由于肥胖率增加，阻塞性睡眠呼吸暂停的患病率呈逐年升高趋势。'},
#             {'human': '阻塞性睡眠呼吸暂停可能导致哪些临床症状？', 'assistant': '该病症可引起一系列临床症状，包括夜尿增多、晨起口干、头痛、日间嗜睡、注意力不集中和认知功能下降等，并显著降低生活质量及交通事故发生风险。'}
#             """
#         }
    
#     def _call_model_api(self, prompt: str, text_chunk: str, system_prompt_key: str) -> str:
#         url = f"{self.base_url}/api/generate"
        
#         payload = {
#             "model": self.model,
#             "prompt": prompt,
#             "system": self.cleaning_prompts[system_prompt_key],
#             "stream": False,
#             "options": {
#                 "temperature": 0.1,
#                 "top_p": 0.8,
#                 "num_predict": min(len(text_chunk) * 2, 4000),
#                 "repeat_penalty": 1.2
#             }
#         }
        
#         try:
#             response = requests.post(
#                 url,
#                 json=payload,
#                 headers={"Content-Type": "application/json"},
#                 timeout=60
#             )
#             response.raise_for_status()
            
#             result = response.json()
#             return result.get('response', '').strip()
            
#         except requests.exceptions.RequestException as e:
#             print(f"API调用失败: {e}")
#             return text_chunk
#         except json.JSONDecodeError as e:
#             print(f"JSON解析失败: {e}")
#             return text_chunk
#         except Exception as e:
#             print(f"未知错误: {e}")
#             return text_chunk
    
#     def _create_cleaning_prompt(self, text_chunk: str) -> str:
#         return f"需要处理的文本：\n{text_chunk}\n\n请返回清洗后的文本："
    
    
#     def QA_text_check(self, text_chunk: str) -> str:
#         if not text_chunk or not text_chunk.strip():
#             return text_chunk
            
#         cleaning_prompt = self._create_cleaning_prompt(text_chunk)
#         cleaned_text = self._call_model_api(cleaning_prompt, text_chunk, "qa_check")

#         return cleaned_text

# #     def batch_clean_text(self, text_chunks: List[str]) -> List[str]:
# #         return [self.clean_text_chunk(chunk) for chunk in text_chunks]


# # def main():
# #     engine = TextCleaningEngine()
    
# #     sample_texts = [
# #         "这是一个测试文本。它包含一些不的句子，和一些语法误。另一个例子！这里有更多的问题，比如标,.k符号的使不当？",
# #         "另一个测试段落。句子不完整。标点符号错误，还有重复的内容重复的内容。"
# #     ]
    
# #     print("开始文本清洗...")
    
# #     for i, text in enumerate(sample_texts, 1):
# #         print(f"\n--- 示例 {i} ---")
# #         print(f"原始文本: {text}")
        
# #         cleaned_text = engine.clean_text_chunk(text)
# #         print(f"清洗后: {cleaned_text}")


# # if __name__ == "__main__":
# #     main()

# engine = TextCheckEngine()
# with open("out_new/QA.txt", "r", encoding="utf-8") as f:
#     code = f.readlines()
#     for i in code:
#         print(i)
#         checked_text = engine.QA_text_check(i)
#         print(checked_text)
#         if checked_text != " " and checked_text != "{}" and checked_text != "空":
#             with open ("out_new/QA_check_new.txt", "a+", encoding="utf-8") as f2:
#                 f2.write(checked_text + "\n")




import json
from typing import List
import requests


class TextCheckEngine:
    
    def __init__(self, base_url: str = 'http://localhost:11434', model: str = 'qwen2.5:7b'):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self._setup_system_prompts()
    
    def _setup_system_prompts(self) -> None:
        self.cleaning_prompts = {
            "qa_check": """ 角色设定
            你是一个专业的医疗内容质量评估专家，擅长识别和过滤低质量的医疗问答内容。

            评估标准
            必须去除的情况（出现任一即去除）：
            答案为空 - 助理回复为空字符串或null

            答案重复问题 - 答案只是简单重复或轻微改写问题，没有新增信息

            答非所问 - 答案与问题完全不相关或偏离原问题方向

            内容错误 - 包含事实性错误或危险的医疗建议

            过于模糊 - 答案过于笼统，缺乏具体有用的信息

            优质问答对特征：
            问题清晰明确，有实际价值

            答案准确专业，信息量充足

            问答高度相关，直接解决问题

            内容具有临床实用性

            处理流程
            逐一分析每个问答对

            应用上述标准进行严格评判

            直接去除不合格的问答对

            只输出通过质量检查的问答对

            输出要求
            只输出通过质量检查的问答对

            保持原有的JSON格式

            不包含任何评估过程说明

            不保留被去除的问答对 """
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
    
    
    def QA_text_check(self, text_chunk: str) -> str:
        if not text_chunk or not text_chunk.strip():
            return text_chunk
            
        cleaning_prompt = self._create_cleaning_prompt(text_chunk)
        cleaned_text = self._call_model_api(cleaning_prompt, text_chunk, "qa_check")

        return cleaned_text

#     def batch_clean_text(self, text_chunks: List[str]) -> List[str]:
#         return [self.clean_text_chunk(chunk) for chunk in text_chunks]


# def main():
#     engine = TextCleaningEngine()
    
#     sample_texts = [
#         "这是一个测试文本。它包含一些不的句子，和一些语法误。另一个例子！这里有更多的问题，比如标,.k符号的使不当？",
#         "另一个测试段落。句子不完整。标点符号错误，还有重复的内容重复的内容。"
#     ]
    
#     print("开始文本清洗...")
    
#     for i, text in enumerate(sample_texts, 1):
#         print(f"\n--- 示例 {i} ---")
#         print(f"原始文本: {text}")
        
#         cleaned_text = engine.clean_text_chunk(text)
#         print(f"清洗后: {cleaned_text}")


# if __name__ == "__main__":
#     main()

engine = TextCheckEngine()
with open("out/QA_check.txt", "r", encoding="utf-8") as f:
    code = f.readlines()
    for i in code:
        print(i)
        checked_text = engine.QA_text_check(i)
        print(checked_text)
        if checked_text != " " and checked_text != "{}" and checked_text != "空":
            with open ("out/QA_check_new.txt", "a+", encoding="utf-8") as f2:
                f2.write(checked_text + "\n")