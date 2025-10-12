import json
import re

def extract_qa_pairs(data_text):
    """
    从文本中提取问答对数据
    
    Args:
        data_text (str): 包含JSON数据的原始文本
        
    Returns:
        list: 提取的问答对列表，解析失败时返回空列表
    """
    if not data_text or not isinstance(data_text, str):
        print("错误：输入数据为空或不是字符串")
        return []
    
    # 多种JSON格式匹配模式
    json_patterns = [
        r'```json\s*(\[.*?\])\s*```',  # 标准 ```json [...] ```
        r'```\s*(\[.*?\])\s*```',      # 简写 ``` [...] ```
        r'(\[.*\])',                    # 直接匹配数组
        r'(\{.*\})'                     # 直接匹配对象
    ]
    
    json_str = None
    used_pattern = None
    
    # 尝试不同的匹配模式
    for pattern in json_patterns:
        match = re.search(pattern, data_text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            used_pattern = pattern
            break
    
    if not json_str:
        print("未找到JSON数据")
        print(f"输入文本前500字符: {data_text[:500]}...")
        return []
    
    print(f"找到JSON数据，使用模式: {used_pattern}")
    print(f"JSON字符串长度: {len(json_str)}")
    
    try:
        # 直接解析JSON
        data = json.loads(json_str)
        print(f"成功解析JSON，数据类型: {type(data)}")
        return data
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print("尝试清理和修复JSON格式...")
        
        # 尝试修复常见的JSON格式问题
        try:
            fixed_json = clean_json_string(json_str)
            data = json.loads(fixed_json)
            print("修复后成功解析JSON")
            return data
        except json.JSONDecodeError as e2:
            print(f"修复后仍然解析失败: {e2}")
            print(f"有问题的JSON片段: {json_str[:200]}...")
            return []

def clean_json_string(json_str):
    """
    清理和修复JSON字符串中的常见问题
    
    Args:
        json_str (str): 原始JSON字符串
        
    Returns:
        str: 修复后的JSON字符串
    """
    if not json_str:
        return json_str
    
    # 移除BOM字符和其他不可见字符
    json_str = json_str.strip().replace('\ufeff', '')
    
    # 修复常见的JSON格式问题
    fixes = [
        # 修复没有引号的属性名
        (r'(\s*)(\w+)(\s*):', r'\1"\2"\3:'),
        # 修复单引号
        (r"'", '"'),
        # 修复末尾逗号
        (r',\s*}', '}'),
        (r',\s*]', ']'),
        # 修复多余的逗号
        (r',,', ','),
        # 修复没有转义的双引号
        (r'([^\\])"', r'\1\\"')
    ]
    
    for pattern, replacement in fixes:
        json_str = re.sub(pattern, replacement, json_str)
    
    return json_str

# 可选：增强版本，支持更多格式
def extract_qa_pairs_enhanced(data_text):
    """
    增强版的QA对提取函数，支持更多数据格式
    
    Args:
        data_text (str): 包含JSON数据的原始文本
        
    Returns:
        list: 提取的问答对列表
    """
    # 首先尝试标准提取
    result = extract_qa_pairs(data_text)
    if result:
        return result
    
    # 如果标准提取失败，尝试其他格式
    print("尝试其他数据格式...")
    
    # 尝试提取多个JSON对象
    json_objects = re.findall(r'\{[^{}]*"question"[^{}]*"answer"[^{}]*\}', data_text)
    if json_objects:
        print(f"找到 {len(json_objects)} 个可能的问答对象")
        qa_pairs = []
        for obj_str in json_objects:
            try:
                # 尝试修复和解析每个对象
                fixed_obj = clean_json_string(obj_str)
                qa_obj = json.loads(fixed_obj)
                if 'question' in qa_obj and 'answer' in qa_obj:
                    qa_pairs.append(qa_obj)
            except:
                continue
        
        if qa_pairs:
            print(f"成功提取 {len(qa_pairs)} 个问答对")
            return qa_pairs
    
    print("所有提取方法都失败")
    return []

# 使用示例
if __name__ == "__main__":
    # 测试数据
    test_text = """
    以下是一些问答对：
    
    ```json
    [
        {
            "question": "问题1",
            "answer": "答案1"
        },
        {
            question: "问题2",
            answer: "答案2"
        }
    ]
    ```
    """
    
    result = extract_qa_pairs(test_text)
    print(f"提取结果: {result}")
