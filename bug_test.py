import json
from typing import Union
from pathlib import Path

def debug_json_file(file_path):
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        return
    
    print(f"\n调试文件: {file_path}")
    print("=" * 50)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                print(f"第{line_num}行: ✓ 有效")
                print(f"  内容: {data}")
            except json.JSONDecodeError as e:
                print(f"第{line_num}行: ✗ JSON解析错误")
                print(f"  错误: {e}")
                print(f"  原始内容: {line}")
                print(f"  错误位置: 字符 {e.pos}")
          
                if e.pos and e.pos < len(line):
                    start = max(0, e.pos - 20)
                    end = min(len(line), e.pos + 20)
                    context = line[start:end]
                    pointer = " " * (e.pos - start) + "^"
                    print(f"  上下文: ...{context}...")
                    print(f"          ...{pointer}...")
                break  


debug_json_file('train.json')
