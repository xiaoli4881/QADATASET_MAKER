# QADATASET_MAKER
Automated medical Q&amp;A for dataset construction


# 使用说明：

# 安装依赖：

bash
pip install -r requirements.txt

# 版面检测网络：

https://github.com/JaidedAI/EasyOCR 自行下载


# 命令行使用：
bash
python pdf_qa_processor.py --pdf_path /path/to/your/pdf.pdf --output_dir ./results

# Python API使用：
python

from pdf_qa_processor import PDFQAProcessor

processor = PDFQAProcessor()

qa_pairs = processor.process_pdf("your_pdf.pdf", "output_dir")

# 主要特性：

✅ 完整的错误处理和日志记录

✅ 命令行接口和Python API

✅ 配置化参数

✅ 模块化设计，易于扩展

✅ 支持GPU加速

✅ 自动创建输出目录

# 输出文件：

pageo_{n}.txt - 原始提取文本

page_{n}.txt - 清洗后文本

QA.txt - 最终QA对结果

medical_qa.json -医疗问答数据集
