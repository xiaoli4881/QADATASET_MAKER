import os
import logging
import argparse
from typing import List, Dict, Any
import numpy as np
from pdf2image import convert_from_path
import easyocr
from match import extract_qa_pairs_enhanced
from Layout_pic_Order import Layout_Order
from detect_layout import YOLODetector
from optimization import TextCleaningEngine
from QA_create import QAcreate_Engine
from crop_image import crop_image_numpy


class PDFQAProcessor:
    """PDF文档QA对处理主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.setup_logging()
        self.setup_components()
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_components(self):
        """初始化各个组件"""
        try:
            self.detector = YOLODetector("yolov11x_best.pt")
            self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=self.config.get('gpu', True))
            self.text_cleaner = TextCleaningEngine()
            self.qa_creator = QAcreate_Engine()
            self.logger.info("所有组件初始化成功")
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise
    
    def process_pdf(self, pdf_path: str, output_dir: str = "output") -> List[Dict]:
        """
        处理PDF文件的主函数
        
        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录
            
        Returns:
            List[Dict]: 提取的QA对列表
        """
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            self.logger.info(f"开始处理PDF: {pdf_path}")
            pages = convert_from_path(pdf_path, 300)
            all_qa_pairs = []
            
            for i, page in enumerate(pages):
                self.logger.info(f"处理第 {i+1} 页")
                page_qa_pairs = self.process_page(page, i, output_dir)
                all_qa_pairs.extend(page_qa_pairs)
            
            self.save_final_qa(all_qa_pairs, output_dir)
            self.logger.info(f"处理完成，共提取 {len(all_qa_pairs)} 个QA对")
            return all_qa_pairs
            
        except Exception as e:
            self.logger.error(f"处理PDF失败: {e}")
            raise
    
    def process_page(self, page, page_num: int, output_dir: str) -> List[Dict]:
        """处理单个页面"""
        try:
            img_np = np.array(page)
            ssz = img_np.shape[1]
            
            b_list = self.detector.detect_text_boxes(img_np)
            if not b_list:
                self.logger.warning(f"第 {page_num} 页未检测到文本区域")
                return []
            
            processed_boxes = self.process_boxes(b_list, img_np, ssz)
            context = self.extract_text_from_boxes(processed_boxes)
            
            self.save_page_text(context, page_num, output_dir)
            
            cleaned_text = self.text_cleaner.clean_text_chunk(context)
            self.save_cleaned_text(cleaned_text, page_num, output_dir)
            
            qa_pairs = self.extract_qa_pairs(cleaned_text)
            return qa_pairs
            
        except Exception as e:
            self.logger.error(f"处理第 {page_num} 页失败: {e}")
            return []
    
    def process_boxes(self, b_list: List, img_np: np.ndarray, ssz: int) -> List:
        """处理检测到的文本框"""
        pppd = []
        for b in b_list:
            left, right, top, bott = b[0], b[2], b[1], b[3]
            pppd.append([left, top, (right - left), ssz / 2, right, bott])
        
        ld = Layout_Order(pppd)
        boxes_to_reg = []
        
        for b in ld:
            left, top, right, bott = b[0], b[1], b[2], b[3]
            bp = crop_image_numpy(img_np, b)
            pppd.append([left, top, (right - left), ssz / 2, right, bott])
            boxes_to_reg.append(bp)
        
        return boxes_to_reg
    
    def extract_text_from_boxes(self, boxes: List) -> str:
        """从图片框中提取文本"""
        context = ""
        for box in boxes:
            try:
                result = self.reader.readtext(box, detail=0)
                context += ''.join(result) + "\n"
            except Exception as e:
                self.logger.warning(f"文本提取失败: {e}")
                continue
        return context
    
    def extract_qa_pairs(self, text: str) -> List[Dict]:
        """从文本中提取QA对"""
        try:
            qa_pairs = self.qa_creator.create_qa_pairs(text)
            enhanced_pairs = extract_qa_pairs_enhanced(qa_pairs)
            return enhanced_pairs if enhanced_pairs else []
        except Exception as e:
            self.logger.error(f"QA对提取失败: {e}")
            return []
    
    def save_page_text(self, text: str, page_num: int, output_dir: str):
        """保存原始页面文本"""
        file_path = os.path.join(output_dir, f"pageo_{page_num}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
    
    def save_cleaned_text(self, text: str, page_num: int, output_dir: str):
        """保存清洗后的文本"""
        file_path = os.path.join(output_dir, f"page_{page_num}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
    
    def save_final_qa(self, qa_pairs: List[Dict], output_dir: str):
        """保存最终的QA对"""
        file_path = os.path.join(output_dir, "QA.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            for item in qa_pairs:
                f.write(str(item))
                f.write("\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PDF文档QA对提取工具")
    parser.add_argument("--pdf_path", type=str, default="/keeson/code/lwd/Medical_QA/pdf-convert-markdown/医学指南/1_指南——PDF/失眠/2023 BSA指南：成人失眠的诊断与治疗.pdf", help="PDF文件路径")
    parser.add_argument("--output_dir", type=str, default="output", help="输出目录")
    parser.add_argument("--gpu", type=bool, default=True, help="是否使用GPU")
    
    args = parser.parse_args()
    
    config = {
        'gpu': args.gpu
    }
    
    processor = PDFQAProcessor(config)
    qa_pairs = processor.process_pdf(args.pdf_path, args.output_dir)
    
    print(f"处理完成！共提取 {len(qa_pairs)} 个QA对")
    print(f"结果保存在: {args.output_dir}/QA.txt")


if __name__ == "__main__":
    main()
