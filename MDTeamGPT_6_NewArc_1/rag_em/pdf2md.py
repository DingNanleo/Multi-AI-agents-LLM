import pdfplumber
import re
from pathlib import Path
from difflib import SequenceMatcher

def pdf_to_markdown(pdf_path, output_dir, max_pages=None):
    """
    将PDF转换为结构清晰的Markdown文件
    参数：
        pdf_path: PDF文件路径
        output_dir: 输出目录
        max_pages: 最大转换页数（测试时可限制）
    """
    # 创建输出目录
    md_path = Path(output_dir) / f"{Path(pdf_path).stem}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 用于检测重复内容的哈希集合
    previous_text = None
    duplicate_count = 0

    with pdfplumber.open(pdf_path) as pdf, open(md_path, 'w', encoding='utf-8') as f:
        total_pages = len(pdf.pages)
        print(f"Total pages in PDF: {total_pages}")  # 调试信息

        for i, page in enumerate(pdf.pages, 1):
            if max_pages and i > max_pages:
                break

            # 文本提取（平衡速度和准确性）
            text = page.extract_text(
                x_tolerance=2,
                y_tolerance=2,
                layout=False,  # 保持禁用布局分析以提升速度
                extra_attrs=[]  # 不需要额外属性
            ) or ""
            
            # 多重清洗流程
            clean_text = remove_footers(text)
            clean_text = fix_line_breaks(clean_text)
            clean_text = clean_to_markdown(clean_text)
            clean_text = remove_references(clean_text)
            clean_text = remove_citation_numbers(clean_text)  # 新增：移除引用数字

            # 跳过空页面
            if not clean_text.strip():
                continue

            # 相似性检测（与前一页比较）
            if previous_text and is_similar(clean_text, previous_text):
                duplicate_count += 1
                print(f"Skipping similar page {i}")
                continue
                
            previous_text = clean_text

            # 写入分页标记
            f.write(f"## Page {i}\n\n{clean_text}\n\n")  
            print(f"Processed page {i}")

        print(f"Removed {duplicate_count} duplicate pages")
    
    print(f"转换完成 -> {md_path}")

def is_similar(text1, text2, threshold=0.85):
    """检查两段文本是否相似（基于内容比例）"""
    # 忽略前100个字符（通常包含重复的标题信息）
    text1 = text1[100:] if len(text1) > 100 else text1
    text2 = text2[100:] if len(text2) > 100 else text2
    
    # 取较短的文本长度作为比较基准
    min_length = min(len(text1), len(text2))
    if min_length < 50:  # 太短的文本不比较
        return False
        
    return SequenceMatcher(None, text1[:500], text2[:500]).ratio() > threshold

def remove_citation_numbers(text):
    """移除文本中的引用数字[1][2]或1,2等"""
    # 匹配方括号中的数字 [1] [2-4] 等
    text = re.sub(r'\[\d+[-\d]*\]', '', text)
    # 匹配上标数字 ¹²³
    text = re.sub(r'[\u00B9-\u00BE\u2070-\u2079]+', '', text)
    # 匹配行内单独数字 1 2 3（前后有空格或标点）
    text = re.sub(r'(?<=\s)\d{1,2}(?=\s|[,.;])', '', text)
    return text

def remove_footers(text):
    """清除各类页脚/页眉"""
    # 匹配您提供的页脚模式
    patterns = [
        r'Downloaded \d{4}[\-­]\d{1,2}[\-­]\d{1,2}.*?Your IP is \d+\.\d+\.\d+\.\d+',
        r'Page \d+ / \d+',
        r'Chapter \d+: .+?[;;] .+? Page \d+',
        r'[\u00A9\u00AE].*All rights reserved',  # 版权符号
        r'Confidential|Proprietary',
        r'文档编号：\w+-\d+'  # 中文文档编号
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text

def remove_references(text):
    """清除参考文献章节（中英文版）"""
    ref_patterns = [
        r'(?s)^References\s*$.*', 
        r'(?s)^REFERENCES\s*$.*',
        r'(?s)^Bibliography\s*$.*'
    ]

    for pattern in ref_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    return text

def fix_line_breaks(text):
    # 规则1：连字符结尾的真实换行（如injury­prevention）
    text = re.sub(r'([a-z])[\-­]\n([a-z])', r'\1\2', text)
    
    # 规则2：非句子结尾的换行转空格
    text = re.sub(r'(?<![.!?])\n(?=[a-z])', ' ', text)
    
    # 规则3：保留项目符号/标题的换行
    text = re.sub(r'(\d+\.|\*)\s*\n', r'\1 ', text)
    return text

def clean_to_markdown(text):
    """将PDF文本转换为Markdown格式"""
    # 第一步：清除纯数字行（关键修复）
    text = re.sub(r'^(\d+\s*\.?\s*)+$', '', text, flags=re.MULTILINE)
    
    # 第二步：标题检测（必须放在数字清理后）
    text = re.sub(r'^(\d+\.\d+\s+.+)$', r'### \1', text, flags=re.MULTILINE)
    
    # 第三步：处理真实列表项（需跟随内容）
    text = re.sub(r'^(\d+)\.\s+([a-zA-Z])', r'\1. \2', text, flags=re.MULTILINE)

    # 移除多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 特殊字符处理
    text = text.replace('•', '  * ')  # 项目符号
    text = re.sub(r'[\uE000-\uF8FF]', '', text)  # 移除PDF特殊字符

    # 标准化空格
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()

def main():
    pdf_dir="data/EM_textbook/raw_pdfs_sample"
    mk_dir="data/EM_textbook/processed_markdown"
    
    for pdf_file in Path(pdf_dir).glob("*.pdf"):
        print(f"Converting {pdf_file.name}...")
        pdf_to_markdown(pdf_file, mk_dir)


if __name__ == "__main__":
    main()