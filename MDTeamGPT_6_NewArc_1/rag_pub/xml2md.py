import xml.etree.ElementTree as ET
import re
from pathlib import Path
import json
import os
import hashlib

# 硬编码路径
INPUT_DIR = "data/Pubmed_paper/raw_xml"
INPUT_FILES = ["pubmed25n0001.xml", "pubmed25n1274.xml"]
OUTPUT_DIR = "data/Pubmed_paper/processed_markdown"
METADATA_FILE = "data/Pubmed_paper/processed_markdown/pubmed_metadata.json"

def clean_text(text: str) -> str:
    """基础清洗：移除XML标签和特殊字符"""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r'\s+', ' ', text).strip()

def generate_fallback_id(article: ET.Element) -> str:
    """为没有PMID的文章生成唯一ID"""
    title = clean_text(article.find(".//ArticleTitle").text) if article.find(".//ArticleTitle") else ""
    if title:
        return hashlib.md5(title.encode('utf-8')).hexdigest()[:8]
    return hashlib.md5(ET.tostring(article)).hexdigest()[:8]

def extract_metadata(article: ET.Element) -> dict:
    """提取文献元数据"""
    # 提取PMID
    pmid = None
    pmid_element = article.find(".//PMID")
    if pmid_element is not None:
        pmid = pmid_element.text
    
    # 尝试从ArticleId中获取
    if pmid is None:
        for id_type in ["pubmed", "pmid", "doi"]:
            id_element = article.find(f".//ArticleId[@IdType='{id_type}']")
            if id_element is not None:
                pmid = id_element.text
                break
    
    # 生成回退ID
    if pmid is None:
        pmid = f"no_pmid_{generate_fallback_id(article)}"
    
    # 提取其他元数据
    return {
        "pmid": pmid,
        "title": clean_text(article.find(".//ArticleTitle").text) if article.find(".//ArticleTitle") else "无标题",
        "abstract": clean_text(article.find(".//AbstractText").text) if article.find(".//AbstractText") else "",
        "journal": clean_text(article.find(".//Journal/Title").text) if article.find(".//Journal/Title") else "未知期刊",
        "year": article.find(".//PubDate/Year").text if article.find(".//PubDate/Year") else "未知年份",
        "authors": [
            f"{author.find('LastName').text}, {author.find('ForeName').text}"
            for author in article.findall(".//AuthorList/Author")
            if author.find("LastName") and author.find("ForeName")
        ],
        "mesh_terms": [
            term.find(".//DescriptorName").text
            for term in article.findall(".//MeshHeading")
            if term.find(".//DescriptorName")
        ]
    }

def sanitize_filename(filename: str) -> str:
    """确保文件名有效"""
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

def main():
    """主函数"""
    # 确保输出目录存在
    try:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"创建输出目录失败: {e}")
        return
    
    all_metadata = []
    stats = {"processed": 0, "errors": 0}

    for xml_file in INPUT_FILES:
        xml_path = Path(INPUT_DIR) / xml_file
        if not xml_path.exists():
            print(f"警告：文件 {xml_path} 不存在，已跳过")
            stats["errors"] += 1
            continue

        try:
            tree = ET.parse(xml_path)
            articles = tree.findall(".//PubmedArticle")
            if not articles:
                print(f"警告：文件 {xml_path} 中没有找到PubmedArticle节点")
                stats["errors"] += 1
                continue

            for article in articles:
                try:
                    metadata = extract_metadata(article)
                    md_content = f"""# {metadata['title']}

**PMID:** {metadata['pmid']}  
**Journal:** {metadata['journal']}, {metadata['year']}  
**Authors:** {"; ".join(metadata['authors'][:3])}{" et al." if len(metadata['authors']) > 3 else ""}

## Abstract  
{metadata['abstract']}

## MeSH Terms  
{", ".join(metadata['mesh_terms'])}
"""
                    filename = sanitize_filename(f"pubmed_{metadata['pmid']}.md")
                    output_path = Path(OUTPUT_DIR) / filename
                    
                    try:
                        output_path.write_text(md_content, encoding="utf-8")
                        all_metadata.append(metadata)
                        stats["processed"] += 1
                    except Exception as e:
                        print(f"写入文件 {filename} 失败: {e}")
                        stats["errors"] += 1
                
                except Exception as e:
                    print(f"处理文章时出错（文件 {xml_file}）: {e}")
                    stats["errors"] += 1

        except ET.ParseError as e:
            print(f"XML解析错误（文件 {xml_path}）: {e}")
            stats["errors"] += 1

    # 保存元数据
    try:
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)
        print(f"转换完成！\n成功处理 {stats['processed']} 篇文献，遇到 {stats['errors']} 个错误")
        print(f"Markdown文件保存在: {OUTPUT_DIR}\n元数据文件: {METADATA_FILE}")
    except Exception as e:
        print(f"保存元数据失败: {e}")

if __name__ == "__main__":
    main()