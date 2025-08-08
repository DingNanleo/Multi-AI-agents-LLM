import re
from pathlib import Path
from typing import List,Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.schema import Document
from collections import defaultdict

class MedicalVectorProcessor:
    def __init__(self, use_openai=False):
        self.embeddings = OpenAIEmbeddings() if use_openai else HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self._init_splitters()
        self.specialty_keywords = {
            "Trauma Surgery": ["fracture", "hemorrhage", "gunshot", "amputation", "exploratory laparotomy", "damage control"],
            "Pediatrics": ["child", "neonatal", "vaccine", "growth chart", "ADHD", "pediatrician", "well-baby"],
            "Toxicology": ["overdose", "poison", "antidote", "venom", "heavy metals", "naloxone", "toxidrome"],
            "Neurology": ["stroke", "seizure", "Alzheimer's", "migraine", "EEG", "multiple sclerosis", "neuropathy"],
            "Pulmonology": ["COPD", "asthma", "ventilator", "pneumonia", "bronchoscopy", "pulmonary fibrosis", "ABG"],
            "Gastroenterology": ["endoscopy", "colonoscopy", "IBS", "GERD", "cirrhosis", "Crohn's", "hepatitis"],
            "Infectious Disease": ["antibiotics", "sepsis", "HIV", "tuberculosis", "COVID-19", "antimicrobial", "viral load"],
            "Orthopedics": ["fracture", "ACL", "arthroplasty", "spinal fusion", "carpal tunnel", "dislocation", "cast"],
            "Obstetrics": ["prenatal", "C-section", "placenta", "eclampsia", "ultrasound", "postpartum", "OB/GYN"],
            "Psychiatry": ["depression", "bipolar", "schizophrenia", "SSRI", "psychosis", "anxiety", "therapy"],
            "Radiology": ["X-ray", "MRI", "CT scan", "ultrasound", "fluoroscopy", "mammogram", "contrast"],
            "Anesthesiology": ["intubation", "propofol", "epidural", "general anesthesia", "sedation", "pain management"],
            "Nephrology": ["dialysis", "AKI", "CKD", "kidney stone", "glomerulonephritis", "creatinine", "ESRD"],
            "Hematology": ["anemia", "leukemia", "hemoglobin", "coagulation", "blood transfusion", "lymphoma", "DVT"],
            "Endocrinology": ["diabetes", "insulin", "thyroid", "hormone", "osteoporosis", "metabolic", "HbA1c"],
            "Rheumatology": ["arthritis", "lupus", "gout", "autoimmune", "biologics", "Sjögren's", "inflammation"],
            "Dermatology": ["acne", "biopsy", "melanoma", "eczema", "psoriasis", "rash", "Mohs surgery"],
            "General Medicine": ["primary care", "PCP", "referral", "annual physical", "hypertension", "preventive care"],
            "Urology": ["kidney stone", "BPH", "cystoscopy", "prostate", "UTI", "incontinence", "vasectomy"],
            "Cardiology": ["ECG", "echocardiogram", "heart failure", "arrhythmia", "stent", "myocardial", "hypertension"],
            "Emergency": ["trauma", "resuscitation", "ER", "critical care", "CPR", "triage", "life-threatening"],
        }

        # 2. 急诊子专业映射（新）
        self.emergency_subspecialties = {
        
            # 创伤急症
            "Trauma Surgery": [
                "gunshot wound", "stab wound", "polytrauma", "hemorrhagic shock",
                "flail chest", "tension pneumothorax", "FAST exam", "abdominal trauma",
                "traumatic brain injury", "amputation", "crush injury"
            ],
             # 儿科急症
            "Pediatrics": [
                "neonatal resuscitation", "pediatric code", "child abuse", "SIDS",
                "bronchiolitis", "croup", "epiglottitis", "pediatric trauma",
                "febrile seizure", "intussusception", "Reye syndrome"
            ],
             # 心脏急症
            "Cardiology": [
                "STEMI", "NSTEMI", "chest pain", "cardiac arrest", "ACS", 
                "ventricular fibrillation", "unstable angina", "aortic dissection",
                "tamponade", "cardiogenic shock", "complete heart block"
            ],
            # 中毒急症
            "Toxicology": [
                "overdose", "opioid toxicity", "TCA overdose", "carbon monoxide",
                "organophosphate", "caustic ingestion", "anticholinergic crisis",
                "serotonin syndrome", "lithium toxicity", "methanol poisoning"
            ],
            # 神经急症
            "Neurology": [
                "CVA", "hemorrhagic stroke", "ischemic stroke", "status epilepticus",
                "meningitis", "encephalitis", "GCS <8", "brain herniation",
                "spinal cord compression", "myasthenic crisis", "Guillain-Barré"
            ],
            # 呼吸急症
            "Pulmonology": [
                "respiratory arrest", "ARDS", "pulmonary embolism", "tension pneumothorax",
                "severe asthma", "COPD exacerbation", "massive hemoptysis", 
                "difficult airway", "foreign body aspiration", "pulmonary edema"
            ],
            # 消化急症
            "Gastroenterology": [
                "GI bleed", "esophageal varices", "perforated ulcer", "bowel obstruction",
                "acute pancreatitis", "toxic megacolon", "liver failure", 
                "Mallory-Weiss tear", "volvulus", "mesenteric ischemia"
            ],
            # 感染急症
            "Infectious Disease": [
                "septic shock", "meningococcemia", "necrotizing fasciitis",
                "toxic shock syndrome", "rabies exposure", "malaria complications",
                "dengue shock", "Ebola", "anthrax", "plague"
            ],
            # 骨科急症
            "Orthopedics": [
                "open fracture", "compartment syndrome", "cauda equina",
                "septic arthritis", "dislocation", "spinal fracture",
                "pelvic fracture", "pathologic fracture", "osteomyelitis"
            ],
            # 妇产急症
            "Obstetrics": [
                "eclampsia", "postpartum hemorrhage", "placental abruption",
                "uterine rupture", "amniotic fluid embolism", "shoulder dystocia",
                "ruptured ectopic", "HELLP syndrome", "uterine inversion"
            ],
            # 精神急症
            "Psychiatry": [
                "suicidal ideation", "violent behavior", "catatonia",
                "neuroleptic malignant", "akathisia", "serotonin syndrome",
                "alcohol withdrawal", "delirium tremens", "excited delirium"
            ],
            # 影像急症
            "Radiology": [
                "contrast reaction", "aortic rupture", "pneumothorax",
                "bowel perforation", "cerebral hemorrhage", "foreign body",
                "dissection", "ischemic bowel", "abscess"
            ],
            # 麻醉急症
            "Anesthesiology": [
                "difficult airway", "failed intubation", "malignant hyperthermia",
                "local anesthetic toxicity", "anaphylaxis", "opioid overdose",
                "bronchospasm", "aspiration", "total spinal"
            ],
            # 肾脏急症
            "Nephrology": [
                "hyperkalemia", "uremic encephalopathy", "dialysis emergency",
                "acute kidney injury", "renal colic", "bladder rupture",
                "nephrotic crisis", "contrast nephropathy"
            ],
            # 血液急症
            "Hematology": [
                "DIC", "massive transfusion", "thrombotic thrombocytopenic purpura",
                "hemolytic crisis", "neutropenic fever", "coagulopathy",
                "hemophilia bleed", "sickle cell crisis"
            ],
            # 内分泌急症
            "Endocrinology": [
                "DKA", "HHS", "thyroid storm", "myxedema coma",
                "adrenal crisis", "hypoglycemia", "pheochromocytoma crisis",
                "hypercalcemic crisis"
            ],
            # 风湿急症
            "Rheumatology": [
                "vasculitic emergency", "scleroderma renal crisis",
                "acute gout", "SLE flare", "myositis", "ankylosing spondylitis fracture"
            ],
            # 皮肤急症
            "Dermatology": [
                "SJS", "TEN", "necrotizing fasciitis", "angioedema",
                "erythroderma", "purpura fulminans", "toxic epidermal necrolysis"
            ],
            # 泌尿急症
            "Urology": [
                "testicular torsion", "priapism", "renal colic", "urinary retention",
                "Fournier gangrene", "bladder rupture", "uretral trauma"
            ]
        }

    def _init_splitters(self):
        """初始化分块器"""
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "Chapter"), ("##", "Section")]
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "。", "!", "?", "..."],
            keep_separator=True  # 保留分隔符避免语义断裂
        )

    def process_file(self, file_path: str) -> List[Document]:
        text = Path(file_path).read_text(encoding='utf-8')
        chunks = self.header_splitter.split_text(text)  # 先分块
        
        final_chunks = []
        for chunk in chunks:
            # 对每个分块单独检测专业
            chunk_specialty = self._detect_specialty(chunk.page_content)
            
            # 处理过大分块
            if len(chunk.page_content) > 800:
                sub_chunks = self.text_splitter.split_documents([chunk])
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata.update({
                        "source": Path(file_path).stem,
                        "specialty": chunk_specialty  # 子块继承父块专业
                    })
                final_chunks.extend(sub_chunks)
            else:
                chunk.metadata.update({
                    "source": Path(file_path).stem,
                    "specialty": chunk_specialty  # 直接赋值
                })
                final_chunks.append(chunk)
        return final_chunks

    def _detect_specialty(self, text: str) -> str:
        """Enhanced specialty detection with emergency mapping"""
        #self.debug_classification(text)

        text_lower = text.lower()
        keyword_counts = defaultdict(int)
    
        # 统计所有匹配的关键词频次
        for spec, keywords in self.specialty_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    keyword_counts[spec] += 1

        # print("\n=== DEBUG: Keyword Counts ===")  # Debug info
        # for spec, count in keyword_counts.items():
        #     print(f"{spec}: {count} hits")

        if not keyword_counts:
            print("DEBUG: No keywords matched, returning 'General'")
            return "General Medicine"

        # 返回频次最高的专业（同频次时按预设优先级）
        max_count = max(keyword_counts.values())
        candidates = [k for k,v in keyword_counts.items() if v == max_count]

        #print(f"\nDEBUG: Max count={max_count}, Candidates={candidates}")  # Debug info

        # # 优先级：急诊子专业 > 常规专业 > 通用急诊
        # for spec in candidates:
        #     if spec in self.emergency_subspecialties:
        #         return spec
        # if "Emergency" in candidates:
        #     print("DEBUG: Selected 'Emergency' (fallback)")
        #     return "Emergency"
        
        # print(f"DEBUG: Default selection, first candidate: '{candidates[0]}'")
        return candidates[0]  # 默认返回第一个
    
    def debug_classification(self, text):
        text_lower = text.lower()
        print("\n=== Debug Start ===")
        
        # 检查急诊子专业
        print("Emergency Subspecialty Check:")
        for spec, terms in self.emergency_subspecialties.items():
            matches = [t for t in terms if t.lower() in text_lower]
            if matches:
                print(f"→ {spec}: {matches}")
        
        # 检查常规专业
        print("\nRegular Specialty Check:")
        for spec, terms in self.specialty_keywords.items():
            if spec == "Emergency": continue
            matches = [t for t in terms if t.lower() in text_lower]
            if matches:
                print(f"→ {spec}: {matches}")
        
        print("=== Debug End ===\n")

     

    def build_index(self, dir_path: str, output_path: str) -> FAISS:
        """构建向量索引"""
        """从目录构建索引，自动处理所有.md文件"""
        dir_path = Path(dir_path)
        if not dir_path.exists():
            raise ValueError(f"Directory not found: {dir_path}")
        
        md_files = list(dir_path.glob("*.md"))
        if not md_files:
            raise ValueError(f"No .md files found in {dir_path}")
        
        print(f"Found {len(md_files)} Markdown files:")
        for md_file in md_files:
            print(f"- {md_file.name}")
            
        all_chunks = []
        for md_file in md_files:
            print(f"Processing {md_file}...")
            all_chunks.extend(self.process_file(str(md_file)))

        print(f"Total chunks: {len(all_chunks)}")
        vector_db = FAISS.from_documents(all_chunks, self.embeddings)
        if output_path:
            vector_db.save_local(output_path)
            print(f"✅ Index saved to {output_path}")
        return vector_db
    

    def update_index(self, new_dir: str, existing_index_path: str) -> FAISS:
        """
        增量更新现有向量数据库
        :param new_markdown_dir: 新增Markdown文件目录
        :param existing_index_path: 已存在的FAISS索引路径
        """
        # 1. 加载现有索引
        try:
            db = FAISS.load_local(existing_index_path, self.embeddings)
            print(f"✅ Load existing index from {existing_index_path} successfully !")
        except Exception as e:
            print(f"⚠️ Failed to load index ({str(e)}), creating new index.")
            db = FAISS.from_documents([], self.embeddings)  # 空索引

        # 2. 处理新增文件
        new_files = list(Path(new_dir).glob("*.md"))
        if not new_files:
            print(f"⏭️ No .md files found in {new_dir}")
            return db

        # 3. 去重检查（避免重复添加）
        existing_sources = set(doc.metadata.get("source") for doc in db.docstore._dict.values())
        new_chunks = []
        
        for md_file in new_files:
            if md_file.stem not in existing_sources:
                new_chunks.extend(self.process_file(str(md_file)))
                print(f"Added {md_file}")
            else:
                print(f"⏭️ Jump exist files: {md_file.name}")

        # 4. 增量添加
        if new_chunks:
            db.add_documents(new_chunks)
            print(f"🆕 New add {len(new_chunks)} chunks from {len(new_files)} files ")
        else:
            print("🔄 No new cotent for update!")

        # 5. 保存更新后的索引（覆盖原路径）
        db.save_local(existing_index_path)
        print(f"💾 Index has update and store at {existing_index_path}")
        return db

# 测试增量更新
def test_incremental_update():
    # 准备测试数据
    test_dir = "test_data"
    Path(test_dir).mkdir(exist_ok=True)
    Path(f"{test_dir}/old").mkdir(exist_ok=True)
    Path(f"{test_dir}/new").mkdir(exist_ok=True)

    Path(f"{test_dir}/old/file1.md").write_text("# 旧文件\n内容1")
    Path(f"{test_dir}/new/file2.md").write_text("# 新文件\n内容2")
    
    # 首次构建
    processor = MedicalVectorProcessor()
    processor.build_index(f"{test_dir}/old", "test_index")
    
    # 增量更新
    processor.update_index(f"{test_dir}/new", "test_index")
    
    # 验证结果
    db = FAISS.load_local("test_index", processor.embeddings)
    sources = {doc.metadata["source"] for doc in db.docstore._dict.values()}
    assert "file1" in sources and "file2" in sources
    print("✅ 测试通过")

if __name__ == "__main__":
   # test_incremental_update()
   # 初始化处理器
    processor = MedicalVectorProcessor(use_openai=False)  # 使用免费模型
    
    # 首次运行：处理包含9个Markdown的文件夹
    print("=== Building initial index ===")
    db = processor.build_index(
        dir_path="data/EM_textbook/processed_markdown",  # 存放9个初始文件的文件夹
        output_path="data/EM_textbook/FAISS_index"
    )
    
    # # 示例：后续更新（可以注释掉初始运行）
    # print("\n=== Updating index ===")
    # processor.update_index(
    #     new_dir="data/EM_textbook/processed_markdown",
    #     existing_index_path="data/EM_textbook/FAISS_index"
    # )
  