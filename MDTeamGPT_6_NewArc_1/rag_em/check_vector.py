
#!/usr/bin/env python3
"""
FAISS Index Validation Tool
Purpose: Verify if the generated index.faiss and index.pkl files contain correct metadata and content
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from collections import defaultdict
import argparse
import random

def load_vector_db(index_path):
    """Load FAISS vector database"""
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

def method1_direct_inspection(db):
    """Method 1: Stratified random sampling"""
    print("\n" + "="*60)
    print("Method 1: Document Sampling (1 per specialty + randoms)")
    print("="*60)
    
    all_docs = list(db.docstore._dict.values())
    specialties = set(doc.metadata.get('specialty') for doc in all_docs)
    
    # Get at least one sample per specialty
    samples = []
    for spec in specialties:
        spec_docs = [d for d in all_docs if d.metadata.get('specialty') == spec]
        if spec_docs:
            samples.append(random.choice(spec_docs))
    
    # Fill remaining slots with random docs
    remaining_slots = 5 - len(samples)
    if remaining_slots > 0:
        samples.extend(random.sample(
            [d for d in all_docs if d not in samples],
            min(remaining_slots, len(all_docs))
        ))
    
    for i, doc in enumerate(samples):
        print(f"\n📄 Sample {i+1}/{len(samples)}")
        print(f"📌 Source: {doc.metadata.get('source', 'Not set')}")
        print(f"🏥 Specialty: {doc.metadata.get('specialty', 'Not set')}")
        print("\nContent preview:")
        print(doc.page_content[:200] + "...")
        print("-" * 50)

def method2_semantic_search(db):
    """Method 2: Verify with MedQA test questions"""
    print("\n" + "="*60)
    print("Method 2: Semantic Search with MedQA Test Questions")
    print("="*60)
    
    # Selected questions from MedQA dataset (USMLE style)
    medqa_questions = [
        "A 62-year-old woman comes to the physician because of increasing blurring of vision in both eyes. She says that the blurring has made it difficult to read, although she has noticed that she can read a little better if she holds the book below or above eye level. She also requires a bright light to look at objects. She reports that her symptoms began 8 years ago and have gradually gotten worse over time. She has hypertension and type 2 diabetes mellitus. Current medications include glyburide and lisinopril. When looking at an Amsler grid, she says that the lines in the center appear wavy and bent. An image of her retina, as viewed through fundoscopy is shown. Which of the following is the most likely diagnosis?",
        "A 23-year-old man comes to the physician because of recurrent episodes of chest pain, shortness of breath, palpitations, and a sensation of choking. The symptoms usually resolve with deep breathing exercises after about 5 minutes. He now avoids going to his graduate school classes because he is worried about having another episode. Physical examination is unremarkable. Treatment with lorazepam is initiated. The concurrent intake of which of the following drugs should be avoided in this patient?",
        "A case-control study looking to study the relationship between infection with the bacterium Chlamydia trachomatis and having multiple sexual partners was conducted in the United States. A total of 100 women with newly diagnosed chlamydial infection visiting an outpatient clinic for sexually transmitted diseases (STDs) were compared with 100 women from the same clinic who were found to be free of chlamydia and other STDs. The women diagnosed with this infection were informed that the potential serious consequences of the disease could be prevented only by locating and treating their sexual partners. Both groups of women were queried about the number of sexual partners they had had during the preceding 3 months. The group of women with chlamydia reported an average of 4 times as many sexual partners compared with the group of women without chlamydia; the researchers, therefore, concluded that women with chlamydia visiting the clinic had significantly more sexual partners compared with women who visited the same clinic but were not diagnosed with chlamydia. What type of systematic error could have influenced the results of this study?",
        "A 52-year-old female with a history of poorly-controlled diabetes presents to her primary care physician because of pain and tingling in her hands. These symptoms began several months ago and have been getting worse such that they interfere with her work as a secretary. She says that the pain is worse in the morning and she has been woken up at night by the pain. The tingling sensations have been located primarily in the thumb, index and middle fingers. On physical exam atrophy of the thenar eminence is observed and the pain is reproduced when the wrist is maximally flexed. The most likely cause of this patient's symptoms affects which of the nerves shown in the image provided?",
        "A 66-year-old man undergoes a coronary artery bypass grafting. Upon regaining consciousness, he reports that he cannot see from either eye and cannot move his arms. Physical examination shows bilaterally equal, reactive pupils. A fundoscopy shows no abnormalities. An MRI of the brain shows wedge-shaped cortical infarcts in both occipital lobes. Which of the following is the most likely cause of this patient's current symptoms?",
        "A 35-year-old man arrives at the emergency department within minutes after a head-on motor vehicle accident. He suffered from blunt abdominal trauma, several lacerations to his face as well as lacerations to his upper and lower extremities. The patient is afebrile, blood pressure is 45/25 mmHg and pulse is 160/minute. A CBC is obtained and is most likely to demonstrate which of the following?"
    ]
    
    for query in medqa_questions:
        print(f"\n🔍 Query: '{query}'")
        results = db.similarity_search(query, k=2)
        for j, doc in enumerate(results):
            print(f"\nResult {j+1}:")
            print(f"✅ Match confidence: {1/(j+1):.1f}")  # Simple ranking metric
            print(f"📂 Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"🏥 Specialty: {doc.metadata.get('specialty', 'Unknown')}")
            print("\nRelevant content:")
            print(doc.page_content[:500] + "...")
            print("-" * 50)

def method3_metadata_analysis(db):
    """Method 3: Metadata statistical analysis"""
    print("\n" + "="*60)
    print("Method 3: Metadata Statistical Analysis")
    print("="*60)
    
    stats = {
        "total_chunks": len(db.docstore._dict),
        "sources": set(),
        "specialties": defaultdict(int)
    }
    
    for doc in db.docstore._dict.values():
        stats["sources"].add(doc.metadata.get("source"))
        stats["specialties"][doc.metadata.get("specialty")] += 1
    
    print(f"\n📊 Total chunks: {stats['total_chunks']}")
    print(f"📂 Unique sources: {len(stats['sources'])}")
    
    print("\nSpecialty distribution:")
    for spec, count in sorted(stats['specialties'].items(), key=lambda x: x[1], reverse=True):
        print(f"- {spec}: {count} docs ({count/stats['total_chunks']:.1%})")
    
    print("\nSource files:")
    for source in sorted(stats['sources']):
        print(f"- {source}")

def main():
    parser = argparse.ArgumentParser(description='FAISS Index Validation Tool')
    parser.add_argument('--index', type=str, default='data/EM_textbook/FAISS_index',
                       help='FAISS index path (without extension)')
    args = parser.parse_args()
    
    print("🔍 Starting FAISS index validation...")
    try:
        db = load_vector_db(args.index)
        print(f"✅ Successfully loaded index with {len(db.docstore._dict)} chunks")
        
        print("\nRunning validation tests...")
        #method1_direct_inspection(db)
        #method2_semantic_search(db)
        method3_metadata_analysis(db)
        
        print("\n🎉 Validation completed successfully!")
    except Exception as e:
        print(f"\n❌ Validation failed: {str(e)}")
        print("Possible solutions:")
        print("- Verify index path exists")
        print("- Check file permissions")
        print("- Try regenerating the index")

if __name__ == "__main__":
    main()


#     #!/usr/bin/env python3
# """
# FAISS Index Metadata Analyzer
# Purpose: Analyze and visualize metadata distribution in FAISS index
# """

# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from collections import defaultdict
# import matplotlib.pyplot as plt
# import argparse

# def load_vector_db(index_path):
#     """Load FAISS vector database"""
#     embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
#     return FAISS.load_local(
#         index_path,
#         embeddings,
#         allow_dangerous_deserialization=True
#     )

# def analyze_metadata(db, output_prefix="metadata_stats"):
#     """Analyze metadata and generate visualizations"""
#     # Collect statistics
#     stats = {
#         "total_chunks": len(db.docstore._dict),
#         "sources": set(),
#         "specialties": defaultdict(int)
#     }
    
#     for doc in db.docstore._dict.values():
#         stats["sources"].add(doc.metadata.get("source", "Unknown"))
#         stats["specialties"][doc.metadata.get("specialty", "Unknown")] += 1
    
#     # Print text summary
#     print("\n📊 Metadata Statistics Summary:")
#     print(f"Total chunks: {stats['total_chunks']}")
#     print(f"Unique sources: {len(stats['sources'])}")
#     print("\nSpecialty Distribution:")
#     for spec, count in sorted(stats['specialties'].items(), key=lambda x: x[1], reverse=True):
#         print(f"- {spec}: {count} chunks ({count/stats['total_chunks']:.1%})")
    
#     # Generate visualizations
#     specialties = list(stats['specialties'].keys())
#     counts = list(stats['specialties'].values())
    
#     # Bar chart
#     plt.figure(figsize=(10, 6))
#     plt.bar(specialties, counts)
#     plt.title('Chunk Distribution by Specialty')
#     plt.xlabel('Specialty')
#     plt.ylabel('Number of Chunks')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.savefig(f'{output_prefix}_bar.png')
#     print(f"\n📈 Saved bar chart to: {output_prefix}_bar.png")
    
#     # Pie chart (for top 10 specialties if many)
#     plt.figure(figsize=(8, 8))
#     if len(specialties) > 10:
#         # Combine smaller specialties into "Other"
#         sorted_specs = sorted(stats['specialties'].items(), key=lambda x: x[1], reverse=True)
#         top_specs = [x[0] for x in sorted_specs[:9]]
#         other_count = sum(x[1] for x in sorted_specs[9:])
        
#         pie_labels = top_specs + ['Other']
#         pie_counts = [stats['specialties'][s] for s in top_specs] + [other_count]
#     else:
#         pie_labels = specialties
#         pie_counts = counts
    
#     plt.pie(pie_counts, labels=pie_labels, autopct='%1.1f%%')
#     plt.title('Specialty Distribution (Percentage)')
#     plt.savefig(f'{output_prefix}_pie.png')
#     print(f"📊 Saved pie chart to: {output_prefix}_pie.png")

# def main():
#     parser = argparse.ArgumentParser(description='FAISS Metadata Analyzer')
#     parser.add_argument('--index', type=str, default='data/EM_textbook/FAISS_index',
#                        help='FAISS index path (without extension)')
#     parser.add_argument('--output', type=str, default='metadata_stats',
#                        help='Output prefix for saved charts')
#     args = parser.parse_args()
    
#     print("🔍 Starting FAISS metadata analysis...")
#     try:
#         db = load_vector_db(args.index)
#         print(f"✅ Successfully loaded index with {len(db.docstore._dict)} chunks")
#         analyze_metadata(db, args.output)
#         print("\n🎉 Analysis completed successfully!")
#     except Exception as e:
#         print(f"\n❌ Analysis failed: {str(e)}")

# if __name__ == "__main__":
#     main()