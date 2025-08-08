Read me

The Objective and Challenges of this experiment:

Goal: Build a multi-agent LLM framework for MDT medical consultations with:
->Role-based reasoning & collaboration.
->Knowledge accumulation (CorrectKB & ChainKB).
->Self-evolving mechanisms for accuracy improvement.

Key Challenges:
->Managing long dialogue histories.
->Ensuring consensus among agents.
->Retaining and reusing medical knowledge.

Techinical Infrastructure
LLM Backbone: GPT-4-turbo / Deepseek 
Multi-Agent Framework: 
--------> Use libraries like LangChain for agent orchestration or AutoGen for role-based collaboration.
-------->Define agent roles (e.g., Radiologist, Oncologist, General Practitioner).
Knowledge Base:

OpenAI:
Required libraries :
faiss-cpu==1.7.4
langchain==0.3.25
openai==1.30.1
tiktoken==0.9.0
sqlalchemy==2.0.40
psycopg2-binary==2.9.10  
flask==3.1.0
streamlit==1.45.0
numpy==1.24.0  # Required for faiss-cpu 1.7.4
packaging==24.2

Deepseek : 
langchain
deepseek
pydantic
pyyaml
fastapi
uvicorn
numpy
scikit-learn
pandas
tqdm
sentence-transformers
faiss-cpu


python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install faiss-cpu==1.7.4
pip install langchain openai tiktoken sqlalchemy psycopg2-binary flask streamlit packaging

Structure of the project:
MDTeam/
├── README.md
├── agents
│   ├── base_agent.py
│   ├── cot_reviewer.py
│   ├── lead_physician.py
│   ├── primary_care.py
│   ├── safety_ethics.py
│   └── specialist_obgyn.py
├── app
│   ├── api.py
│   └── static
├── config.yaml
├── core
│   ├── __init__.py
│   ├── consultation_system.py
│   └── discussion_manager.py
├── data
│   ├── medqa
│   └── pubmedqa
├── knowledge_bases
│   ├── chain_kb
│   └── correct_kb
├── main.py
├── outputs
│   ├── consultations
│   └── evals
├── requirements.txt
├── scripts
│   ├── fine_tune.py
│   └── init_kb.py
├── tests
│   ├── test_agents.py
│   └── test_kb.py
├── utils
│   ├── consensus.py
│   ├── embeddings.py
│   └── eval.py
└── venv
    ├── bin
    ├── etc
    ├── include
    ├── lib
    ├── pyvenv.cfg
    └── share


after creating all codes:
Set OpenAI API Key (if using GPT-3.5/4):

bash
export OPENAI_API_KEY="your-api-key"

Execute the script:

bash
python3 mdt_gpt.py

expected output: 
Case: The patient has a 3cm lung nodule with spiculated margins. What are the next steps?

Radiologist: The findings suggest a high probability of malignancy. Recommend a PET-CT scan and biopsy for further evaluation.

Oncologist: Given the concern for malignancy, options include surgical resection if localized, or chemotherapy/radiation if metastatic. A biopsy is urgent.