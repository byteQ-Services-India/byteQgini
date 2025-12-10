🧠 byteQgennie
Local PDF Intelligence • FAISS Retrieval • Ollama Reasoning
<p align="center"> <img src="https://img.shields.io/badge/Build-Passing-00C853?style=for-the-badge"> <img src="https://img.shields.io/badge/Version-1.0.0-2962FF?style=for-the-badge"> <img src="https://img.shields.io/badge/Framework-Flask-blue?style=for-the-badge"> <img src="https://img.shields.io/badge/Vector_DB-FAISS-orange?style=for-the-badge"> <img src="https://img.shields.io/badge/Embeddings-MiniLM--L6--v2-purple?style=for-the-badge"> <img src="https://img.shields.io/badge/LLM-Ollama_llama3.2-red?style=for-the-badge"> <img src="https://img.shields.io/badge/License-OpenSource-green?style=for-the-badge"> </p>
<p align="center"></p>
🌟 What is byteQgennie?

byteQgennie is a local Retrieval-Augmented AI assistant that reads PDFs, converts them into semantic embeddings, stores them in FAISS, and answers user questions with context-aware refinement using Ollama (llama3.2).

🔐 Fully local — no cloud calls

📚 Automatic knowledge ingestion from /data

⚡ Fast semantic search via FAISS

🔁 Continuous live update when new PDFs arrive

🤖 Human-like response refinement via local LLM

🚀 Current Capabilities
Feature	Status
PDF ingestion & chunking	✔ Live
FAISS vector index caching	✔ Live
Incremental updates on new PDFs	✔ Live
llama3.2 refined AI answers	✔ Live
Web hook /get?msg= response	✔ Live
Greetings / farewells handling	✔ Live
Automatic background scanner	✔ Live
Fully offline RAG processing	✔ Live
🔧 Tech Stack
Layer	Technology
Web Server	Flask
Vector DB	FAISS
Embeddings	HuggingFace MiniLM-L6-v2
LLM	Ollama llama3.2
Chunking	RecursiveCharacterTextSplitter
Loader	PyPDFLoader
Docstore	InMemoryDocstore
Scheduler	Threading loop
🧱 Architecture Overview
 PDFs (/data)
      │
      ▼
PyPDFLoader → Chunking → HuggingFace Embeddings → FAISS Index
                                                      │
                                                      ▼
                                           Top-1 Vector Match
                                                      │
                                                      ▼
                                            llama3.2 Response


✔ No external API
✔ Offline capable
✔ Self-learning via new PDFs

📂 Folder Structure
byteQgennie/
├─ app.py
├─ data/                     # drop PDFs here
├─ precomputed_data/
│   ├─ index.faiss
│   ├─ docs.pkl
│   └─ processed_files.pkl
├─ templates/
│   └─ index.html
└─ README.md

⚙️ Setup
Install system
git clone https://github.com/byteQ-services/byteQgennie.git
cd byteQgennie
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Prepare storage
mkdir data precomputed_data templates

Install Ollama
curl https://ollama.ai/install.sh | sh
ollama pull llama3.2

Launch
python app.py


👉 Opens at: http://127.0.0.1:5000/

🧠 How Retrieval Works

PDFs placed in /data

Split into 400-character chunks with overlap

Embeddings generated & fed into FAISS

Best match (k=1) returned

llama3.2 refines context into natural language

🔁 Auto Updating

New PDFs → detected automatically

Embedded + appended → index updated

Saved into:

index.faiss

docs.pkl

processed_files.pkl

No index rebuild needed.

🔥 API
/get?msg=your+text

Accepts GET & POST

Returns refined answer

Example:

curl "http://127.0.0.1:5000/get?msg=explain chapter 2"

🛠 Known Limitations
Issue	Detail
top-1 chunk only	multi-chunk merging planned
periodic timer set to 1440s	mislabeled as 24h (configurable soon)
embedding model init per query	will move to persistent instance
no metadata return (page/file)	upcoming feature
🛣 Roadmap

v1.5

Multi-chunk context retrieval

Page number + filename in answers

v2.0

Chat UI redesign (React + animations)

Document deletion + re-indexing

v3.0

3D onboarding + memory persona

Multi-model selection (Q4 2025)

🤝 Contributing

Fork → Improve → PR
Every PR must include:

Clean code comments

One-line summary commit message

Explanation if FAISS/index logic modified

Example commit:

feat: optimized chunk search with k=5 expansion

🆘 Troubleshooting
Problem	Solution
FAISS corrupt	delete precomputed_data/ & restart
model missing	ollama pull llama3.2
PDFs not reading	ensure OCR / selectable text
answers irrelevant	increase k or re-embed
🪪 License

📌 Open Source (MIT recommended / pending addition)

🎯 Final Thoughts

byteQgennie is built to grow daily:

add PDFs → it learns

remove PDFs → re-index upcoming

integrate UI → production-ready

Fully local. Fully controllable.
Your documents. Your model. Your intelligence.
