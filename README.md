# 🏛️ Social Support AI — Abu Dhabi Government (DGE)

AI-powered social support eligibility assessment system built for the XDigit / DGE technical assessment.
Automates the full application workflow using LangGraph agents, local LLM (Ollama), and a RandomForest ML model — delivering decisions in minutes instead of 5-20 working days.

---

## 📺 Demo

> Form submission → 4 AI agents → ML prediction → APPROVE/DECLINE + enablement recommendations

![Pipeline](docs/pipeline_flow.png)

---

## 🏗️ Architecture
```
Streamlit UI (frontend)
      ↓
FastAPI Backend (api/)
      ↓
Master Orchestrator — LangGraph
      ↓
┌─────────────────────────────────────┐
│  Agent 1: Data Extractor            │
│  Agent 2: Data Validator            │
│  Agent 3: Eligibility Checker  ──→  RandomForest ML Model (scikit-learn + SHAP)
│  Agent 4: Decision + Enablement     │
└─────────────────────────────────────┘
      ↓
PostgreSQL · MongoDB · Qdrant · Ollama (llama3)
```

---

## ⚙️ Tech Stack

| Layer | Tools |
|---|---|
| Frontend | Streamlit |
| Backend API | FastAPI + Uvicorn |
| Agent Orchestration | LangGraph |
| Agent Reasoning | ReAct + Reflexion |
| Local LLM | Ollama (llama3) |
| Embeddings | nomic-embed-text via Ollama |
| ML Model | scikit-learn RandomForest + SHAP |
| Observability | Langfuse |
| Relational DB | PostgreSQL |
| Document DB | MongoDB |
| Vector DB | Qdrant |
| Infrastructure | Docker Compose |

---

## 🚀 How to Run (Windows)

### Prerequisites
- Python 3.10+
- Docker Desktop running
- Git
- Ollama installed → [ollama.com/download](https://ollama.com/download)

---

### Step 1 — Clone the repo
```bash
git clone https://github.com/sarbdeol/social-support-ai.git
cd social-support-ai
```

### Step 2 — Start Ollama and pull models
Open a terminal and run:
```bash
ollama serve
```
Open a second terminal:
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### Step 3 — Start all databases
```bash
cd docker
docker-compose up -d
cd ..
```
Verify with `docker ps` — you should see 4 containers running.

### Step 4 — Create virtual environment and install dependencies
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 5 — Generate synthetic data and train ML model
```bash
python ml/generate_data.py
python ml/train.py
```
Expected output: **91% accuracy** RandomForest classifier.

### Step 6 — Start FastAPI backend
Open a new terminal:
```bash
venv\Scripts\activate
python -m uvicorn api.main:app --reload --port 8000
```
API docs available at → http://localhost:8000/docs

### Step 7 — Start Streamlit UI
Open another new terminal:
```bash
venv\Scripts\activate
streamlit run frontend/app.py
```
App available at → http://localhost:8501

---

## 📁 Project Structure
```
social-support-ai/
├── agents/
│   ├── orchestrator.py     # LangGraph master orchestrator
│   ├── extractor.py        # Data extraction agent
│   ├── validator.py        # Data validation agent (Reflexion)
│   ├── eligibility.py      # ML eligibility checker agent
│   └── decision.py         # Decision + enablement agent (ReAct)
├── api/
│   ├── main.py             # FastAPI app
│   ├── routes.py           # /predict /health endpoints
│   └── schemas.py          # Pydantic models
├── frontend/
│   └── app.py              # Streamlit chatbot UI
├── ml/
│   ├── generate_data.py    # Synthetic data generator (Faker)
│   ├── train.py            # RandomForest training + SHAP
│   ├── predict.py          # Inference + SHAP explanation
│   └── artifacts/          # Saved model files
├── db/
│   ├── postgres.py
│   ├── mongodb.py
│   └── qdrant.py
├── data/
│   └── synthetic/          # Generated training data
├── docker/
│   └── docker-compose.yml  # All services
├── docs/
│   └── solution_summary.md
├── .env
├── requirements.txt
└── README.md
```

---

## 🤖 Agent Pipeline

| Agent | Role | Framework |
|---|---|---|
| Orchestrator | Routes tasks, manages state | LangGraph |
| Data Extractor | Structures raw applicant data | Ollama LLM |
| Data Validator | Cross-checks inconsistencies | Reflexion pattern |
| Eligibility Checker | Calls ML model, applies rules | FastAPI + SHAP |
| Decision Agent | Final decision + enablement recs | ReAct reasoning |

---

## 📊 ML Model Performance

| Metric | Score |
|---|---|
| Accuracy | 91% |
| Precision (APPROVE) | 91% |
| Recall (APPROVE) | 89% |
| CV Mean Accuracy | 91.4% |

**Top features:** monthly_income → employment_status → monthly_expenses → credit_score

---

## 🔍 Observability

Langfuse dashboard available at → http://localhost:3000

Traces every LLM call, agent step, token count, and latency across the full pipeline.

---

## 📄 Solution Summary

See [docs/solution_summary.md](docs/solution_summary.md) for the full 10-page solution design document including architecture decisions, tool justifications, and future improvements.

---

## 👤 Author

Built for XDigit / DGE Senior AI/ML Engineer Assessment
```

Now commit:
```
git add .
git commit -m "docs: add full README with setup instructions and architecture"
git push origin main