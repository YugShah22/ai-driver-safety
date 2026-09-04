# 🚗 AI Driver Safety & Intelligence Platform

A production-quality full-stack AI platform that analyzes dashcam driving footage using computer vision, deep learning and machine learning to identify road objects, understand the driving environment, extract driving-risk features, and generate a comprehensive driving-risk assessment.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User / Browser                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTPS
┌───────────────────────────▼─────────────────────────────────────┐
│              Frontend  (Next.js + TypeScript + Tailwind)        │
│  Landing Page · Dashboard · Upload · Results · Auth             │
└───────────┬───────────────────────────────────┬─────────────────┘
            │ REST / JSON                        │ Supabase JS SDK
            │                                   │
┌───────────▼────────────┐          ┌───────────▼─────────────────┐
│   Backend  (FastAPI)   │          │          Supabase            │
│  /health               │          │  PostgreSQL · Auth           │
│  /api/v1/analyze       │          │  Storage · Realtime          │
│  /api/v1/results       │          └─────────────────────────────┘
└───────────┬────────────┘
            │ Internal
┌───────────▼────────────────────────────────────────────────────┐
│                        ML Pipeline                              │
│  Video Processing → Object Detection (YOLO) → CNN Features     │
│  Transfer Learning → Feature Extraction → Risk Engine          │
│  ANN Classifier → XGBoost Scorer → Risk Assessment             │
└────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
ai-driver-safety/
├── frontend/          # Next.js application (TypeScript, Tailwind, App Router)
├── backend/           # FastAPI application (Python)
├── ml/                # ML/AI pipeline modules
│   ├── cnn/               # Custom CNN models
│   ├── transfer_learning/ # Pre-trained model fine-tuning
│   ├── object_detection/  # YOLO-based object detection
│   ├── segmentation/      # Road/lane segmentation
│   ├── feature_extraction/# Driving feature extractors
│   ├── ann/               # Artificial Neural Network classifier
│   ├── traditional_ml/    # scikit-learn & XGBoost models
│   ├── risk_engine/       # Risk scoring & assessment logic
│   ├── training/          # Training scripts & pipelines
│   ├── evaluation/        # Model evaluation & metrics
│   ├── inference/         # Inference pipeline
│   └── video_processing/  # OpenCV video utilities
├── models/            # Trained model weights & checkpoints
├── data/              # Datasets (gitignored; structure only)
├── scripts/           # Utility & automation scripts
└── docs/              # Architecture & API documentation
```

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** ≥ 18.x
- **Python** ≥ 3.10
- **npm** ≥ 9.x
- A **Supabase** project (free tier works)

---

### 1. Clone & Setup Environment

```bash
git clone <repo-url>
cd ai-driver-safety
```

Copy environment templates:

```bash
# Backend
cp backend/.env.example backend/.env

# Frontend
cp frontend/.env.local.example frontend/.env.local
```

Fill in your Supabase credentials in both files.

---

### 2. Start the Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Verify: `http://localhost:8000/health` → `{"status": "ok"}`  
API Docs: `http://localhost:8000/docs`

---

### 3. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open: `http://localhost:3000`

---

## 🔐 Environment Variables

### Frontend (`frontend/.env.local`)

| Variable | Description |
|---|---|
| `NEXT_PUBLIC_SUPABASE_URL` | Your Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Supabase anonymous/public key |
| `NEXT_PUBLIC_FASTAPI_URL` | FastAPI backend URL (default: `http://localhost:8000`) |

> ⚠️ **NEVER** add `SUPABASE_SERVICE_ROLE_KEY` to the frontend environment.

### Backend (`backend/.env`)

| Variable | Description |
|---|---|
| `SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service-role key (server-side only) |
| `DATABASE_URL` | PostgreSQL connection string |
| `FASTAPI_URL` | This server's URL |

---

## 🤖 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 15, TypeScript, Tailwind CSS, App Router |
| **Backend** | Python, FastAPI, Uvicorn |
| **AI/ML** | PyTorch, OpenCV, scikit-learn, XGBoost, Ultralytics YOLO, CNN, Transfer Learning (ResNet/EfficientNet) ANN |
| **Database** | Supabase PostgreSQL |
| **Auth** | Supabase Auth |
| **Storage** | Supabase Storage |
| **Realtime** | Supabase Realtime |

---

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest tests/ -v

# Frontend (when implemented)
cd frontend
npm run test
```

---

## 📖 Documentation

See the `/docs` directory for:
- `architecture.md` — System design & data flow
- `api.md` — API reference
- `ml-pipeline.md` — ML pipeline documentation
- `deployment.md` — Deployment guide

---

## 🛣️ Roadmap

- [x] **Phase 1** — Project scaffold, FastAPI health endpoint, Next.js landing page
- [ ] **Phase 2** — Supabase integration, Auth, video upload pipeline
- [ ] **Phase 3** — ML pipeline (YOLO object detection, OpenCV video processing)
- [ ] **Phase 4** — CNN + Transfer Learning feature extraction
- [ ] **Phase 5** — Risk engine (ANN + XGBoost scoring)
- [ ] **Phase 6** — Dashboard, realtime results, reporting

---

## 📄 License

MIT License — see `LICENSE` for details.
"# AI-Driver-Safety-intelligence-Platform" 
"# AI-Driver-Safety-intelligence-Platform" 
"# AI-Driver-Safety-intelligence-Platform" 
