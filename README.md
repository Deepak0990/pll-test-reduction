# AI-Optimized PLL Test Time Reduction Framework

An end-to-end industrial machine learning and full-stack system designed to optimize Phase-Locked Loop (PLL) semiconductor testing by replacing slow, worst-case static test loops with data-driven per-die decisions.

## 🚀 Key Features
* **Machine Learning Jitter Prediction:** Uses a LightGBM regression model to predict RMS jitter based on fast parametric chip measurements[cite: 3, 16].
* **Conformal Prediction Safety Shield:** Implements split conformal calibration to mathematically bound error rates and guarantee strict Defective Parts Per Million (DPPM) quality escape budgets[cite: 4, 7, 8].
* **FastAPI Backend Microservice (`api/main.py`):** Handles real-time die evaluations, leakage pre-screening gates, and decision logic (`SKIP` vs. `RUN`).
* **Streamlit Interactive Dashboard (`app/streamlit_app.py`):** Provides an operator interface with parametric slider controls, adaptive lock-time polling analysis, and impact metric visualizations.

---

## 🛠️ Project Architecture
- `api/main.py`: FastAPI server handling live inference and decision routing.
- `models/train_model.py`: Model training, hyperparameter configuration, and conformal calibration script.
- `app/streamlit_app.py`: Streamlit frontend visualization and simulator app.
- `data/`: Contains simulation datasets for jitter and lock-time behavior[cite: 2].

---

## ⚙️ How to Run Locally

1. **Activate your virtual environment:**
   ```bash
   # Windows PowerShell
   .\venv\Scripts\Activate 