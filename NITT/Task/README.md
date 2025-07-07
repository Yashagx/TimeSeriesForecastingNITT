
# 🧠 COVID-19 Time Series Forecasting Using Multimodal Deep Learning

This repository contains the codebase and experimental workflow for my research internship at **NIT Trichy** on **Time Series Forecasting** of COVID-19 case numbers using advanced **deep learning** and **ensemble models**.

---

## 📌 Project Title:
**Forecasting COVID-19 Case Counts Using Multimodal Time Series and Deep Neural Ensemble Models**

---

## 👨‍🔬 Research Institution:
**National Institute of Technology, Tiruchirappalli (NIT Trichy)**  
**Department of Computer Applications**  
**Internship Duration:** June–July 2025

---

## 🎯 Objective

To build a robust and accurate forecasting system that predicts **new COVID-19 case counts** using:
- Past numerical time series data (epidemiological)
- Social media/news textual signals
- Climate and location metadata

---

## 🧰 Technologies & Tools

### 💻 Programming Languages & Libraries
- `Python 3.10+`
- `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`
- `Scikit-learn` for preprocessing and metrics
- `TensorFlow`, `Keras`, `PyTorch` for model building
- `NLTK`, `spaCy`, `transformers (HuggingFace)` for NLP
- `pmdarima`, `statsmodels` for ARIMA models
- `NetworkX`, `PageRank` for graph-based summarization

### 🧠 Deep Learning Models
- BPNN (Backpropagation Neural Network)
- Elman RNN (Recurrent Neural Network)
- LSTM (Long Short-Term Memory)
- ANFIS (Adaptive Neuro-Fuzzy Inference System)
- Transformer-based models (fine-tuned BERT)

### ⚙ Optimization Algorithms
- SCWOA: Self-Adaptive Chaotic Whale Optimization Algorithm (for ensemble weight tuning)

### 🌐 APIs & Real-Time Data Sources
- [api.rootnet.in](https://api.rootnet.in/) – Real-time COVID-19 India stats
- Twitter Developer API – Geotagged tweets with COVID signals
- News Articles (custom web scraping or news APIs)

### 🛠 Platforms & Tools
- Jupyter Notebooks, Google Colab, VS Code
- SQLite3/CSV for local persistence
- Git & GitHub for version control

---

## 📊 Methodology

### Phase 1: Baseline Modeling
- Extract epidemiological time series for Indian states
- Models: AR, ARMA, ARIMA
- Combined SEIR + ARIMA framework (epidemic + statistical)

### Phase 2: Multimodal Supervised Learning
- Build dataset combining:
  - Lagged case counts
  - Text embeddings from social media/news
  - Metadata: State, region, climate, population
- Preprocess text using BERT (fine-tuned on COVID news/tweets)
- Generate embeddings → feed to deep models

### Phase 3: Ensemble Modeling
- Train BPNN, Elman RNN, LSTM, ANFIS individually
- Combine predictions using SCWOA-based ensemble model
- Evaluate on metrics: MAE, RMSE, MAPE

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE    | Mean Absolute Error |
| RMSE   | Root Mean Squared Error |
| MAPE   | Mean Absolute Percentage Error |
| R²     | Coefficient of Determination |

---

## 📂 Directory Structure

```
📁 covid19-forecasting-nittrichy
├── data/                     # Raw & processed data files
├── notebooks/                # Jupyter notebooks per phase
├── models/                   # Trained models and weights
├── src/                      # Source code modules
│   ├── preprocessing.py
│   ├── models/
│   ├── utils/
├── ensemble/                 # SCWOA logic and ensemble pipelines
├── api/                      # Real-time ingestion scripts
├── results/                  # Visualizations and predictions
├── README.md                 # This file
```

---

## 🔍 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/covid19-forecasting-nittrichy.git
   cd covid19-forecasting-nittrichy
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the preprocessing and training pipeline from Jupyter/Colab:
   ```bash
   jupyter notebook
   ```

---

## 🧪 Experiments Conducted

- ✅ Extractive summarization of COVID articles using PageRank graph algorithms
- ✅ Fine-tuning BERT on COVID-specific corpus
- ✅ COVID prediction using time + text signals (hybrid inputs)
- ✅ SCWOA for optimal neural network ensemble weights
- ✅ Compared AR, ARMA, ARIMA vs deep models on Indian state-wise data

---

## 🏁 Results Snapshot

- MAE < 500 for high-case states like Maharashtra & Kerala
- LSTM outperformed ARIMA in volatile trend regions
- Ensemble model reduced RMSE by 12–18% compared to base learners

---

## 🤝 Acknowledgements

- **NIT Trichy** Department of Computer Applications
- Internship Supervisor: Prof. Divya Mam
- Dataset Sources: Rootnet.in, Twitter, Kaggle

---

## 📜 License

This project is licensed under the MIT License.

---

## 📬 Contact

**Yash Agarwal**  
Email: yashagarwal@example.com  
GitHub: [@yashagarwal](https://github.com/YOUR_USERNAME)
