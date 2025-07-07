# 🧠 COVID-19 Time Series Forecasting Using Multimodal Deep Learning

This repository contains the comprehensive codebase, experimental setup, and implementation details for a research internship project conducted at **NIT Trichy**, focused on **forecasting COVID-19 case counts** using a **multimodal deep learning and ensemble modeling framework**. The project brings together epidemiological data, climate variables, and social media/news signals to build a robust prediction model.

---

## 📌 Project Title:

**Forecasting COVID-19 Case Counts Using Multimodal Time Series and Deep Neural Ensemble Models**

---

## 👨‍픬 Research Institution:

**National Institute of Technology, Tiruchirappalli (NIT Trichy)**
**Department of Computer Applications**
**Internship Duration:** May-August 2025
**Research Mode:** Remote + On-campus

---

## 🎯 Objective

To design a **highly accurate and generalizable forecasting framework** that predicts daily new COVID-19 cases in Indian states by integrating multiple data modalities, including:

* Historical epidemiological time series (case counts, recoveries, fatalities)
* Climate and geographical features (temperature, humidity, region)
* Textual signals (social media, news headlines, public sentiment)
* Metadata such as population density, healthcare capacity, and mobility

This integrated approach helps capture hidden patterns, non-linear dependencies, and policy effects more effectively than unimodal baselines.

---

## 🧰 Technologies & Tools

### 💻 Programming Languages & Libraries

* `Python 3.10+`
* Data Handling: `NumPy`, `Pandas`
* Visualization: `Matplotlib`, `Seaborn`, `Plotly`
* Modeling: `Scikit-learn`, `TensorFlow`, `Keras`, `PyTorch`
* NLP: `NLTK`, `spaCy`, `transformers`, `TfidfVectorizer`
* Statistical Modeling: `pmdarima`, `statsmodels`
* Graph-based NLP: `NetworkX` for PageRank summarization

### 🧠 Deep Learning Architectures

* **BPNN** (Backpropagation Neural Network)
* **Elman RNN** (Contextual Recurrent Model)
* **LSTM** (Long Short-Term Memory)
* **ANFIS** (Adaptive Neuro-Fuzzy Inference System)
* **Transformer-based BERT** (fine-tuned for COVID corpus)

### ⚙ Optimization Algorithms

* **SCWOA**: Self-Adaptive Chaotic Whale Optimization Algorithm

  * Used for optimal ensemble weighting
  * Adaptive convergence to avoid local minima

### 🌐 External Data Sources & APIs

* [api.rootnet.in](https://api.rootnet.in/) – Real-time COVID-19 case updates for India
* **Twitter API v2** – Keyword-based geotagged tweet extraction
* **NewsAPI & Web Scraping** – Extraction of COVID-19 headlines and summaries
* **Custom Climate CSV** – Delhi daily temperature data (2021–2024)

### ⚖️ Development & Experimentation Tools

* IDEs: Jupyter Notebooks, Google Colab, VS Code
* Version Control: Git, GitHub
* Data Storage: CSV, SQLite3 (lightweight DB for structured inputs)
* Hardware: Colab GPU runtime, Local CPU testing

---

## 📊 Methodology

### ✅ Phase 1: Epidemiological Baseline Modeling

* Raw data preprocessing (Rootnet + CSV)
* Generate lagged features (daily, weekly trends)
* Models built: AR, MA, ARMA, ARIMA
* Combined **SEIR + ARIMA** for hybrid modeling

### 🤖 Phase 2: Multimodal Supervised Learning

* Merge epidemiological, climate, and text data
* Preprocess and clean tweets/news
* Extract TF-IDF and BERT embeddings
* Combine structured (numerical) and unstructured (textual) features
* Models trained:

  * BPNN, LSTM, Elman RNN, ANFIS
  * Evaluate using MAE, RMSE, MAPE, R²

### ♻ Phase 3: Ensemble Modeling

* Base models: BPNN, LSTM, Elman, ANFIS
* Use SCWOA for dynamic weight optimization
* Combine predictions using weighted average strategy
* Validate on Indian states with varying case patterns

---

## 📊 Evaluation Metrics

| Metric   | Description                                                       |
| -------- | ----------------------------------------------------------------- |
| **MAE**  | Mean Absolute Error – Overall deviation from true counts          |
| **RMSE** | Root Mean Squared Error – Penalizes larger errors more strongly   |
| **MAPE** | Mean Absolute Percentage Error – % error relative to actual count |
| **R²**   | Coefficient of Determination – Model fit score                    |

---

## 📂 Directory Structure

```
📁 covid19-forecasting-nittrichy
├── data/                     # Raw datasets and processed CSVs
├── notebooks/                # Phase-wise Jupyter notebooks
├── models/                   # Pretrained weights and saved models
├── src/                      # Core logic & modules
│   ├── preprocessing.py      # Data cleaning, merging, feature engineering
│   ├── models/               # Deep learning architecture scripts
│   ├── utils/                # Helper methods and evaluators
├── ensemble/                 # SCWOA and voting pipeline
├── api/                      # Real-time data ingestion scripts
├── results/                  # Visualizations, plots, evaluation outputs
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview (this file)
```

---

## 🧪 How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/covid19-forecasting-nittrichy.git
cd covid19-forecasting-nittrichy
```

### Step 2: Install All Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Launch the Jupyter Pipeline

```bash
jupyter notebook
```

Open the notebook for the desired phase (e.g., `notebooks/phase2_multimodal.ipynb`) and run all cells sequentially.

---

## 🔮 Key Experiments Conducted

* ✅ **PageRank Summarization** of COVID news articles
* ✅ **Tweet Embedding using BERT** (fine-tuned on COVID dataset)
* ✅ **Epidemiological Trend Modeling** using ARIMA + SEIR
* ✅ **Hybrid Input Modeling**: Numeric + Text + Climate
* ✅ **Neural Network Ensemble** optimized using SCWOA
* ✅ **Comparative Study** of traditional vs deep models

---

## 🏁 Results Summary

* **LSTM** outperformed classical ARIMA in fluctuating case regions
* **MAE** consistently below **500 cases** for Maharashtra, Kerala
* **Ensemble model** showed **12–18% lower RMSE** over base models
* BERT-based embeddings captured **semantic case trends** in news/tweets

---

## 🙏 Acknowledgements

* **NIT Trichy**, Department of Computer Applications
* Internship Guides: **Prof. Dr. E. Sivasankar** & **Prof. Divya**
* Data Providers: [api.rootnet.in](https://api.rootnet.in), Twitter API, Kaggle Datasets
* Research tools: Google Colab, HuggingFace, NetworkX

---

## 📜 License

This repository is released under the **MIT License**. Feel free to use, modify, and cite this work.

---

## 👨‍🏫 Academic Supervisor

**Prof. Dr. E. Sivasankar**
Associate Professor
Room No. 202 A, Department of Computer Science and Engineering
National Institute of Technology, Tiruchirappalli – 620015, Tamil Nadu, India

---

## 📨 Contact

**Yash Agarwal**
Research Intern – NIT Trichy (Summer 2025)
Email: [ya8009672@gmail.com](mailto:ya8009672@gmail.com)
GitHub: [github.com/Yashagx](https://github.com/Yashagx)

---

> For detailed methodology, visualizations, and model performances, please refer to the `notebooks/` and `results/` directories.
