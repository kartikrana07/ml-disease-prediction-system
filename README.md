# 🏥 ML Disease Prediction System

A Machine Learning-based system that predicts multiple diseases using trained models, preprocessing pipelines, and real-world datasets — with an interactive **Streamlit** web application, **Docker** support, and deployment on **Hugging Face Spaces**.

---

## 📌 Project Overview

This project predicts the likelihood of multiple diseases based on patient health data. Each disease has its own trained ML model with a dedicated preprocessing pipeline (imputers, encoders, scalers). Users can interact with the system through a clean Streamlit web interface.

---

## 🩺 Diseases Covered

| Disease | Dataset |
|---|---|
| Breast Cancer | Wisconsin Breast Cancer Dataset |
| Diabetes | Pima Indians Diabetes Dataset |
| Heart Disease | Cleveland Heart Disease Dataset |
| Parkinson's Disease | UCI Parkinson's Dataset |

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
- Streamlit
- Docker
- Hugging Face Spaces

---

## 📊 Features

- Data preprocessing using `SimpleImputer`, `KNNImputer`, `StandardScaler`, `OneHotEncoder`
- Multiple trained ML models — one per disease
- Real-time prediction through a Streamlit web UI
- Clean modular project structure
- Dockerized for consistent, portable deployment
- Live deployment on Hugging Face Spaces

---

## 📁 Project Structure

```
ml-disease-prediction-system/
│
├── models/               # Trained .pkl model files
├── scalers/              # Saved StandardScaler objects
├── encoders/             # Saved OneHotEncoder objects
├── imputers/             # Saved imputer objects
├── datasets/             # Raw dataset CSV files
├── app.py                # Streamlit web application
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker configuration
└── README.md             # Project documentation
```

---

## 🚀 How to Run Locally

### Option 1 — Run Directly with Python

```bash
# 1. Clone the repository
git clone https://github.com/kartikrana07/ml-disease-prediction-system.git
cd ml-disease-prediction-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

App opens at `http://localhost:8501`

## 🐳 Dockerfile

```dockerfile
# Use official Python base image
FROM python:3.11

# Set working directory inside the container
WORKDIR /app


# Copy all project files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Expose the Streamlit default port
EXPOSE 7860

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
```

---

## 🤗 Live Demo — Hugging Face Spaces

The app is deployed live on **Hugging Face Spaces** and can be accessed without any local setup:

🔗 **[Click here to open the live app](https://kartikrana-7-disease-predictor.hf.space)**

> Hugging Face Spaces hosts Streamlit apps for free. The app automatically builds from the repository whenever changes are pushed.

### How Deployment Works

1. A `README.md` with special Hugging Face metadata at the top tells Spaces to use the Streamlit SDK
2. Hugging Face reads `requirements.txt` and installs all dependencies automatically
3. It runs `streamlit run app.py` and serves the app publicly
4. Any `git push` to the Space repository triggers an automatic redeploy

---

## 📦 Requirements

```
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
```

---

## 👤 Author

**Kartik Rana**
- GitHub: [@kartikrana07](https://github.com/kartikrana07)
- LinkedIn: [linkedin.com/in/kartikrana-](https://linkedin.com/in/kartikrana-)
- Email: kartikrana8284@gmail.com

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
