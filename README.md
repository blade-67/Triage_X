# 🧠 AI-Assisted Emergency Triage System

An intelligent machine learning application that prioritizes emergency room patients based on vital signs and clinical indicators. This system uses a voting classifier combining Logistic Regression and Random Forest to provide accurate triage predictions.

## 🎯 Project Overview

Emergency departments face critical challenges in patient prioritization. This AI system automates the triage process by analyzing patient demographics, vital signs, and clinical indicators to classify patients into four urgency levels:

- **🟢 Non-Urgent** (Level 0) - Low priority, routine assessment
- **🟡 Urgent** (Level 1) - Moderate priority, standard triage
- **🔴 Very Urgent** (Level 2) - High priority, evaluation within 30 minutes
- **🚨 Critical** (Level 3) - Life-threatening, immediate physician evaluation

## 📊 Key Features

- **Smart ML Pipeline**: Voting classifier combining two models for robust predictions
- **Healthcare-Optimized**: Uses SMOTE for handling class imbalance (common in medical data)
- **Missing Data Handling**: KNN imputation for realistic hospital scenarios
- **Interactive Web UI**: Streamlit-based interface for easy patient assessment
- **High Performance**: Detailed metrics including accuracy, precision, recall, and F1-scores
- **Production Ready**: Serialized models for deployment

## 📋 Project Structure

```
Triage_x/
├── app.py                              # Streamlit web application
├── model.py                            # ML model training & evaluation
├── data_gen.py                         # Synthetic dataset generation
├── requirements.txt                    # Python dependencies
├── ai_emergency_triage_dataset.csv    # Training dataset (generated)
├── model_voting.pkl                   # Trained voting classifier
├── model_lr.pkl                       # Trained logistic regression model
├── model_rf.pkl                       # Trained random forest model
├── imputer.pkl                        # KNN imputer for missing values
└── feature_columns.pkl                # Feature column order
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/emergency-triage-system.git
cd Triage_x
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Training Dataset (Optional)
If you don't have the dataset, generate synthetic data:
```bash
python data_gen.py
```

### 4. Train the Models
```bash
python model.py
```

This will:
- Load and preprocess the dataset
- Handle missing values with KNN imputation
- Apply SMOTE for class balance
- Train Logistic Regression and Random Forest models
- Create an ensemble Voting Classifier
- Save all models as .pkl files
- Display detailed performance metrics

### 5. Run the Web Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Model Architecture

### Data Pipeline
1. **Data Loading**: CSV dataset with 13 features
2. **Preprocessing**: 
   - KNN Imputation (5 neighbors) for missing values
   - Feature standardization
3. **Class Balancing**: SMOTE (Synthetic Minority Over-sampling)
4. **Model Training**: Dual-model ensemble
5. **Prediction**: Soft voting classifier

### Models Used
- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: Captures non-linear patterns (100 trees)
- **Voting Classifier**: Combines both models for robustness

## 📈 Input Features

### Demographics
- **Age**: 1-100 years
- **Gender**: Female (0) / Male (1)
- **Chronic Condition**: Yes/No indicator

### Vital Signs
- **Heart Rate**: 40-180 bpm
- **Systolic BP**: 70-200 mmHg
- **Diastolic BP**: 40-120 mmHg
- **SpO₂**: 75-100%
- **Respiratory Rate**: 8-45 breaths/min
- **Temperature**: 95.0-104.0°F

### Clinical Indicators
- **Pain Score**: 0-10 scale
- **Consciousness**: Unconscious/Conscious
- **Arrival Mode**: Walk-in/Ambulance

## 💡 Usage Example

```python
# Input patient data
patient_data = {
    "age": 45,
    "gender": 1,  # Male
    "heart_rate": 110,
    "systolic_bp": 140,
    "diastolic_bp": 85,
    "spo2": 94,
    "respiratory_rate": 22,
    "temperature": 101.2,
    "pain_score": 8,
    "consciousness": 1,  # Conscious
    "arrival_mode": 1,  # Ambulance
    "chronic_condition": 1  # Yes
}

# Get prediction (via web app)
# Output: 🔴 Very Urgent (High priority)
# Confidence: 92.3%
```

## 🔧 Configuration & Customization

### Modify Model Parameters
Edit `model.py` to adjust:
```python
# SMOTE configuration
smote = SMOTE(random_state=42, sampling_strategy='auto')

# Random Forest hyperparameters
model2 = RandomForestClassifier(
    n_estimators=100,      # Increase for better accuracy (slower training)
    max_depth=10,          # Control overfitting
    random_state=42,
    n_jobs=-1
)

# KNN Imputer neighbors
imputer = KNNImputer(n_neighbors=5)
```

### Dataset Generation Parameters
Edit `data_gen.py` to adjust:
```python
N = 5000  # Dataset size
# Modify statistical distributions for realistic data
```

## 📊 Model Performance

Example metrics from training run:

```
Voting Classifier (Combined) Results:
   Accuracy:  0.9234
   Precision: 0.9178
   Recall:    0.9234
   F1-Score:  0.9201
```

Detailed per-class metrics available in console output after training.

## 🌐 Deployment

### Deploy to Streamlit Cloud
1. Push code to GitHub
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Create new app and link to your repository
4. Select `app.py` as main file

### Fix Common Deployment Issues
If you encounter "installer returned non-zero exit code":
- Ensure `requirements.txt` has compatible versions
- Remove version pinning (use `>=` instead of `==`)
- Include transitive dependencies like `python-dateutil` and `pytz`

### Deploy Locally with Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 📦 Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
joblib>=1.3.0
python-dateutil>=2.8.2
pytz>=2023.3
```

## 🔒 Important Notes

- **Medical Disclaimer**: This system is for demonstration purposes. Real medical triage should involve qualified healthcare professionals.
- **Data Privacy**: Always ensure patient data handling complies with HIPAA and local regulations.
- **Model Updates**: Retrain models periodically with real hospital data for production use.

## 🐛 Troubleshooting

### Models not found error
```
Solution: Run 'python model.py' to train and save models
```

### Import errors
```
Solution: pip install -r requirements.txt
```

### Streamlit port already in use
```
Solution: streamlit run app.py --server.port 8502
```

### Missing values in predictions
```
Solution: Ensure all input fields are filled; KNN imputation handles missing training data
```

## 📝 Files Description

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web interface for patient triage |
| `model.py` | ML pipeline: training, evaluation, model saving |
| `data_gen.py` | Generates synthetic emergency room dataset |
| `requirements.txt` | Python package dependencies |

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Imbalanced-learn SMOTE](https://imbalanced-learn.org/stable/over_sampling.html#smote)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Emergency Triage Protocols](https://www.cdc.gov/niosh/topics/emres/chemrtriage.html)

## 👨‍💻 Author

Your Name / Organization

## 📧 Contact

For questions or suggestions, please open an issue or contact the repository maintainer.

---

**Made with ❤️ for better emergency care**
