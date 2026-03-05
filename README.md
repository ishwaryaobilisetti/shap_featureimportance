# SHAP Feature Importance Analysis for Customer Churn Prediction

## Project Overview

This project demonstrates the use of **SHAP (SHapley Additive exPlanations)** to interpret machine learning model predictions for a customer churn classification task. The analysis goes beyond simple accuracy metrics to understand *why* a model makes certain decisions, which is crucial for building trust in AI systems, debugging models, and deriving actionable business insights.

### Key Objectives
- Build a high-performance classification model (AUC > 0.80) to predict customer churn
- Use SHAP to provide both **global** and **local** model interpretability
- Generate actionable business insights from the interpretability analysis
- Create a comprehensive, reproducible analysis notebook

## Repository Structure

```
shap_featureimportance/
│
├── README.md                          # This file - Project documentation
├── requirements.txt                   # Python dependencies
├── shap_analysis_colab.ipynb         # Main Jupyter Notebook with complete analysis
├── churn_model.joblib                # Trained model (generated after running notebook)
└── label_encoders.joblib             # Fitted label encoders for preprocessing
```

> **Note**: The dataset is automatically downloaded from IBM's GitHub repository when running the notebook. No manual download required.

## Dataset

This project uses the **Telco Customer Churn** dataset, which contains information about:
- Customer demographics (gender, senior citizen status, partners, dependents)
- Account information (tenure, contract type, payment method, billing preferences)
- Services subscribed (phone, internet, online security, tech support, etc.)
- Monthly and total charges
- Churn status (target variable)

The dataset is automatically downloaded when running the notebook.

## Analysis Highlights

### Model Performance
- **Algorithm**: XGBoost Classifier
- **AUC-ROC Score**: **0.8312** ✅ (exceeds 0.80 requirement)
- **Accuracy**: 75.23%
- **Precision**: 52.39%
- **Recall**: 73.26%
- **F1-Score**: 61.09%
- **Validation Strategy**: Train/Test split (80/20) with stratification

### SHAP Visualizations Included
1. **Global Interpretability**
   - Summary (Beeswarm) Plot - Shows feature importance and impact distribution
   - Bar Chart - Mean absolute SHAP values for feature ranking
   - Comparison Chart - SHAP vs native XGBoost feature importance

2. **Local Interpretability**
   - Force Plots - 2 individual prediction explanations (True Positive & False Negative)
   - Waterfall Plots - 2 additive feature contribution views

3. **Feature Interactions**
   - Dependence Plot: Tenure vs SHAP (colored by Contract)
   - Dependence Plot: MonthlyCharges vs SHAP (colored by InternetService)
   - Dependence Plot: Contract vs SHAP (colored by Tenure)

## Setup Instructions

### Option 1: Google Colab (Recommended - Quick Start)
1. Upload `shap_analysis_colab.ipynb` to Google Colab
2. Run all cells (Runtime → Run all)
3. The notebook will automatically install required packages

### Option 2: Local Installation

#### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Jupyter Notebook or JupyterLab

#### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd shap_featureimportance
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

5. **Open and run the analysis**
   - Open `shap_analysis_colab.ipynb`
   - Run all cells from top to bottom (Cell → Run All)

## Key Findings Summary

### Top 5 Features by SHAP Importance
| Rank | Feature | Mean |SHAP Value| |
|------|---------|---------------------|
| 1 | **Contract** | 0.899 |
| 2 | **tenure** | 0.536 |
| 3 | **MonthlyCharges** | 0.462 |
| 4 | **TotalCharges** | 0.362 |
| 5 | **OnlineSecurity** | 0.274 |

### Top 3 Actionable Business Insights

1. **Contract Type is the Strongest Churn Predictor**
   - Customers on month-to-month contracts show significantly higher churn risk
   - **Recommendation**: Offer incentives for longer-term contract commitments

2. **Tenure and Customer Loyalty**
   - New customers (low tenure) have substantially higher churn probability
   - **Recommendation**: Implement enhanced onboarding and early engagement programs

3. **Service Bundle Opportunities**
   - Customers without additional services (online security, tech support) churn more
   - **Recommendation**: Create value-added service bundles to increase stickiness

## Technical Implementation Details

### Libraries Used
| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation and preprocessing |
| `numpy` | Numerical computations |
| `scikit-learn` | Model evaluation and preprocessing |
| `xgboost` | Gradient boosting model training |
| `shap` | Model interpretability and explanations |
| `matplotlib/seaborn` | Data visualization |
| `joblib` | Model serialization |

### SHAP Explainer Choice
We use `shap.TreeExplainer` for our XGBoost model because:
- It's optimized for tree-based models
- Provides exact SHAP values (not approximations)
- Significantly faster than model-agnostic alternatives
- Handles feature interactions properly

## Reproducibility

To ensure reproducibility:
- All random seeds are set (`random_state=42`)
- Package versions are pinned in `requirements.txt`
- The notebook runs sequentially from top to bottom
- All outputs are saved and versioned

## Video Walkthrough

📹 **[Watch the Video Walkthrough](https://drive.google.com/file/d/1Cl_tiYQPvWOSwSjN9yPQYqGPXjtZGJ53/view?usp=sharing)**

The video covers:
- Problem statement and objectives (0:00-0:45)
- Project architecture (0:45-1:15)
- Data loading and preprocessing (1:15-2:00)
- Model training with XGBoost (2:00-2:30)
- SHAP global and local interpretability (2:30-3:30)
- Challenges and business insights (3:30-4:00)

## License

This project is for educational and demonstration purposes.

## Author

[Your Name]

## Acknowledgments

- [SHAP Library](https://github.com/slundberg/shap) by Scott Lundberg
- [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle
- XGBoost development team
