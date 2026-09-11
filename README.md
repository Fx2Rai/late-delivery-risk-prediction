# APL Logistics — Late Delivery Risk Prediction

## Project Overview
This project provides an end-to-end Machine Learning pipeline and an interactive Operations Dashboard (built with Streamlit) designed to predict **Late Delivery Risk** in global supply chain operations. 

By analysing historical shipping patterns, customer segments, product categories, and geographical data, this system enables logistics operators to proactively identify high-risk shipments *before* they occur and trigger expedited shipping protocols.

## Methodology & Models
The objective of this project was to predict late delivery risk using only data available *prior to shipment*, explicitly removing any post-shipment target leakage (such as real shipping days, delivery status, etc.).

### Preprocessing & Feature Engineering
- **Data Cleaning**: Handled missing values (e.g., Zipcodes, Names), removed extraneous PII columns, and resolved string-encoding issues.
- **Unified Pipeline**: Implemented a scikit-learn `ColumnTransformer` that imputes missing values, standard-scales continuous variables, and one-hot-encodes categorical variables globally to handle unseen deployment data gracefully.
- **Engineered Indicators**:
  1. `shipping_pressure_idx`: Ratio of scheduled days to order quantity.
  2. `mode_risk_flag`: Binary indicator isolating express vs. economy shipping.
  3. `regional_congestion_idx`: Target-encoded historical risk per region.
  4. `order_complexity_score`: A normalized composite metric representing order difficulty based on quantity, discount, and scheduled days.

### Machine Learning Results
Three classification models were trained and evaluated on 144k training records using a stratified 80/20 train/test split. XGBoost was selected for deployment due to its superior discrimination capability (ROC-AUC: 0.7745). Standard binary classification metrics (evaluated at the default 0.50 decision threshold) are shown below:

| Model | ROC-AUC | Binary Precision (0.50) | Recall (0.50) | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.7405 | 0.8499 | 0.5420 | 0.6619 |
| Random Forest | 0.7543 | 0.8450 | 0.5517 | 0.6676 |
| **XGBoost (Selected)** | **0.7745** | **0.8412** | **0.5635** | **0.6749** |

*Note on Operational Tiers*: In deployment, probabilities are categorized into three business risk tiers: Low (< 0.40), Medium (0.40–0.70), and High (≥ 0.70). The operational High-Risk tier achieves **89.7% precision** (95.6% for scores ≥ 0.80), ensuring that interventions target genuinely vulnerable shipments.

## Project Structure
```text
Project DS3/
├── data/                       # Stored train/test split data
├── models/                     # Pickled pipelines and metadata
│   ├── best_model.pkl          # Final XGBoost pipeline 
│   └── feature_columns.pkl     # Feature index for Streamlit
├── figures/                    # Generated ROC curves, confusion matrices, and SHAP plots
├── outputs/                    # CSV predictions and regional aggregates
├── group1_preprocessing.py     # Script: Data cleaning & split
├── group2_ml.py                # Script: Feature engineering & model training
├── group3_evaluation.py        # Script: Metrics, outputs, and explainability
├── app.py                      # Streamlit Operations Dashboard
├── requirements.txt            # Python dependencies
├── research_paper.md           # Formal academic report
└── executive_summary.md        # High-level stakeholder summary
```

## Installation & Execution

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Pipeline (Optional)**
   If you wish to retrain the models from scratch (requires `APL_Logistics.csv` in root directory):
   ```bash
   python group1_preprocessing.py
   python group2_ml.py
   python group3_evaluation.py
   ```

3. **Launch the Dashboard**
   ```bash
   streamlit run app.py
   ```

## Streamlit Dashboard Features
- **Module 1 (Risk Overview)**: KPIs and macro-level distribution charts.
- **Module 2 (Order Prediction)**: Real-time risk probability prediction on custom inputs. Utilises SHAP values to explain the top drivers for *why* an order was flagged as high or low risk.
- **Module 3 (Region & Mode Analysis)**: Geographic density heatmaps and mode performance metrics.
- **Module 4 (Action Panel)**: An operational queue dynamically filtering the highest risk orders so the logistics team can take immediate action.
