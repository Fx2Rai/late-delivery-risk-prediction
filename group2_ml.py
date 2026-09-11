"""
==============================================================================
GROUP 2 -- Feature Engineering & Machine Learning
APL Logistics -- Late Delivery Risk Prediction
==============================================================================
Parts:
  6.  Feature Engineering (all 4 spec-required indicators + extras)
  7.  Encoding & sklearn Pipeline (ColumnTransformer -- reusable for Streamlit)
  8.  Logistic Regression  (baseline)
  9.  Random Forest        (advanced)
  10. XGBoost              (advanced)

Output:
  models/preprocessor.pkl         -- fitted ColumnTransformer (shared)
  models/lr_pipeline.pkl          -- full LR pipeline
  models/rf_pipeline.pkl          -- full RF pipeline
  models/xgb_pipeline.pkl         -- full XGB pipeline
  models/best_pipeline.pkl        -- best model pipeline (used by Streamlit)
  models/region_risk_map.pkl      -- region congestion map (train-derived)
  models/feature_meta.json        -- feature names / column lists
  models/metrics_summary.json     -- all model metrics
  reports/group2_report.txt       -- full run report
  figures/...                     -- ROC, confusion matrices, feature importance
==============================================================================
"""

import os, sys, io, json, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
os.makedirs("models",  exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("figures", exist_ok=True)

RANDOM_STATE = 42
TARGET       = "Late_delivery_risk"

# ── Report logger ─────────────────────────────────────────────────────────────
report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(str(msg))

def sep(title="", char="=", width=72):
    if title:
        side = max(2, (width - len(title) - 2) // 2)
        log(char * side + f" {title} " + char * side)
    else:
        log(char * width)

# ==============================================================================
# LOAD CLEANED DATA  (from Group 1 output)
# ==============================================================================
sep("LOADING DATA")
DATA_PATH = "APL_Logistics.csv"
log(f"\nLoading raw data: {DATA_PATH}")
df_raw = pd.read_csv(DATA_PATH, encoding="latin-1")

# Apply Group 1 cleaning inline (keeps this script self-contained)
df = df_raw.copy()
df["Customer Lname"].fillna("Unknown", inplace=True)
df["Customer Zipcode"].fillna(df["Customer Zipcode"].median(), inplace=True)

DROP_COLS = [
    # PII
    "Customer Fname", "Customer Lname", "Customer Street",
    "Customer City", "Customer Zipcode", "Customer Id", "Order Customer Id",
    # High-cardinality / redundant
    "Product Name", "Order City", "Order State",
    "Order Country", "Customer Country", "Customer State",
    # LEAKAGE
    "Days for shipping (real)", "Delivery Status", "Order Status",
]
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
log(f"Shape after cleaning + leakage removal: {df.shape}")

# ==============================================================================
# TRAIN / TEST SPLIT  (before any feature engineering to prevent leakage)
# ==============================================================================
X_raw = df.drop(columns=[TARGET])
y     = df[TARGET]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
log(f"Train: {X_train_raw.shape}  |  Test: {X_test_raw.shape}")

# ==============================================================================
# PART 6 -- FEATURE ENGINEERING
# ==============================================================================
sep("PART 6 -- FEATURE ENGINEERING")

# ── 6A. Regional Congestion Indicator (target encoding -- TRAIN ONLY) ─────────
log("\n[6A] Regional Congestion Indicator")
log("     Method : Target encoding -- mean late-delivery rate per Order Region")
log("             Computed on TRAIN only, applied to test via lookup + global mean fallback")

region_risk_map = (
    X_train_raw
    .assign(target=y_train.values)
    .groupby("Order Region")["target"]
    .mean()
    .to_dict()
)
global_mean = y_train.mean()
log(f"     Regions mapped: {len(region_risk_map)}")
log(f"     Global mean (fallback for unseen regions): {global_mean:.4f}")
log(f"     Sample region rates:")
for region, rate in list(region_risk_map.items())[:5]:
    log(f"       {region:<30} -> {rate:.4f}")

joblib.dump(region_risk_map, "models/region_risk_map.pkl")

def add_features(X, region_risk_map, global_mean):
    """
    Add all engineered features. Works on any DataFrame subset.
    All computations use only input features (no target).
    """
    X = X.copy()

    # ── 1. Shipping Pressure Index (spec-required) ────────────────────────────
    # How tight is the schedule relative to the volume being shipped?
    # High pressure = fewer days per unit = more likely to be late
    X["shipping_pressure_idx"] = (
        X["Days for shipment (scheduled)"] / (X["Order Item Quantity"] + 1)
    )

    # ── 2. Mode Risk Flag (spec-required) ────────────────────────────────────
    # Express modes (First Class, Same Day) = low risk (0)
    # Economy modes (Standard, Second Class) = high risk (1)
    express_modes = {"First Class", "Same Day"}
    X["mode_risk_flag"] = X["Shipping Mode"].apply(
        lambda x: 0 if x in express_modes else 1
    )

    # ── 3. Regional Congestion Indicator (spec-required) ─────────────────────
    # Target-encoded historical late-delivery rate per region
    # Higher value = region historically more congested / delayed
    X["regional_congestion_idx"] = X["Order Region"].map(region_risk_map).fillna(global_mean)

    # ── 4. Order Complexity Score (spec-required) ─────────────────────────────
    # Composite score: high quantity + high discount + long schedule = complex order
    # Normalised to [0, 1] range using min-max of typical values
    qty_norm   = X["Order Item Quantity"] / 5.0          # max qty = 5
    disc_norm  = X["Order Item Discount Rate"]            # already 0-1
    sched_norm = X["Days for shipment (scheduled)"] / 4.0  # max sched = 4
    X["order_complexity_score"] = (qty_norm + disc_norm + sched_norm) / 3.0

    # ── 5. Order Value Per Unit ───────────────────────────────────────────────
    X["order_value_per_unit"] = X["Sales"] / (X["Order Item Quantity"] + 1)

    # ── 6. Profit Margin ──────────────────────────────────────────────────────
    X["profit_margin"] = X["Order Profit Per Order"] / (X["Sales"] + 1)

    # ── 7. High Discount Flag ────────────────────────────────────────────────
    X["high_discount_flag"] = (X["Order Item Discount Rate"] > 0.10).astype(int)

    # ── 8. Benefit Per Unit ──────────────────────────────────────────────────
    X["benefit_per_unit"] = X["Benefit per order"] / (X["Order Item Quantity"] + 1)

    return X

X_train_fe = add_features(X_train_raw, region_risk_map, global_mean)
X_test_fe  = add_features(X_test_raw,  region_risk_map, global_mean)

new_features = [
    "shipping_pressure_idx",    # spec-required
    "mode_risk_flag",           # spec-required
    "regional_congestion_idx",  # spec-required
    "order_complexity_score",   # spec-required
    "order_value_per_unit",
    "profit_margin",
    "high_discount_flag",
    "benefit_per_unit",
]
log(f"\n     Engineered features added ({len(new_features)}):")
for f in new_features:
    log(f"       {f}")
log(f"\n     X_train shape after FE: {X_train_fe.shape}")
log(f"     X_test  shape after FE: {X_test_fe.shape}")

# Save synchronized clean datasets for Group 3 and downstream reproducibility
os.makedirs("data", exist_ok=True)
X_train_fe.to_csv("data/X_train.csv", index=False)
X_test_fe.to_csv("data/X_test.csv",   index=False)
y_train.to_csv("data/y_train.csv",    index=False)
y_test.to_csv("data/y_test.csv",      index=False)
log("     Saved clean train/test data to data/ (aligned with best_pipeline.pkl)")

log("\n[PART 6 COMPLETE]")

# ==============================================================================
# PART 7 -- ENCODING & PIPELINE
# ==============================================================================
sep("PART 7 -- ENCODING & PIPELINE")

log("\n[7.1] Identifying column types after feature engineering:")

# Identify categorical and numerical columns
cat_features = X_train_fe.select_dtypes(include="object").columns.tolist()
num_features = X_train_fe.select_dtypes(include="number").columns.tolist()

log(f"     Categorical ({len(cat_features)}): {cat_features}")
log(f"     Numerical   ({len(num_features)}): {num_features}")

# ── 7.2 Build ColumnTransformer ──────────────────────────────────────────────
log("\n[7.2] Building ColumnTransformer pipeline:")
log("""
     Numerical path:
       SimpleImputer(strategy='median')
         --> StandardScaler()

     Categorical path:
       SimpleImputer(strategy='most_frequent')
         --> OneHotEncoder(handle_unknown='ignore', sparse_output=False)

     Both pipelines fit ONLY on training data.
     The combined preprocessor handles any future unseen categories gracefully.
""")

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features),
    ],
    remainder="drop"
)

log("[7.3] Fitting preprocessor on X_train ONLY...")
preprocessor.fit(X_train_fe)

# Get transformed feature names for explainability
num_feature_names = num_features
ohe_feature_names = (
    preprocessor
    .named_transformers_["cat"]
    .named_steps["onehot"]
    .get_feature_names_out(cat_features)
    .tolist()
)
all_feature_names = num_feature_names + ohe_feature_names
log(f"     Total features after encoding: {len(all_feature_names)}")

joblib.dump(preprocessor, "models/preprocessor.pkl")
log("     Saved: models/preprocessor.pkl")

# ── 7.4 Save feature metadata ────────────────────────────────────────────────
feature_meta = {
    "cat_features": cat_features,
    "num_features": num_features,
    "all_feature_names": all_feature_names,
    "engineered_features": new_features,
    "region_risk_map_path": "models/region_risk_map.pkl",
}
with open("models/feature_meta.json", "w") as f:
    json.dump(feature_meta, f, indent=2)
log("     Saved: models/feature_meta.json")
log("\n[PART 7 COMPLETE]")

# ==============================================================================
# SHARED EVALUATION UTILITY
# ==============================================================================
def evaluate(name, pipeline, X_train, X_test, y_train, y_test):
    """Fit pipeline and return full metrics dict + per-class report."""
    log(f"\n[Training] {name}...")
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    auc  = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                    target_names=["On-Time","Late"],
                                    zero_division=0)

    log(f"  ROC-AUC   : {auc:.4f}")
    log(f"  Precision : {prec:.4f}")
    log(f"  Recall    : {rec:.4f}")
    log(f"  F1 Score  : {f1:.4f}")
    log(f"\n{report}")

    return {
        "ROC-AUC":   round(auc,  4),
        "Precision": round(prec, 4),
        "Recall":    round(rec,  4),
        "F1 Score":  round(f1,   4),
        "confusion_matrix": cm.tolist(),
        "pipeline": pipeline,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }


def cv_score(name, pipeline, X_raw, y, cv=5):
    """
    Stratified k-fold CV with per-fold regional target encoding.

    X_raw must be the PRE-feature-engineering DataFrame (X_train_raw, not
    X_train_fe). The regional_congestion_idx is recomputed inside each fold
    using only that fold's training labels, preventing target-encoding
    leakage into the validation split.
    """
    log(f"\n[Cross-Validation] {name} (StratifiedKFold n={cv}, scoring=roc_auc)...")
    log(f"  Regional target encoding recomputed per fold (leak-free).")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
        X_tr  = X_raw.iloc[train_idx].reset_index(drop=True)
        X_val = X_raw.iloc[val_idx].reset_index(drop=True)
        y_tr  = y.iloc[train_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)

        # Compute region risk map from THIS FOLD's training labels ONLY
        fold_region_map = (
            X_tr.assign(_t=y_tr.values)
            .groupby("Order Region")["_t"]
            .mean()
            .to_dict()
        )
        fold_global_mean = float(y_tr.mean())

        # Engineer features using the fold-specific encoding
        X_tr_fe  = add_features(X_tr,  fold_region_map, fold_global_mean)
        X_val_fe = add_features(X_val, fold_region_map, fold_global_mean)

        # Clone to ensure a fresh, unfitted pipeline for each fold
        fold_pipe = clone(pipeline)
        fold_pipe.fit(X_tr_fe, y_tr)

        y_proba   = fold_pipe.predict_proba(X_val_fe)[:, 1]
        fold_auc  = roc_auc_score(y_val, y_proba)
        scores.append(fold_auc)
        log(f"    Fold {fold_i + 1}: AUC = {fold_auc:.4f}")

    scores = np.array(scores)
    log(f"  CV AUC: {scores.mean():.4f} +/- {scores.std():.4f}  | folds: {scores.round(4).tolist()}")
    return {"mean_auc": round(float(scores.mean()), 4),
            "std_auc":  round(float(scores.std()),  4)}


# ==============================================================================
# PART 8 -- LOGISTIC REGRESSION  (Baseline)
# ==============================================================================
sep("PART 8 -- LOGISTIC REGRESSION (Baseline)")

lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])

lr_cv      = cv_score("Logistic Regression", lr_pipeline, X_train_raw, y_train)
lr_results = evaluate("Logistic Regression", lr_pipeline, X_train_fe, X_test_fe, y_train, y_test)

joblib.dump(lr_pipeline, "models/lr_pipeline.pkl")
log("Saved: models/lr_pipeline.pkl")
log("\n[PART 8 COMPLETE]")

# ==============================================================================
# PART 9 -- RANDOM FOREST
# ==============================================================================
sep("PART 9 -- RANDOM FOREST")

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )),
])

rf_cv      = cv_score("Random Forest", rf_pipeline, X_train_raw, y_train)
rf_results = evaluate("Random Forest", rf_pipeline, X_train_fe, X_test_fe, y_train, y_test)

joblib.dump(rf_pipeline, "models/rf_pipeline.pkl")
log("Saved: models/rf_pipeline.pkl")
log("\n[PART 9 COMPLETE]")

# ==============================================================================
# PART 10 -- XGBOOST
# ==============================================================================
sep("PART 10 -- XGBOOST")

scale_pos = float(y_train.value_counts()[0] / y_train.value_counts()[1])

xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )),
])

xgb_cv      = cv_score("XGBoost", xgb_pipeline, X_train_raw, y_train)
xgb_results = evaluate("XGBoost", xgb_pipeline, X_train_fe, X_test_fe, y_train, y_test)

joblib.dump(xgb_pipeline, "models/xgb_pipeline.pkl")
log("Saved: models/xgb_pipeline.pkl")
log("\n[PART 10 COMPLETE]")

# ==============================================================================
# COMPARISON TABLE & BEST MODEL SELECTION
# ==============================================================================
sep("MODEL COMPARISON")

all_results = {
    "Logistic Regression": {"metrics": lr_results, "cv": lr_cv},
    "Random Forest":       {"metrics": rf_results, "cv": rf_cv},
    "XGBoost":             {"metrics": xgb_results,"cv": xgb_cv},
}

log(f"\n{'Model':<22} {'AUC':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'CV AUC':>10}")
log("-" * 72)
for name, data in all_results.items():
    m = data["metrics"]
    c = data["cv"]
    log(f"{name:<22} {m['ROC-AUC']:>8.4f} {m['Precision']:>10.4f} {m['Recall']:>8.4f} {m['F1 Score']:>8.4f} {c['mean_auc']:>10.4f}")

best_name = max(all_results, key=lambda n: all_results[n]["metrics"]["ROC-AUC"])
best_pipeline = all_results[best_name]["metrics"]["pipeline"]
log(f"\nBest model: {best_name}  (AUC = {all_results[best_name]['metrics']['ROC-AUC']:.4f})")

joblib.dump(best_pipeline, "models/best_pipeline.pkl")
log("Saved: models/best_pipeline.pkl")

# Also save native XGBoost JSON representation for modern XGBoost standard
try:
    if "model" in best_pipeline.named_steps and hasattr(best_pipeline.named_steps["model"], "get_booster"):
        best_pipeline.named_steps["model"].get_booster().save_model("models/xgboost_booster.json")
        log("Saved: models/xgboost_booster.json (native XGBoost format)")
except Exception as e:
    pass

# ==============================================================================
# FIGURES
# ==============================================================================
sep("GENERATING FIGURES")

# ── Confusion matrices ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
palette = {"Logistic Regression": "Blues", "Random Forest": "Greens", "XGBoost": "Oranges"}
for ax, (name, data) in zip(axes, all_results.items()):
    cm  = np.array(data["metrics"]["confusion_matrix"])
    auc = data["metrics"]["ROC-AUC"]
    sns.heatmap(cm, annot=True, fmt="d", cmap=palette[name], ax=ax,
                xticklabels=["On-Time","Late"], yticklabels=["On-Time","Late"],
                linewidths=1, linecolor="white")
    ax.set_title(f"{name}\nAUC = {auc:.4f}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
plt.suptitle("Confusion Matrices -- All Models", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("figures/g2_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
log("Saved: figures/g2_confusion_matrices.png")

# ── ROC Curves ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
colors  = {"Logistic Regression": "#3b82f6", "Random Forest": "#22c55e", "XGBoost": "#f59e0b"}
for name, data in all_results.items():
    y_proba = data["metrics"]["y_proba"]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = data["metrics"]["ROC-AUC"]
    ax.plot(fpr, tpr, lw=2.5, color=colors[name], label=f"{name}  (AUC = {auc:.4f})")
ax.plot([0,1],[0,1],"k--", lw=1, alpha=0.5, label="Random Classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate",  fontsize=12)
ax.set_title("ROC Curves -- Model Comparison", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=11)
ax.set_facecolor("#f8f9fa")
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("figures/g2_roc_curves.png", dpi=150)
plt.close()
log("Saved: figures/g2_roc_curves.png")

# ── Model Comparison Bar Chart ────────────────────────────────────────────────
metrics_to_plot = ["ROC-AUC", "Precision", "Recall", "F1 Score"]
model_names = list(all_results.keys())
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
bar_colors = list(colors.values())
for ax, metric in zip(axes, metrics_to_plot):
    vals = [all_results[n]["metrics"][metric] for n in model_names]
    bars = ax.bar(model_names, vals, color=bar_colors, edgecolor="white", linewidth=1.5)
    ax.set_ylim(0, 1.05)
    ax.set_title(metric, fontsize=12, fontweight="bold")
    ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
plt.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/g2_model_comparison.png", dpi=150)
plt.close()
log("Saved: figures/g2_model_comparison.png")

# ── Feature Importance (RF + XGB) ────────────────────────────────────────────
for name, pipe_key in [("Random Forest", rf_pipeline), ("XGBoost", xgb_pipeline)]:
    model = pipe_key.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        # Only show top 20
        fi = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10, 7))
        gradient = plt.cm.RdYlGn_r(np.linspace(0.05, 0.95, len(fi)))
        ax.barh(fi.index[::-1], fi.values[::-1], color=gradient[::-1])
        ax.set_xlabel("Feature Importance", fontsize=12)
        ax.set_title(f"Top 20 Feature Importances -- {name}", fontsize=14, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        fname = f"figures/g2_feature_importance_{name.lower().replace(' ','_')}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        log(f"Saved: {fname}")
        log(f"\nTop 10 features ({name}):")
        for feat, imp in fi.head(10).items():
            log(f"  {feat:<45} {imp:.4f}")

# ==============================================================================
# SAVE METRICS SUMMARY
# ==============================================================================
sep("SAVING METRICS")

metrics_summary = {
    "best_model": best_name,
    "test_metrics": {
        name: {k: v for k, v in data["metrics"].items()
               if k not in ("pipeline", "y_proba", "y_pred")}
        for name, data in all_results.items()
    },
    "cv_results": {
        name: data["cv"] for name, data in all_results.items()
    },
    "feature_names": all_feature_names,
    "num_features": num_features,
    "cat_features": cat_features,
    "engineered_features": new_features,
    "target_col": TARGET,
    "n_train": int(len(X_train_fe)),
    "n_test":  int(len(X_test_fe)),
}
with open("models/metrics_summary.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)
log("Saved: models/metrics_summary.json")

# ── Save full report ──────────────────────────────────────────────────────────
with open("reports/group2_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
log("Saved: reports/group2_report.txt")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
sep("GROUP 2 SUMMARY")
log(f"""
  Feature Engineering:
    1. shipping_pressure_idx      (spec-required) -- schedule / quantity
    2. mode_risk_flag             (spec-required) -- express vs economy
    3. regional_congestion_idx    (spec-required) -- target-encoded region rate
    4. order_complexity_score     (spec-required) -- composite qty+disc+sched
    5. order_value_per_unit       -- sales / quantity
    6. profit_margin              -- profit / sales
    7. high_discount_flag         -- discount rate > 10%
    8. benefit_per_unit           -- benefit / quantity

  Pipeline Architecture:
    ColumnTransformer:
      Numeric  -> SimpleImputer(median) -> StandardScaler
      Categoric-> SimpleImputer(mode)   -> OneHotEncoder(handle_unknown=ignore)
    Full pipeline per model: preprocessor + model (single .pkl file)

  Results:
    {'Model':<22} {'AUC':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}""")
for name, data in all_results.items():
    m = data["metrics"]
    log(f"    {name:<22} {m['ROC-AUC']:>8.4f} {m['Precision']:>10.4f} {m['Recall']:>8.4f} {m['F1 Score']:>8.4f}")
log(f"""
  Best Model  : {best_name}
  Saved to    : models/best_pipeline.pkl  (full pipeline -- use directly in Streamlit)

  NOTE: The Streamlit app.py should now call:
        best_pipeline.predict_proba(X_fe)[:, 1]
        where X_fe = add_features(order_df, region_risk_map, global_mean)
        No separate scaler/encoder calls needed.
""")
sep()
log("GROUP 2 COMPLETE -- Ready for Group 3 (Evaluation & Dashboard)")
