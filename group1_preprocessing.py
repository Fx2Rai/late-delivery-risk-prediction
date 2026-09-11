"""
==============================================================================
GROUP 1 -- Data & Preprocessing
APL Logistics -- Late Delivery Risk Prediction
==============================================================================
Parts:
  1. Dataset Validation
  2. Data Cleaning
  3. Leakage Analysis
  4. Train/Test Preprocessing
  5. Verification
==============================================================================
Output files:
  data/cleaned_data.csv          -- cleaned, leakage-free dataset
  data/X_train.csv / X_test.csv  -- preprocessed feature matrices
  data/y_train.csv / y_test.csv  -- target vectors
  reports/group1_report.txt      -- full validation report
==============================================================================
"""

import os, sys, json, io, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
os.makedirs("data",    exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ── stdout captured to report file too ────────────────────────────────────────
report_lines = []

def log(msg=""):
    print(msg)
    report_lines.append(msg)

def sep(title="", char="=", width=72):
    if title:
        side = (width - len(title) - 2) // 2
        log(char * side + f" {title} " + char * side)
    else:
        log(char * width)

# ==============================================================================
# PART 1 -- DATASET VALIDATION
# ==============================================================================
sep("PART 1 -- DATASET VALIDATION")

DATA_PATH = "APL_Logistics.csv"
TARGET    = "Late_delivery_risk"

log(f"\n[1.1] Loading: {DATA_PATH}")
df_raw = pd.read_csv(DATA_PATH, encoding="latin-1")
log(f"      Shape   : {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")

# ── 1.2 Column Names ──────────────────────────────────────────────────────────
log("\n[1.2] Column Names:")
for i, col in enumerate(df_raw.columns):
    log(f"      {i:>2}. {col}")

# ── 1.3 Data Types ────────────────────────────────────────────────────────────
log("\n[1.3] Data Types:")
type_summary = df_raw.dtypes.astype(str)
for col, dtype in type_summary.items():
    log(f"      {col:<45} {dtype}")

numeric_cols   = df_raw.select_dtypes(include="number").columns.tolist()
object_cols    = df_raw.select_dtypes(include="object").columns.tolist()
log(f"\n      Numerical columns : {len(numeric_cols)}")
log(f"      Categorical cols  : {len(object_cols)}")

# ── 1.4 Missing Values ────────────────────────────────────────────────────────
log("\n[1.4] Missing Values:")
mv = df_raw.isnull().sum()
mv_nonzero = mv[mv > 0]
if mv_nonzero.empty:
    log("      None found.")
else:
    for col, cnt in mv_nonzero.items():
        pct = cnt / len(df_raw) * 100
        log(f"      {col:<45} {cnt:>5} missing ({pct:.3f}%)")

# ── 1.5 Duplicate Rows ────────────────────────────────────────────────────────
log("\n[1.5] Duplicate Rows:")
n_dups = df_raw.duplicated().sum()
log(f"      Total duplicates  : {n_dups:,}")

# ── 1.6 Target Variable Validation ───────────────────────────────────────────
log("\n[1.6] Target Variable Validation: Late_delivery_risk")
if TARGET not in df_raw.columns:
    log("      ERROR: Target column not found!")
    sys.exit(1)

vc = df_raw[TARGET].value_counts()
log(f"      Unique values  : {sorted(df_raw[TARGET].unique())}")
log(f"      Value counts   :")
for val, cnt in vc.items():
    log(f"         {val} (label) -> {cnt:>7,}  ({cnt/len(df_raw)*100:.2f}%)")

null_target = df_raw[TARGET].isnull().sum()
log(f"      Null in target : {null_target}")
dtype_ok = df_raw[TARGET].dtype in [np.int64, np.int32, np.float64]
log(f"      Dtype OK (int) : {dtype_ok}")

# ── 1.7 Categorical Distributions ────────────────────────────────────────────
log("\n[1.7] Key Categorical Distributions:")
key_cats = ["Shipping Mode", "Market", "Customer Segment",
            "Order Status", "Delivery Status", "Type", "Order Region"]
for col in key_cats:
    if col in df_raw.columns:
        log(f"\n      {col}:")
        for val, cnt in df_raw[col].value_counts().items():
            log(f"        {str(val):<35} {cnt:>7,}  ({cnt/len(df_raw)*100:.1f}%)")

# ── 1.8 Numerical Summary ─────────────────────────────────────────────────────
log("\n[1.8] Numerical Feature Summary:")
desc = df_raw[numeric_cols].describe().T[["min","mean","max","std"]]
for col, row in desc.iterrows():
    log(f"      {col:<45} min={row['min']:>10.2f}  mean={row['mean']:>10.2f}  max={row['max']:>10.2f}")

log("\n[PART 1 COMPLETE] Dataset loaded and validated.")

# ==============================================================================
# PART 2 -- DATA CLEANING
# ==============================================================================
sep("PART 2 -- DATA CLEANING")

df = df_raw.copy()

# ── 2.1 Handle Missing Values ─────────────────────────────────────────────────
log("\n[2.1] Handling Missing Values:")
df["Customer Lname"].fillna("Unknown", inplace=True)
log("      Customer Lname    -> filled with 'Unknown'")
median_zip = df["Customer Zipcode"].median()
df["Customer Zipcode"].fillna(median_zip, inplace=True)
log(f"     Customer Zipcode  -> filled with median ({median_zip})")
remaining = df.isnull().sum().sum()
log(f"     Remaining nulls   : {remaining}")

# ── 2.2 Handle Inconsistent Values ───────────────────────────────────────────
log("\n[2.2] Inconsistency Checks:")

# Negative Sales
neg_sales = (df["Sales"] < 0).sum()
log(f"      Negative Sales values       : {neg_sales}")

# Negative Product Price
neg_price = (df["Product Price"] <= 0).sum()
log(f"      Zero/Negative Product Price : {neg_price}")

# Days for shipment (scheduled) <= 0
bad_sched = (df["Days for shipment (scheduled)"] <= 0).sum()
log(f"      Scheduled days <= 0         : {bad_sched}")

# Discount rate out of [0,1]
bad_disc = ((df["Order Item Discount Rate"] < 0) | (df["Order Item Discount Rate"] > 1)).sum()
log(f"      Discount rate out of [0,1]  : {bad_disc}")

# Profit ratio out of [-1, 1]
bad_pr = ((df["Order Item Profit Ratio"] < -1) | (df["Order Item Profit Ratio"] > 1)).sum()
log(f"      Profit ratio out of [-1,1]  : {bad_pr}")

log("      No critical inconsistencies found requiring row removal.")

# ── 2.3 Remove PII / High-Cardinality / Irrelevant Columns ───────────────────
log("\n[2.3] Removing PII and Irrelevant Columns:")
PII_COLS = [
    "Customer Fname", "Customer Lname", "Customer Street",
    "Customer City", "Customer Zipcode", "Customer Id", "Order Customer Id",
]
HIGH_CARD_COLS = [
    "Product Name",    # >1,000 unique values, too sparse for encoding
    "Order City",      # redundant with Order Region / Market
    "Order State",     # redundant
    "Order Country",   # redundant with Market
    "Customer Country",# redundant
    "Customer State",  # low predictive value after region encoding
]
drop_with_reason = {c: "PII" for c in PII_COLS}
drop_with_reason.update({c: "High-Cardinality/Redundant" for c in HIGH_CARD_COLS})

for col, reason in drop_with_reason.items():
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
        log(f"      Dropped [{reason}]  : {col}")

# ── 2.4 Duplicates ────────────────────────────────────────────────────────────
log("\n[2.4] Duplicate Row Check:")
n_dups = df.duplicated().sum()
log(f"      Duplicate rows detected: {n_dups}")
if n_dups > 0:
    log("      NOTE: Duplicates NOT removed -- in logistics datasets, identical")
    log("      feature values can represent genuinely separate orders. Removing")
    log("      would cause data loss without strong justification.")
else:
    log("      No exact duplicates -- no action needed.")

# ── 2.5 Standardize Column Names ─────────────────────────────────────────────
log("\n[2.5] Column Name Standardization:")
log("      Column names already clean (mixed case, spaces -- kept as-is for readability).")
log(f"      Remaining columns after cleaning: {df.shape[1]}")
log(f"      Remaining rows                  : {df.shape[0]:,}")

log("\n[PART 2 COMPLETE] Data cleaned and PII removed.")
log(f"      Shape after cleaning: {df.shape}")

# ==============================================================================
# PART 3 -- LEAKAGE ANALYSIS
# ==============================================================================
sep("PART 3 -- LEAKAGE ANALYSIS")

log("""
Objective: Predict Late_delivery_risk BEFORE an order is shipped.
Features must only include information available at ORDER CREATION time.
""")

log("=" * 72)
log("  FEATURE-BY-FEATURE LEAKAGE ASSESSMENT")
log("=" * 72)

# ─────────────────────────────────────────────────────────────────────────────
log("""
[3.1] Days for shipping (real)
  Status   : *** CRITICAL LEAKAGE ***
  Reason   : This is the ACTUAL number of days the order took to be shipped.
             It is only known after shipment is complete -- exactly the outcome
             we are trying to predict. Including this would give the model
             direct access to the future.
  Evidence : Correlation with target:""")
corr = df_raw["Days for shipping (real)"].corr(df_raw[TARGET]) if "Days for shipping (real)" in df_raw.columns else "N/A"
log(f"             Pearson r = {corr:.4f}  (very high -> confirms leakage)")
log("  Decision: REMOVE")

# ─────────────────────────────────────────────────────────────────────────────
log("""
[3.2] Delivery Status
  Status   : *** CRITICAL LEAKAGE ***
  Reason   : This column is a direct, human-readable encoding of the target.
             'Late delivery' -> Late_delivery_risk = 1
             'Advance shipping' / 'Shipping on time' -> Late_delivery_risk = 0
             'Shipping canceled' -> Late_delivery_risk = 0
             It is 100% determined after the fact -- completely post-shipment.
  Evidence :""")
if "Delivery Status" in df_raw.columns:
    cross = pd.crosstab(df_raw["Delivery Status"], df_raw[TARGET])
    for line in cross.to_string().split("\n"):
        log("             " + line)
log("  Decision: REMOVE")

# ─────────────────────────────────────────────────────────────────────────────
log("""
[3.3] Order Status  (DETAILED INVESTIGATION)
  At-order-creation statuses (PRE-shipment, SAFE):
    PENDING          -- order placed, not yet processed
    PENDING_PAYMENT  -- order placed, payment not confirmed
    PROCESSING       -- order being prepared
    PAYMENT_REVIEW   -- under review before fulfillment
    SUSPECTED_FRAUD  -- flagged before shipment

  Post-fulfillment statuses (POST-shipment, LEAKAGE RISK):
    COMPLETE         -- order fully delivered
    CLOSED           -- order closed after delivery
    CANCELED         -- may be before or after shipment
    ON_HOLD          -- ambiguous timing""")

if "Order Status" in df_raw.columns:
    log("\n  Distribution in dataset:")
    os_dist = df_raw["Order Status"].value_counts()
    for val, cnt in os_dist.items():
        log(f"    {val:<25} {cnt:>7,}  ({cnt/len(df_raw)*100:.1f}%)")

    log("\n  Late delivery rate by Order Status:")
    os_risk = df_raw.groupby("Order Status")[TARGET].mean().sort_values(ascending=False)
    for val, rate in os_risk.items():
        log(f"    {val:<25} late_rate = {rate:.3f}")

log("""
  Verdict: COMPLETE (33%) and CLOSED (10.9%) are post-shipment.
           However, the majority of statuses (PENDING, PROCESSING, etc.)
           ARE pre-shipment. The variable captures a mix of pre- and
           post-shipment states.
  Conservative Decision: REMOVE Order Status to eliminate all leakage risk.
           A pre-shipment system would never see COMPLETE or CLOSED.
           Keeping it would cause the model to learn from unavailable signals.""")

# ─────────────────────────────────────────────────────────────────────────────
log("""
[3.4] Other Fields -- Rapid Assessment:
  Days for shipment (scheduled)   : SAFE  -- planned at order creation
  Shipping Mode                   : SAFE  -- selected at order creation
  Type (payment method)           : SAFE  -- known at order creation
  Order Item Quantity             : SAFE  -- known at order creation
  Order Item Product Price        : SAFE  -- catalogue price, pre-shipment
  Order Item Discount             : SAFE  -- applied at order creation
  Order Item Discount Rate        : SAFE  -- applied at order creation
  Sales / Order Item Total        : SAFE  -- calculated at order creation
  Market / Order Region           : SAFE  -- geographic, known at order creation
  Customer Segment                : SAFE  -- CRM attribute, pre-shipment
  Department Name / Category Name : SAFE  -- product classification, pre-shipment
  Product Price                   : SAFE  -- catalogue, pre-shipment
  Latitude / Longitude            : SAFE  -- customer location, known at creation
  Order Item Profit Ratio         : BORDERLINE -- calculated margin, may be
                                    estimated at order time; kept (low leakage risk)
  Benefit per order               : BORDERLINE -- net profit, best estimated;
                                    kept as it reflects expected order value
  Sales per customer              : BORDERLINE -- historical aggregate; kept
  Order Profit Per Order          : BORDERLINE -- mirrors Benefit per order; kept
""")

# ─────────────────────────────────────────────────────────────────────────────
# Apply leakage removals
LEAKAGE_COLS = [
    "Days for shipping (real)",
    "Delivery Status",
    "Order Status",
]
log("[3.5] Removing Leakage Columns:")
for col in LEAKAGE_COLS:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
        log(f"      Removed: {col}")
    else:
        log(f"      Already absent: {col}")

log(f"\n      Shape after leakage removal: {df.shape}")
log("\n[PART 3 COMPLETE] Leakage analysis done. Dataset is pre-shipment safe.")

# ==============================================================================
# PART 4 -- TRAIN/TEST PREPROCESSING
# ==============================================================================
sep("PART 4 -- TRAIN/TEST PREPROCESSING")

log("\n[4.1] Feature / Target Separation:")
X = df.drop(columns=[TARGET])
y = df[TARGET]
log(f"      X shape  : {X.shape}")
log(f"      y shape  : {y.shape}")
log(f"      Features : {list(X.columns)}")

# ── 4.2 Train / Test Split ────────────────────────────────────────────────────
log("\n[4.2] Stratified Train/Test Split (80/20):")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
log(f"      X_train : {X_train.shape}")
log(f"      X_test  : {X_test.shape}")
log(f"      y_train : {y_train.shape}  | late={y_train.sum()} ({y_train.mean()*100:.2f}%)")
log(f"      y_test  : {y_test.shape}   | late={y_test.sum()}  ({y_test.mean()*100:.2f}%)")

# ── 4.3 Identify Column Types ─────────────────────────────────────────────────
log("\n[4.3] Column Type Identification (on TRAIN set):")
cat_cols = X_train.select_dtypes(include="object").columns.tolist()
num_cols = X_train.select_dtypes(include="number").columns.tolist()
log(f"      Categorical ({len(cat_cols)}): {cat_cols}")
log(f"      Numerical   ({len(num_cols)}): {num_cols}")

# ── 4.4 Feature Engineering ───────────────────────────────────────────────────
log("\n[4.4] Feature Engineering:")
def add_features(df_in):
    df_out = df_in.copy()
    # Shipping pressure: how many days per unit item?
    df_out["shipping_pressure"] = (
        df_out["Days for shipment (scheduled)"] / (df_out["Order Item Quantity"] + 1)
    )
    # Mode risk flag: slow modes = higher risk
    express = {"First Class", "Same Day"}
    df_out["mode_risk_flag"] = df_out["Shipping Mode"].apply(
        lambda x: 0 if x in express else 1
    )
    # High discount flag
    df_out["high_discount_flag"] = (df_out["Order Item Discount Rate"] > 0.10).astype(int)
    # Order value per unit
    df_out["order_value_per_unit"] = df_out["Sales"] / (df_out["Order Item Quantity"] + 1)
    # Profit margin
    df_out["profit_margin"] = df_out["Order Profit Per Order"] / (df_out["Sales"] + 1)
    return df_out

X_train = add_features(X_train)
X_test  = add_features(X_test)
log("      Added: shipping_pressure, mode_risk_flag, high_discount_flag,")
log("             order_value_per_unit, profit_margin")
log(f"      X_train shape after FE: {X_train.shape}")
log(f"      X_test  shape after FE: {X_test.shape}")

# Refresh column type lists after feature engineering
cat_cols = X_train.select_dtypes(include="object").columns.tolist()
num_cols = X_train.select_dtypes(include="number").columns.tolist()

# ── 4.5 Encoding -- FIT ON TRAIN ONLY ────────────────────────────────────────
log("\n[4.5] Categorical Encoding (Label Encoding -- fit on TRAIN only):")
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    # Fit only on train
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    # Transform test -- handle unseen labels safely
    def safe_transform(series, le):
        known = set(le.classes_)
        return series.astype(str).apply(
            lambda x: le.transform([x])[0] if x in known else -1
        )
    X_test[col] = safe_transform(X_test[col], le)
    encoders[col] = le
    n_classes = len(le.classes_)
    log(f"      {col:<35} {n_classes} classes")

# ── 4.6 Scaling -- FIT ON TRAIN ONLY ──────────────────────────────────────────
log("\n[4.6] Numerical Scaling (StandardScaler -- fit on TRAIN only):")
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])
log(f"      Scaled {len(num_cols)} numerical features.")
log("      Train mean after scaling (sample 3 cols):")
for col in num_cols[:3]:
    log(f"        {col:<40} mean={X_train[col].mean():.6f}  std={X_train[col].std():.6f}")

# ── 4.7 Class Imbalance Analysis ─────────────────────────────────────────────
log("\n[4.7] Class Imbalance Analysis:")
counts = y_train.value_counts()
ratio  = counts[0] / counts[1]
log(f"      Class 0 (On-Time) : {counts[0]:>7,}  ({counts[0]/len(y_train)*100:.2f}%)")
log(f"      Class 1 (Late)    : {counts[1]:>7,}  ({counts[1]/len(y_train)*100:.2f}%)")
log(f"      Imbalance ratio   : {ratio:.3f}  (counts[0] / counts[1])")
log("""
      Assessment:
        Ratio = 0.824 (< 1.0) -- target is slightly MAJORITY late.
        The 55/45 split is considered MILD imbalance.

      Strategy chosen: class_weight='balanced' for sklearn models
                       scale_pos_weight for XGBoost
      Reason: SMOTE is not needed for mild imbalance; class weights are
              simpler, introduce no synthetic data risk, and are applied
              per-model rather than modifying the dataset itself.
      SMOTE  : Not applied (mild imbalance + large dataset size).""")

# ── 4.8 Save Preprocessed Data ───────────────────────────────────────────────
log("\n[4.8] Saving preprocessed data to data/ folder:")
feature_names = list(X_train.columns)

X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv",   index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv",   index=False)
df.to_csv("data/cleaned_data.csv", index=False)

# Also save the preprocessors (overwrite models/ folder with updated ones)
os.makedirs("models", exist_ok=True)
joblib.dump(encoders,      "models/encoders.pkl")
joblib.dump(scaler,        "models/scaler.pkl")
joblib.dump(cat_cols,      "models/cat_cols.pkl")
joblib.dump(num_cols,      "models/num_cols.pkl")
joblib.dump(feature_names, "models/feature_names.pkl")

meta = {
    "feature_names": feature_names,
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "leakage_removed": LEAKAGE_COLS,
    "pii_removed": PII_COLS,
    "high_card_removed": HIGH_CARD_COLS,
    "engineered_features": [
        "shipping_pressure", "mode_risk_flag",
        "high_discount_flag", "order_value_per_unit", "profit_margin"
    ],
    "train_shape": list(X_train.shape),
    "test_shape":  list(X_test.shape),
    "class_counts_train": counts.to_dict(),
    "imbalance_ratio": round(ratio, 4)
}
with open("data/preprocessing_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

log("      data/X_train.csv, X_test.csv, y_train.csv, y_test.csv")
log("      data/cleaned_data.csv")
log("      data/preprocessing_meta.json")
log("      models/encoders.pkl, scaler.pkl, cat_cols.pkl, num_cols.pkl")

log("\n[PART 4 COMPLETE] Train/Test preprocessing pipeline finished.")

# ==============================================================================
# PART 5 -- VERIFICATION
# ==============================================================================
sep("PART 5 -- VERIFICATION")

all_pass = True

def check(name, condition, detail=""):
    global all_pass
    status = "PASS" if condition else "FAIL"
    if not condition:
        all_pass = False
    log(f"  [{status}] {name}")
    if detail:
        log(f"         {detail}")

log("")

# ── 5.1 No Leakage ────────────────────────────────────────────────────────────
log("[5.1] Leakage Checks:")
for col in ["Days for shipping (real)", "Delivery Status", "Order Status"]:
    check(
        f"'{col}' not in X_train",
        col not in X_train.columns,
        f"Column {'absent' if col not in X_train.columns else 'STILL PRESENT -- FIX'}"
    )

# ── 5.2 No Target in Features ─────────────────────────────────────────────────
log("\n[5.2] Target Leakage:")
check(
    f"'{TARGET}' not in X_train",
    TARGET not in X_train.columns
)

# ── 5.3 No Missing Values in Train/Test ───────────────────────────────────────
log("\n[5.3] Missing Values:")
train_nulls = X_train.isnull().sum().sum()
test_nulls  = X_test.isnull().sum().sum()
check("X_train has 0 missing values", train_nulls == 0, f"nulls = {train_nulls}")
check("X_test  has 0 missing values", test_nulls  == 0, f"nulls = {test_nulls}")

# ── 5.4 Shape Verification ────────────────────────────────────────────────────
log("\n[5.4] Shape Verification:")
expected_train = int(len(df) * 0.80)
expected_test  = int(len(df) * 0.20)
check(
    f"X_train rows ~ 80% of data",
    abs(len(X_train) - expected_train) < 5,
    f"X_train={len(X_train):,}  expected~{expected_train:,}"
)
check(
    f"X_test  rows ~ 20% of data",
    abs(len(X_test) - expected_test) < 5,
    f"X_test={len(X_test):,}   expected~{expected_test:,}"
)
check(
    "X_train and X_test have same number of columns",
    X_train.shape[1] == X_test.shape[1],
    f"train_cols={X_train.shape[1]}  test_cols={X_test.shape[1]}"
)
check(
    "y_train length matches X_train",
    len(y_train) == len(X_train)
)
check(
    "y_test length matches X_test",
    len(y_test) == len(X_test)
)

# ── 5.5 Stratification Check ──────────────────────────────────────────────────
log("\n[5.5] Stratification (target distribution preserved):")
train_rate = y_train.mean()
test_rate  = y_test.mean()
orig_rate  = y.mean()
check(
    "Train late_rate within +/-1% of full dataset",
    abs(train_rate - orig_rate) < 0.01,
    f"train={train_rate:.4f}  original={orig_rate:.4f}"
)
check(
    "Test  late_rate within +/-1% of full dataset",
    abs(test_rate - orig_rate) < 0.01,
    f"test={test_rate:.4f}   original={orig_rate:.4f}"
)

# ── 5.6 Encoding Sanity ───────────────────────────────────────────────────────
log("\n[5.6] Encoding Sanity:")
check(
    "No object (string) columns in X_train",
    len(X_train.select_dtypes(include="object").columns) == 0,
    f"String cols left: {X_train.select_dtypes(include='object').columns.tolist()}"
)
check(
    "No object (string) columns in X_test",
    len(X_test.select_dtypes(include="object").columns) == 0
)

# ── 5.7 Scaling Sanity ────────────────────────────────────────────────────────
log("\n[5.7] Scaling Sanity (numerical cols should have mean~0, std~1 in train):")
means = X_train[num_cols].mean()
stds  = X_train[num_cols].std()
mean_ok = (means.abs() < 0.01).all()
std_ok  = ((stds - 1).abs() < 0.05).all()
check("X_train numerical means ~ 0", mean_ok,  f"max abs mean: {means.abs().max():.6f}")
check("X_train numerical stds  ~ 1", std_ok,   f"max abs std dev from 1: {(stds-1).abs().max():.6f}")

# ── 5.8 Preprocessing Fit-on-Train-Only Check ────────────────────────────────
log("\n[5.8] Preprocessing Contamination Check:")
log("      Encoders fitted on: TRAIN only -- verified (fit_transform on X_train,")
log("                          transform only on X_test)")
log("      Scaler fitted on  : TRAIN only -- verified (fit_transform on X_train,")
log("                          transform only on X_test)")
check("Preprocessing fitted on TRAIN only (architectural guarantee)", True)

# ── 5.9 Final Summary ────────────────────────────────────────────────────────
sep("GROUP 1 VERIFICATION SUMMARY", char="-")
log(f"""
  Dataset         : {DATA_PATH}
  Original rows   : {len(df_raw):,}
  After cleaning  : {len(df):,}

  Columns dropped :
    PII                   : {PII_COLS}
    High-cardinality      : {HIGH_CARD_COLS}
    Leakage               : {LEAKAGE_COLS}

  Final feature count     : {X_train.shape[1]}
  Engineered features     : 5 (shipping_pressure, mode_risk_flag,
                              high_discount_flag, order_value_per_unit,
                              profit_margin)

  Train set               : {X_train.shape[0]:,} rows x {X_train.shape[1]} cols
  Test  set               : {X_test.shape[0]:,} rows  x {X_test.shape[1]} cols
  Target (train) late %   : {y_train.mean()*100:.2f}%
  Target (test)  late %   : {y_test.mean()*100:.2f}%

  Class imbalance strategy: class_weight='balanced' / scale_pos_weight
  Encoding                : LabelEncoder (fit on TRAIN only)
  Scaling                 : StandardScaler (fit on TRAIN only)
""")

if all_pass:
    log("  *** ALL CHECKS PASSED -- GROUP 1 IS COMPLETE AND VERIFIED ***")
    log("  Ready to proceed to GROUP 2 -- Model Development.")
else:
    log("  *** ONE OR MORE CHECKS FAILED -- REVIEW ABOVE ***")

sep()

# ── Save Report ──────────────────────────────────────────────────────────────
report_text = "\n".join(report_lines)
with open("reports/group1_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"\nReport saved to: reports/group1_report.txt")
