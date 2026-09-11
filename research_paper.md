# Machine Learning--based Late Delivery Risk Prediction in Global Supply Chain Operations

## Extended Research Paper

### 1. Abstract

Late delivery is a major operational risk in global supply chains
because a delay at one point in the logistics network can propagate into
inventory shortages, service-level agreement (SLA) breaches, customer
dissatisfaction, expedited-shipping costs, and inefficient use of
warehouse and transportation capacity. This study develops a
pre-shipment machine-learning framework for estimating the probability
that an order will be delivered late, using the APL Logistics
supply-chain dataset.

The source dataset contains **180,519 records and 40 original columns**.
After removal of personally identifiable information, high-cardinality
fields, redundant fields, and post-shipment variables that would expose
the outcome, the modeling dataset is split into **144,415 training
observations and 36,104 held-out test observations**. The target,
`Late_delivery_risk`, has an overall positive rate of **54.83%**, making
late delivery the majority class but not an extreme class-imbalance
problem.

The analytical pipeline combines missing-value handling, numerical
standardization, categorical one-hot encoding, stratified train-test
splitting, and supply-chain-specific feature engineering. Eight
engineered variables are used, including a Shipping Pressure Index, Mode
Risk Flag, Regional Congestion Index, Order Complexity Score, Order
Value per Unit, Profit Margin, High Discount Flag, and Benefit per Unit.
A leakage-aware implementation of regional target encoding is included
in the supplied `group2_ml.py` pipeline, where the regional encoding is
recomputed from the training portion of each cross-validation fold.

Three classifiers are evaluated: Logistic Regression, Random Forest, and
XGBoost. On the independent test set, XGBoost produces the strongest
discrimination with **ROC-AUC = 0.7745** and an **F1 score = 0.6749** at
the standard 0.50 decision threshold. Its binary precision is **84.12%**
and recall is **56.35%**. The operational probability framework
separates orders into Low (\<0.40), Medium (0.40--0.70), and High
(≥0.70) risk. On the held-out test set, the High-Risk group contains
**10,438 orders**, of which **9,361 are actually late**, yielding
**89.7% precision**. For the most critical orders (probability ≥0.80),
precision rises to **95.6%**.

The project also incorporates SHAP explainability and a Streamlit
Operations Action Panel, allowing logistics users to inspect risk
probabilities, compare regions and shipping modes, and prioritize
individual orders. The extended analysis additionally highlights an
important interpretation issue: although Standard Class contributes the
largest *number* of late orders because of its large volume, the
observed late-delivery *rate* is highest for First Class in the supplied
dataset. This distinction should be reflected in the final business
recommendations.

------------------------------------------------------------------------

## 2. Introduction

### 2.1 Background

Modern supply chains operate as interconnected networks involving
customers, warehouses, transportation services, distribution nodes, and
regional markets. Delivery performance therefore depends not only on the
physical movement of goods but also on the interaction between order
characteristics, shipping commitments, geography, commercial conditions,
and operational capacity.

A conventional monitoring system usually identifies a delivery problem
after the event has already occurred. Such a reactive approach limits
the available intervention window. Once an order is already in transit
and has missed its planned schedule, options such as rerouting,
reprioritization, carrier escalation, customer notification, or
warehouse intervention may be more expensive and less effective.

The objective of this project is consequently different from
retrospective delay analysis. The central question is:

> **Which orders are likely to experience a late delivery before
> shipment occurs?**

A probability-based answer allows the organization to allocate scarce
operational resources according to risk rather than treating every order
equally.

### 2.2 Problem Statement

The project requirements identify three major operational challenges:

1.  Unpredictable shipment delays.
2.  High operational costs associated with last-minute corrective
    actions.
3.  Limited ability to prioritize orders according to their probability
    of failure.

The proposed system addresses these challenges by producing:

-   a continuous late-delivery probability between 0 and 1;
-   a discrete Low/Medium/High risk classification;
-   model-based explanations of important risk drivers; and
-   an operational queue of high-risk orders.

The supplied project specification explicitly emphasizes prediction
**before shipment**, rather than explaining an outcome using information
that only becomes available after shipment.

### 2.3 Research Contributions

The extended study makes the following practical contributions:

-   establishes a leakage-controlled pre-shipment prediction boundary;
-   combines transactional, geographic, commercial, and shipping
    variables;
-   engineers domain-specific indicators rather than relying only on raw
    columns;
-   compares linear and non-linear machine-learning models;
-   translates probabilities into operational risk tiers;
-   reports both model-level and business-level precision;
-   incorporates SHAP explanations for individual decisions;
-   provides a Streamlit-based operational interface; and
-   identifies data-quality and interpretation issues that should be
    addressed before production deployment.

------------------------------------------------------------------------

## 3. Related Work and Methodological Position

The project specification positions the work within supply-chain
analytics and predictive logistics. Earlier analytical approaches
frequently focus on historical performance, transit duration, or
post-shipment status. Those variables can be highly predictive, but they
may not be available when an operational decision must be made.

The key methodological distinction in this study is therefore the
**pre-shipment information boundary**.

For example:

-   `Days for shipping (real)` represents actual shipping duration and
    is only known after the event.
-   `Delivery Status` describes the realized delivery state.
-   `Order Status` may contain information about the eventual state of
    the order.

Including such fields would allow the model to learn from information
that is unavailable at prediction time. The supplied preprocessing
pipeline therefore removes them before modeling.

The study also extends predictive modeling with SHAP, which provides a
feature-level explanation of model output rather than presenting a
probability as an unexplained black-box score.

------------------------------------------------------------------------

## 4. Dataset Description

### 4.1 Source Dataset

The supplied APL Logistics CSV contains:

-   **180,519 records**
-   **40 original columns**
-   **1 binary target variable**
-   a mixture of numerical and categorical variables.

The project guide describes variables covering payment type, scheduled
shipping duration, financial measures, customer information, geographic
information, product information, shipping mode, and delivery outcomes.

The target variable is:

`Late_delivery_risk`

where:

-   `1` = late delivery;
-   `0` = not late.

The observed target distribution in the supplied CSV is:

  Target            Records         Share
  ----------- ------------- -------------
  On-Time            81,542        45.17%
  Late               98,977        54.83%
  **Total**     **180,519**   **100.00%**

Thus, late deliveries form the majority class.

### 4.2 Modeling Boundary

The modeling dataset intentionally excludes three types of information:

**Post-shipment leakage** - `Days for shipping (real)` -
`Delivery Status` - `Order Status`

**Personally identifiable information** - customer first and last
names; - street address; - customer ZIP code; - customer identifiers.

**High-cardinality or deployment-unhelpful location/product fields** -
Product Name; - Order City; - Order State; - Order Country; - Customer
Country; - Customer State.

This creates a cleaner feature space and reduces the risk that the model
memorizes individual entities rather than learning generalizable
operational patterns.

### 4.3 Missing Values

The supplied dataset contains missing values in a small number of
fields. In particular, the project preprocessing identifies missing
values in `Customer Lname` and `Customer Zipcode`.

The modeling pipeline uses:

-   median imputation for numerical variables;
-   most-frequent imputation for categorical variables.

The use of a unified preprocessing pipeline ensures that the same
transformations can be applied during inference.

------------------------------------------------------------------------

## 5. Data Preprocessing

### 5.1 Stratified Train-Test Split

The dataset is divided using an **80/20 stratified split** with random
state 42.

  Dataset      Records
  ---------- ---------
  Training     144,415
  Test          36,104
  Total        180,519

The training set contains 79,181 late and 65,234 on-time records.

The held-out test set contains:

-   **19,796 late orders**
-   **16,308 on-time orders**

Stratification preserves the approximate target distribution between
training and test partitions.

### 5.2 Numerical Processing

The supplied pipeline uses:

`SimpleImputer(strategy="median") → StandardScaler()`

This is applied to the numerical feature path.

### 5.3 Categorical Processing

Categorical variables are processed through:

`SimpleImputer(strategy="most_frequent") → OneHotEncoder(handle_unknown="ignore")`

The `handle_unknown="ignore"` configuration is important for deployment
because a future order may contain a category that was not present in
the training data.

### 5.4 Pipeline Integrity

The project uses a Scikit-Learn `ColumnTransformer` and complete model
pipelines. This prevents preprocessing operations such as scaling and
encoding from being performed inconsistently between training and
inference.

The supplied project metadata reports **24 numerical features and 7
categorical features** after feature engineering, with **124 transformed
features** after encoding.

------------------------------------------------------------------------

## 6. Feature Engineering

The project goes beyond raw variables by creating eight derived
indicators.

### 6.1 Shipping Pressure Index

The supplied implementation defines:

\[ ShippingPressureIndex = `\frac{DaysForShipmentScheduled}`{=tex}
{OrderItemQuantity + 1} \]

The purpose is to represent the scheduled time available relative to
order quantity. The `+1` denominator offset avoids division by zero and
moderates the scale for small quantities.

### 6.2 Mode Risk Flag

The supplied implementation groups:

-   First Class and Same Day → `0`
-   Second Class and Standard Class → `1`

This feature provides a coarse distinction between express-like and
economy-like modes. However, the raw shipping-mode one-hot variables are
also retained, allowing the XGBoost model to learn mode-specific
differences.

### 6.3 Regional Congestion Index

The Regional Congestion Index is a target-encoded estimate of historical
late-delivery frequency for each `Order Region`.

For a region (r):

\[ RegionalCongestionIndex(r) = `\frac{\text{Late orders in }r}`{=tex}
{`\text{All orders in }`{=tex}r} \]

The mapping is calculated from training data and a global mean is used
as the fallback for unseen regions.

The supplied training run maps **23 regions**, with a global fallback
mean of approximately **0.5483**.

### 6.4 Order Complexity Score

The implementation combines normalized quantity, discount rate, and
scheduled days:

\[ Complexity = `\frac{
Quantity/5 +
DiscountRate +
ScheduledDays/4
}{3}`{=tex} \]

This produces a composite indicator of order characteristics that may
increase operational complexity.

### 6.5 Additional Financial and Commercial Features

Four additional variables are created:

**Order Value per Unit**

\[ OrderValuePerUnit = `\frac{Sales}{OrderItemQuantity+1}`{=tex} \]

**Profit Margin**

\[ ProfitMargin = `\frac{OrderProfitPerOrder}{Sales+1}`{=tex} \]

**High Discount Flag**

\[ HighDiscountFlag = I(DiscountRate \> 0.10) \]

**Benefit per Unit**

\[ BenefitPerUnit = `\frac{BenefitPerOrder}{OrderItemQuantity+1}`{=tex}
\]

These features provide the model with relative measures rather than only
absolute monetary values.

------------------------------------------------------------------------

## 7. Exploratory Data Analysis

### 7.1 Overall Late-Delivery Risk

The overall late-delivery rate is **54.83%**, indicating that the
historical process contains a substantial delivery-risk problem.

This also establishes an important business baseline: a system that
simply predicts every order as late would have high recall but poor
operational usefulness. The objective is instead to distinguish
higher-risk orders from lower-risk orders.

### 7.2 Shipping Mode Analysis

A detailed calculation from the supplied CSV reveals the following:

  Shipping Mode       Orders   Late Orders   Late Rate
  ---------------- --------- ------------- -----------
  Standard Class     107,752        41,023      38.07%
  Second Class        35,216        26,987      76.63%
  First Class         27,814        26,513      95.32%
  Same Day             9,737         4,454      45.74%

This finding requires an important interpretation distinction.

**Standard Class accounts for the largest absolute number of late orders
because it represents the largest share of all orders. However, First
Class has the highest observed late-delivery rate.**

The same pattern is visible in scheduled shipping days because First
Class corresponds to one scheduled day in the supplied data, whereas
Standard Class corresponds to four scheduled days.

Therefore, the research paper should not describe Standard Class as
having the highest *late rate*. A more accurate conclusion is that
**shipping mode and scheduled duration are dominant structural
predictors, while high-volume Standard Class generates the largest
absolute number of late cases.**

### 7.3 Scheduled Shipping Duration

The target rate varies substantially by scheduled duration:

    Scheduled Days    Orders   Late Rate
  ---------------- --------- -----------
                 0     9,737      45.74%
                 1    27,814      95.32%
                 2    35,216      76.63%
                 4   107,752      38.07%

This is one of the strongest empirical patterns in the dataset. Very
short scheduled windows are associated with substantially higher
late-delivery incidence.

The finding supports the use of scheduled duration as a central
predictor, but it also indicates that the model is partly learning the
operational difficulty embedded in the original scheduling policy.

### 7.4 Regional Risk

The highest observed historical late rates in the raw dataset include:

  Region             Orders   Late Rate
  ---------------- -------- -----------
  Central Africa      1,677      57.96%
  South Asia          7,731      56.27%
  East Africa         1,852      55.94%
  Western Europe     27,109      55.85%
  South of USA        4,045      55.77%

Central Africa is approximately three percentage points above the global
baseline, while South Asia is approximately 1.4 percentage points above
it.

The model's regional-risk output similarly places Central Africa,
Eastern Europe, South Asia, and South of USA among the higher-risk
regions in the held-out test predictions.

### 7.5 Customer Segment

The raw-data late rates are relatively close across customer segments:

  Customer Segment     Orders   Late Rate
  ------------------ -------- -----------
  Home Office          32,226      55.07%
  Consumer             93,504      54.81%
  Corporate            54,789      54.72%

This suggests that customer segment is unlikely to be as operationally
decisive as shipping mode and scheduled duration.

### 7.6 Discount Rate

The supplied data does not show a simple monotonic relationship between
discount rate and delay risk. For example, the observed late rate is
approximately 54.87% for orders with a 0--5% discount and approximately
54.57% for orders with a 5--10% discount.

The highest discount band above 20% has a late rate of approximately
55.67%.

Consequently, the relationship between discounts and late delivery
should be described as **weak or context-dependent**, rather than as a
direct causal effect.

------------------------------------------------------------------------

## 8. Model Development

Three models are evaluated.

### 8.1 Logistic Regression

Logistic Regression serves as an interpretable linear benchmark. It
establishes how well a relatively simple additive model can distinguish
late from on-time orders.

### 8.2 Random Forest

Random Forest is used to capture non-linear interactions and threshold
effects that Logistic Regression may miss.

The supplied implementation uses:

-   300 trees;
-   maximum depth of 20;
-   minimum leaf size of 5;
-   square-root feature selection;
-   balanced class weights.

### 8.3 XGBoost

XGBoost is the selected model because it achieves the highest held-out
ROC-AUC.

The supplied configuration uses:

-   400 estimators;
-   maximum depth of 8;
-   learning rate of 0.05;
-   0.8 row subsampling;
-   0.8 column subsampling;
-   minimum child weight of 5;
-   gamma = 0.1;
-   regularization through alpha and lambda;
-   class-weight adjustment through `scale_pos_weight`.

------------------------------------------------------------------------

## 9. Cross-Validation and Leakage Control

The project initially identified a methodological risk in regional
target encoding. If the regional late-delivery rate is calculated using
all training labels before cross-validation, the validation records
indirectly influence their own encoded feature.

The supplied `group2_ml.py` contains a corrected implementation. Its
`cv_score()` function:

1.  creates each StratifiedKFold split;
2.  extracts the fold's training subset;
3.  computes the regional target map using only that subset's labels;
4.  engineers the regional feature for both fold-training and
    fold-validation data using that map;
5.  clones a fresh model pipeline;
6.  fits on the fold-training data; and
7.  evaluates AUC on the validation data.

This is the correct direction for leakage control.

However, the stored project artifacts should clearly record whether the
reported CV summary was generated after this correction. The **held-out
test set remains the primary unbiased reference for the final reported
performance**, because it was not used to construct the final regional
map.

------------------------------------------------------------------------

## 10. Model Evaluation

### 10.1 Test-Set Performance

  Model                      ROC-AUC    Precision       Recall           F1
  --------------------- ------------ ------------ ------------ ------------
  Logistic Regression         0.7405       0.8499       0.5420       0.6619
  Random Forest               0.7543       0.8450       0.5517       0.6676
  **XGBoost**             **0.7745**   **0.8412**   **0.5635**   **0.6749**

All binary precision, recall, and F1 values above use the standard
**0.50 threshold**.

XGBoost improves ROC-AUC by:

-   0.0340 over Logistic Regression;
-   0.0202 over Random Forest.

Its F1 score is also the highest.

### 10.2 Confusion Matrix

For XGBoost at the 0.50 binary threshold:

                         Predicted On-Time   Predicted Late
  -------------------- ------------------- ----------------
  **Actual On-Time**                14,202            2,106
  **Actual Late**                    8,640           11,156

This gives:

-   True Negatives = 14,202
-   False Positives = 2,106
-   False Negatives = 8,640
-   True Positives = 11,156

The resulting accuracy is approximately **70.24%**.

Specificity is approximately **87.09%**, while the false-negative rate
is approximately **43.65%**. The latter is operationally important:
although positive predictions are relatively precise, a substantial
number of actual late orders remain below the standard 0.50 threshold.

This is one reason why the continuous probability score and high-risk
operational tier are more useful than treating the 0.50 cutoff as the
only business decision rule.

### 10.3 Cross-Validation Results

The stored project results report:

  Model                   Mean CV AUC   Standard Deviation
  --------------------- ------------- --------------------
  Logistic Regression          0.7397               0.0025
  Random Forest                0.7503               0.0017
  XGBoost                      0.7683               0.0018

XGBoost again ranks first.

The supplied source code now implements per-fold target encoding, so the
reproducibility report should distinguish the corrected implementation
from any earlier artifact generated before the correction.

------------------------------------------------------------------------

## 11. Risk Probability and Operational Classification

A classifier's probability is more useful operationally when converted
into decision bands.

The project defines:

-   **Low Risk:** probability \< 0.40
-   **Medium Risk:** 0.40 ≤ probability \< 0.70
-   **High Risk:** probability ≥ 0.70

### 11.1 Test-Set Tier Results

  Risk Tier     Test Orders   Actually Late   Precision
  ----------- ------------- --------------- -----------
  Low                18,931           6,719       35.5%
  Medium              6,735           3,716       55.2%
  **High**       **10,438**       **9,361**   **89.7%**

The High-Risk group contains **28.9% of the test orders** but captures a
very high concentration of actual late cases.

For probabilities ≥0.80:

-   6,574 orders are flagged;
-   6,282 are actually late;
-   precision = **95.6%**.

This creates a useful hierarchy of interventions:

  Probability   Suggested operational response
  ------------- --------------------------------------------
  \<0.40        Standard monitoring
  0.40--0.70    Proactive watchlist
  0.70--0.80    Immediate review
  ≥0.80         Critical intervention / expedited handling

These are operational recommendations based on the supplied project's
performance outputs; the exact intervention policy should ultimately be
validated against cost, capacity, and SLA requirements.

------------------------------------------------------------------------

## 12. Probability Calibration

The project reports a Brier Score of **0.1885**, compared with an
uninformative baseline of **0.2477**.

A lower Brier Score indicates better probabilistic accuracy.

The supplied calibration observations include:

  Predicted Probability Band     Observed Late Rate
  ---------------------------- --------------------
  0--20%                                      15.2%
  40--60%                                     51.4%
  80--100%                                    95.6%

The close relationship between predicted risk and observed event
frequency supports the use of probability bands for operational
prioritization.

Nevertheless, calibration should be monitored after deployment because
changes in customer mix, network capacity, shipping policies,
seasonality, or external disruptions can cause probability drift.

------------------------------------------------------------------------

## 13. Explainability with SHAP

The project uses SHAP with `TreeExplainer` for XGBoost.

### 13.1 Global Importance

The supplied XGBoost feature-importance output identifies the following
leading predictors:

    Rank Feature                           Importance
  ------ ------------------------------- ------------
       1 Shipping Mode_Standard Class          0.3663
       2 Mode Risk Flag                        0.1904
       3 Shipping Mode_Same Day                0.1484
       4 Days for shipment (scheduled)         0.1223
       5 Shipping Mode_First Class             0.0610
       6 Type_TRANSFER                         0.0080
       7 Shipping Mode_Second Class            0.0071
       8 Shipping Pressure Index               0.0029
       9 Type_CASH                             0.0016
      10 Type_PAYMENT                          0.0015

This confirms that **shipping configuration and scheduled duration
dominate the learned model**.

A critical methodological point is that feature importance indicates
predictive association, not causation. For example, a high importance
for Standard Class does not by itself prove that Standard Class causes
late delivery.

### 13.2 Local Explanations

The supplied SHAP evaluation demonstrates order-level explanations in
which individual feature contributions are expressed as positive or
negative impacts on the model output.

This enables the dashboard to answer a practical question:

> **Why was this particular order flagged as high risk?**

Instead of displaying only "High Risk," the interface can identify the
variables most responsible for increasing or decreasing the score.

------------------------------------------------------------------------

## 14. Business Interpretation of Shipping-Mode Findings

The raw dataset and model outputs together require careful
interpretation.

The dataset shows:

-   First Class: 95.32% late;
-   Second Class: 76.63% late;
-   Same Day: 45.74% late;
-   Standard Class: 38.07% late.

Therefore, the statement that Standard Class has the highest *rate* of
late delivery is not supported by the supplied CSV.

However, Standard Class represents approximately **59.7% of all
records**, so it still contributes the largest absolute number of late
orders.

A stronger business interpretation is:

> **Shipping mode is a dominant risk discriminator, but the relationship
> is not equivalent to "slower service always means higher observed late
> rate." Short scheduled windows appear particularly difficult to
> satisfy in this dataset.**

This finding is operationally more useful because it points toward
**schedule design and SLA feasibility**, not merely carrier speed.

------------------------------------------------------------------------

## 15. Regional Risk and Capacity Planning

Regional aggregation allows the individual probabilities to be converted
into network-level planning information.

The supplied test predictions generate regional measures including:

-   total orders;
-   mean predicted late probability;
-   high-risk order count;
-   percentage of orders classified as high risk.

For example, the generated regional output places Central Africa at
approximately **0.552 mean predicted probability**, with **33.1%** of
its test orders classified as High Risk.

Other high-risk regional groups include Eastern Europe, South Asia,
South of USA, Southeast Asia, and US Center.

Regional risk should not automatically be interpreted as infrastructure
failure. Differences can reflect:

-   shipping-mode mix;
-   scheduled-duration policies;
-   order composition;
-   customer/product mix;
-   historical operational conditions.

Therefore, regional risk maps should be used to prioritize investigation
and resource planning rather than to assign blame to a geographic node.

------------------------------------------------------------------------

## 16. Streamlit Operations Action Panel

The supplied application operationalizes the model through Streamlit.

### 16.1 Delay Risk Overview

The dashboard provides macro-level indicators such as:

-   total orders;
-   risk distribution;
-   high-risk order count;
-   overall risk profile.

### 16.2 Order-Level Prediction

Users can enter or inspect an individual order and receive:

-   predicted late probability;
-   Low/Medium/High risk category;
-   contributing factors through SHAP.

### 16.3 Region and Shipping-Mode Analysis

The dashboard supports:

-   regional risk visualization;
-   shipping-mode comparison;
-   filtering by market and customer segment.

### 16.4 Operations Action Queue

The Action Panel converts the model into a workflow.

A practical priority ordering is:

1.  probability ≥0.80;
2.  probability 0.70--0.80;
3.  probability 0.40--0.70;
4.  probability \<0.40.

This supports differentiated interventions rather than applying the same
treatment to every order.

------------------------------------------------------------------------

## 17. Proposed Operational Workflow

A production implementation can follow this sequence:

**Order placement**

↓

**Pre-shipment data capture**

↓

**Data validation and preprocessing**

↓

**Feature engineering**

↓

**XGBoost probability prediction**

↓

**Risk classification**

↓

**SHAP explanation**

↓

**Operational action**

↓

**Outcome capture**

↓

**Model monitoring and retraining**

This closes the loop between prediction and operational learning.

For example:

-   High-risk orders can receive manual review.
-   Critical orders can be considered for premium shipping or
    reprioritization.
-   Regional concentrations can trigger staffing or capacity planning.
-   Medium-risk orders can receive proactive monitoring.
-   Low-risk orders can follow normal workflows.

------------------------------------------------------------------------

## 18. Business Recommendations

### Recommendation 1: Use Probability-Based Prioritization

Do not treat the 0.50 binary threshold as the only decision rule. The
≥0.70 operational tier has substantially higher precision and is more
aligned with intervention decisions.

### Recommendation 2: Establish a Critical ≥0.80 Queue

The ≥0.80 group has **95.6% precision** in the supplied test set. This
makes it a strong candidate for scarce, high-cost interventions, subject
to business validation.

### Recommendation 3: Audit Scheduling Policies

The strong relationship between scheduled duration and observed
late-delivery rate suggests that SLA and scheduling policy deserve
direct investigation.

In particular, orders with one- or two-day schedules exhibit much higher
historical late rates than four-day schedules.

### Recommendation 4: Do Not Assume Shipping-Class Labels Reflect Actual Risk

The raw data demonstrates that First Class has the highest observed late
rate. Operational teams should therefore analyze the relationship among
shipping mode, scheduled days, and actual network performance instead of
assuming that a service label automatically represents low risk.

### Recommendation 5: Use Regional Predictions for Capacity Planning

Regional heatmaps should guide:

-   warehouse staffing;
-   dispatch prioritization;
-   exception management;
-   capacity allocation;
-   regional performance reviews.

### Recommendation 6: Integrate with WMS/OMS

The prediction pipeline can be integrated with warehouse or
order-management systems so that a risk score is generated automatically
at order placement.

### Recommendation 7: Monitor Model Drift

Production monitoring should track:

-   ROC-AUC;
-   precision and recall;
-   high-risk precision;
-   calibration/Brier score;
-   feature distribution drift;
-   regional performance drift;
-   shipping-mode performance drift.

### Recommendation 8: Preserve the Leakage Boundary

Any future data source should be classified as either:

1.  available at order placement;
2.  available before dispatch; or
3.  available only after shipment.

Only the first two categories should be included if the objective
remains pre-shipment prediction.

------------------------------------------------------------------------

## 19. Limitations

### 19.1 External Dynamic Variables

The model does not include live external variables such as:

-   weather;
-   port congestion;
-   strikes;
-   traffic;
-   geopolitical disruption;
-   real-time transportation capacity.

Such variables could improve responsiveness to sudden disruptions.

### 19.2 Carrier Identification

The dataset does not provide explicit carrier identification.
Consequently, the model cannot directly determine which logistics
provider caused a delay.

### 19.3 Causal Interpretation

The model identifies predictive relationships, not causal effects. A
feature with high SHAP importance should not automatically be
interpreted as the root cause of delay.

### 19.4 Short Scheduled Windows

Because scheduled shipping duration is highly predictive, the model may
partly encode the organization's existing scheduling policy. This is
useful for prediction but should be considered when interpreting the
model as an explanation of underlying logistics causes.

### 19.5 Cross-Validation Artifact Management

The supplied source code contains a corrected fold-specific
target-encoding procedure. Research reporting should ensure that all
stored CV metrics are explicitly tied to the exact code version that
generated them.

### 19.6 Temporal Validation

The current evaluation uses a random stratified train-test split. For a
genuinely forward-looking deployment, an additional **time-based
validation experiment** would be valuable, because a random split may
mix patterns from different periods.

------------------------------------------------------------------------

## 20. Future Research

Several extensions could substantially strengthen the study.

### 20.1 Temporal Validation

Sort records chronologically and train on earlier periods while testing
on later periods. This would more closely simulate real deployment.

### 20.2 Cost-Sensitive Threshold Optimization

The 0.70 and 0.80 thresholds are useful operational bands, but an
optimal threshold should be determined from:

-   cost of a missed late order;
-   cost of an unnecessary intervention;
-   cost of premium shipping;
-   customer/SLA penalties.

A formal expected-cost framework could identify the economically optimal
intervention threshold.

### 20.3 Advanced Calibration

Platt scaling or isotonic calibration could be tested if future
validation demonstrates systematic probability miscalibration.

### 20.4 External Data Integration

Future versions could integrate weather, port, traffic, holiday,
capacity, and geopolitical indicators.

### 20.5 Fairness and Segment Monitoring

Performance should be evaluated separately across markets, customer
segments, and regions to ensure that a strong aggregate AUC does not
conceal weak performance in particular operational groups.

### 20.6 Model Monitoring

A production model should trigger retraining when:

-   feature distributions change;
-   calibration deteriorates;
-   high-risk precision falls;
-   new shipping policies are introduced;
-   regional operational conditions change.

------------------------------------------------------------------------

## 21. Conclusion

This study demonstrates the feasibility of transforming late-delivery
management from a reactive process into a predictive risk-intelligence
workflow.

Using **180,519 supply-chain records**, the project constructs a
leakage-controlled modeling dataset and compares Logistic Regression,
Random Forest, and XGBoost. XGBoost provides the strongest held-out
discrimination with **ROC-AUC = 0.7745**, precision of **84.12%**,
recall of **56.35%**, and F1 score of **0.6749** at the standard 0.50
threshold.

The operational probability framework provides an additional business
perspective. At the ≥0.70 threshold, **10,438 of 36,104 test orders**
are classified as High Risk, with **9,361 actually late**, yielding
**89.7% precision**. At ≥0.80, precision reaches **95.6%**.

The analysis also shows that shipping mode and scheduled duration are
the dominant predictive factors. Importantly, the raw data demonstrates
that Standard Class contributes the largest number of late cases because
of its volume, while First Class has the highest late-delivery rate.
This distinction should guide the final business interpretation and
prevent an incorrect conclusion that Standard Class has the highest rate
of failure.

The Streamlit dashboard, SHAP explanations, regional risk analysis, and
action queue provide the operational layer required to turn model
probabilities into decisions. The next stage should focus on time-based
validation, cost-sensitive threshold selection, external dynamic data,
and continuous production monitoring.

Overall, the project establishes a practical predictive backbone for
intelligent supply-chain operations while highlighting the importance of
strict leakage control, careful statistical interpretation, and
alignment between model outputs and real operational costs.

------------------------------------------------------------------------

## 22. References

1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*.

2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

3. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

4. Breiman, L. (2001). Random forests. *Machine Learning, 45*, 5–32. https://doi.org/10.1023/A:1010933404324

5. Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley. https://doi.org/10.1002/9781118548387

6. Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J. F., & Dennison, D. (2015). Hidden technical debt in machine learning systems. *Advances in Neural Information Processing Systems, 28*.

7. Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data, 6*(4), Article 15. https://doi.org/10.1145/2382577.2382579

8. Kuhn, M., & Johnson, K. (2013). *Applied Predictive Modeling*. Springer. https://doi.org/10.1007/978-1-4614-6849-3

9. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

10. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

11. Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *Proceedings of the 22nd International Conference on Machine Learning*, 625–632. https://doi.org/10.1145/1102351.1102430

12. Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review, 78*(1), 1–3. https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2

13. Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010

14. Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

15. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357. https://doi.org/10.1613/jair.953

16. He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering, 21*(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239

17. Molnar, C. (2022). *Interpretable Machine Learning* (2nd ed.). https://christophm.github.io/interpretable-ml-book/

18. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). “Why should I trust you?” Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135–1144. https://doi.org/10.1145/2939672.2939778

19. Breck, E., Cai, S., Nielsen, E., Salib, M., & Sculley, D. (2017). The ML test score: A rubric for ML production readiness and technical debt reduction. *2017 IEEE International Conference on Big Data*, 1123–1132. https://doi.org/10.1109/BigData.2017.8258038

20. Baryannis, G., Validi, S., Dani, S., & Antoniou, G. (2019). Supply chain risk management and artificial intelligence: State of the art and future research directions. *International Journal of Production Research, 57*(7), 2179–2202. https://doi.org/10.1080/00207543.2018.1530476

21. Min, H. (2010). Artificial intelligence in supply chain management: Theory and applications. *International Journal of Logistics Research and Applications, 13*(1), 13–39. https://doi.org/10.1080/13675560902736537

22. Waller, M. A., & Fawcett, S. E. (2013). Data science, predictive analytics, and big data: A revolution that will transform supply chain design and management. *Journal of Business Logistics, 34*(2), 77–84. https://doi.org/10.1111/jbl.12010

23. Choi, T.-M., Wallace, S. W., & Wang, Y. (2018). Big data analytics in operations management. *Production and Operations Management, 27*(10), 1868–1883. https://doi.org/10.1111/poms.12838

24. Ivanov, D., & Dolgui, A. (2020). Viability of intertwined supply networks: Extending the supply chain resilience angles towards survivability. *International Journal of Production Research, 58*(10), 2904–2915. https://doi.org/10.1080/00207543.2020.1750727

25. Ivanov, D., & Dolgui, A. (2021). A digital supply chain twin for managing the disruption risks and resilience in the era of Industry 4.0. *Production Planning & Control, 32*(9), 775–788. https://doi.org/10.1080/09537287.2020.1768450

26. APL Logistics project specification and dataset documentation supplied with the project files.

27. APL Logistics late-delivery prediction project code, evaluation reports, model metadata, and generated outputs supplied with the project files.
