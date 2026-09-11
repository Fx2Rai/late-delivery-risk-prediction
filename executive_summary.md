# Executive Summary: Late Delivery Risk Intelligence Platform
**APL Logistics (KWE Group)**

## Objective
To develop a proactive machine learning solution capable of predicting late delivery risks across global supply chain operations *before* shipment occurs, enabling operations teams to strategically expedite high-risk shipments while minimising unnecessary logistical costs.

## Dataset & Analytics Overview
The project analysed an extensive dataset of global supply chain records representing over **180,000 global shipments**. 
Through exploratory data analysis, we identified key vulnerability points in the existing logistics pipeline:
- **Average Late Delivery Rate**: A baseline late delivery incidence rate of **54.8%**.
- **Shipping Mode Differences**: Late-delivery rates vary substantially by shipping mode. **First Class** has the highest observed late-delivery rate (**95.32%**), followed by **Second Class (76.63%)**, **Same Day (45.74%)**, and **Standard Class (38.07%)**. Standard Class nevertheless contributes the largest absolute number of late orders (**41,023**) because it represents the largest shipment volume. This distinction separates **late-delivery rate** from **volume contribution**.
- **Regional Congestion Bottlenecks**: Significant variances in regional capability were mapped, identifying high-risk corridors that require systemic infrastructure adjustment. 

## Model Performance & Methodology
The project successfully developed a highly robust predictive engine utilising **XGBoost (eXtreme Gradient Boosting)**. The solution was rigorously tested to guarantee that the model is only exposed to information available *at the time of order placement* (eliminating data leakage).

**Final XGBoost Performance Metrics (binary classification at 0.50 decision threshold):**
- **ROC-AUC (0.7745)**: Demonstrates strong capability in distinguishing between late and on-time shipments across all threshold levels.
- **Binary Precision (84.1% @ 0.50 threshold)**: When the model predicts a delay at the standard 0.50 cutoff, it is correct 84.1% of the time.
- **High-Risk Precision (89.7% @ ≥ 0.70 threshold)**: Orders flagged for immediate operational intervention are genuinely delayed 89.7% of the time (and 95.6% for scores ≥ 0.80).
- **Probability Quality**: The model achieves a **Brier Score of 0.1885**, improving substantially on the **0.2477 uninformative baseline**. This indicates that the predicted probabilities contain meaningful information about late-delivery outcomes; calibration should nevertheless be monitored as the model is deployed and updated.
- **Recall (56.4%)**: Identifies 56.4% of the late shipments in the evaluated test set, while the corresponding precision helps control false delay alarms.

To support the model, the data pipeline dynamically engineers advanced logistical metrics in real-time, including a *Shipping Pressure Index* (scheduled days per unit) and a *Regional Congestion Index*.

## Risk Framework & Operational Tier Precision
The ML pipeline translates continuous probabilities into three actionable risk tiers. The following precision figures are computed directly from the 36,104-order test set and represent **what operations teams observe in practice**:

| Risk Tier | Threshold | Test Orders | Actually Late | Tier Precision |
| :--- | :--- | :--- | :--- | :--- |
| Low | < 40% | 18,931 | 6,719 | 35.5% |
| Medium | 40% – 70% | 6,735 | 3,716 | 55.2% |
| **High** | **≥ 70%** | **10,438** | **9,361** | **89.7%** |

The **High-Risk tier (≥ 70%)** identifies 10,438 high-exposure orders with an exceptional precision of **89.7%**, meaning 9 out of 10 flagged shipments are truly heading for a delivery failure. For ultra-critical interventions, the ≥ 80% sub-tier achieves **95.6%** precision.

The integration of **SHAP (SHapley Additive exPlanations)** ensures that the model is not a "black box". Every prediction surfaced in the dashboard is accompanied by a transparent breakdown of the exact factors (e.g., specific market, shipping class, order size) driving the risk score.

## Business Recommendations
1. **Prioritised Expedited-Service Review**: Prioritise shipments flagged at **≥ 80% probability** for expedited-service or rerouting review. The test-set precision of **95.6%** indicates that this high-confidence group is highly enriched for late deliveries. Final carrier or service changes should remain subject to operational capacity, cost, and service availability.
2. **Shipping-Mode Review**: Review the operational performance and workload associated with each shipping mode. Standard Class should receive attention because it generates the **largest absolute number of late orders (41,023)** due to its high shipment volume, while First Class requires particular attention because it has the **highest late-delivery rate (95.32%)** in the supplied dataset.
3. **Dynamic Resource Allocation**: Utilise the dashboard's Regional Heatmaps to predict upcoming regional congestion. Reallocate warehouse staffing in highly affected geographic nodes prior to peak dispatch windows.

**Conclusion:** The implementation of this Risk Intelligence Platform marks a transition from reactive delay management to proactive logistical strategy. With a model **ROC-AUC of 0.7745** and an operational **High-Risk tier precision of 89.7%**, the platform provides actionable and explainable intelligence for prioritising potentially delayed shipments. Its strongest operational value is the ability to focus intervention on high-probability cases while distinguishing shipment-volume effects from underlying late-delivery rates.
