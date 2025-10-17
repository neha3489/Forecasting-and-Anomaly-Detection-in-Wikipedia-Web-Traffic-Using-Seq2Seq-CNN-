# Forecasting and Anomaly Detection in Wikipedia Web Traffic Using Seq2Seq CNN and N-BEATS

This project focuses on forecasting Wikipedia web traffic data and detecting anomalies using deep learning techniques. Two models were implemented — **Seq2Seq CNN** and **N-BEATS (PyTorch)** — to predict page view trends and identify unusual behavior patterns.

---

## 📊 Dataset Description
- **Source:** Kaggle & Google’s Web Traffic Time Series Forecasting Challenge  
- **Train Range:** July 1, 2015 → November 1, 2016 (490 days)  
- **Test Range:** November 2, 2016 → December 31, 2016 (60 days)  
- **Features:** 5 per day, preprocessed using log normalization  

---

## 🤖 Models Used
### Seq2Seq CNN
-Employs causal convolutions to model temporal dependencies without allowing information leakage from future time steps.
-Learns encoder-decoder representations that handle varying sequence lengths and generate stable multi-step predictions for time series data.


### N-BEATS (PyTorch)
-A deep fully-connected architecture designed specifically for time series forecasting, implemented using PyTorch.
-Decomposes the input signal into trend and seasonal components, making it robust and interpretable for both short- and long-term forecasts.

---

## 🧠 Results

| Model | Test SMAPE |
|--------|--------------|
| Seq2Seq CNN | **40.519** |
| N-BEATS | 120.30 |

Anomaly Detection: Residual-based (σ and MAD thresholds) — top-10 anomalous series identified.
---

## 🧩 Conclusion
The Seq2Seq Causal CNN model achieved a Test SMAPE of 40.52, outperforming the previously reported best value of 42.37 from the paper. We have also used N-BEATS (PyTorch) model, which recorded a SMAPE of around 120. This demonstrates that the Seq2Seq CNN architecture captures temporal dependencies and seasonal patterns more effectively, making it highly suitable for time series forecasting and anomaly detection on the Wikipedia web traffic dataset.
The model shows strong generalization across diverse page categories, maintaining stability even with irregular traffic patterns.
Future work will focus on utilizing a larger dataset and exploring attention mechanisms and hybrid ensemble models to further improve forecasting performance.
---
