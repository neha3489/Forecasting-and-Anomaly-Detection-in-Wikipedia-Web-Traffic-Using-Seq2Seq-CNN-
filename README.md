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
- Uses causal convolutions and an encoder-decoder architecture.
- Captures both short-term and long-term dependencies in sequential data.

### N-BEATS (PyTorch)
- Fully connected deep architecture for time series forecasting.
- Decomposes signals into trend and seasonal components.
- Quantile variant used to estimate uncertainty in predictions.

---

## 🧠 Results

| Model | Test SMAPE |
|--------|--------------|
| Seq2Seq CNN | **42.37** |
| N-BEATS | 120.30 |

Anomaly Detection: Residual-based (σ and MAD thresholds) — top-10 anomalous series identified.
---

## 🧩 Conclusion
- Both models captured key time dependencies and seasonal patterns.
- Seq2Seq CNN (reference) achieved stronger accuracy.
- N-BEATS showed robust generalization but needs hyperparameter tuning.
- Future work: Larger dataset, attention mechanisms, hybrid ensemble models.

---

## 🧰 Requirements
Install the dependencies:
```bash
pip install torch pytorch-lightning pandas numpy matplotlib scikit-learn
