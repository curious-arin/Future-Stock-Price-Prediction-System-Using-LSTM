# Future Stock Price Prediction System using LSTM 📈

## 🙂 Overview

This project focuses on predicting Tesla (TSLA) closing stock prices using historical market data and a Long Short-Term Memory (LSTM) neural network built with PyTorch. The notebook explores stock-price trends, prepares sequential time-series data, and trains an LSTM model to predict the next closing price from recent market observations.

*The goal is to explore how deep learning can learn time related patterns in historical stock prices and produce data-driven price estimates.*

---

## 💡 Problem Statement & Business Impact

- Stock prices fluctuate over time and are influenced by many factors, making their future movement difficult to estimate.
- Reviewing historical prices manually can make it difficult to identify trends and relationships across multiple trading days.
- Time-series models such as LSTM networks can learn patterns from sequential observations and use recent price history to estimate a future value.
- **Goal:** Build an LSTM-based regression model that predicts Tesla's next closing price using a sequence of the previous 60 observations and the selected OHLC [Open, High, Low, and Close] features.
- **Potential value:** Helps understand stock price trends, predict possible price changes, and make smarter stock market decisions using historical data.

---

## 📊 Data Source

- **Dataset:** Historical Tesla (TSLA) stock data retrieved from [Yahoo Finance](https://finance.yahoo.com/) using the `yfinance` Python library.
- The notebook requests approximately 10 years of historical data and uses the available market fields, including Open, High, Low, Close, and Volume.

---

## 🛠️ Tech Stack

- Python
- NumPy, Pandas
- Matplotlib
- PyTorch
- Scikit-learn 
- Google Colab / VS Code

---

## 📂 Project Structure

```text
├── images/
├── models/
│   ├── lstm_model.pth
│   └── scaler.pkl
├── notebooks/
│   └── Notebook.ipynb
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## 🔄 Project Workflow

1. **Data Retrieval:** Download historical TSLA prices from Yahoo Finance using `yfinance`.
2. **Exploratory Data Analysis (EDA):** Inspect summary statistics and visualize Open, High, Low, and Close prices.
3. **Feature Engineering:** Calculate moving average after 15, 30, 60, and 90 days, daily returns, and the daily price gap for exploratory analysis.
4. **Feature Selection:** Use `Close`, `Open`, `High`, and `Low` as model inputs.
5. **Chronological Data Split:** Split the series into 70% training, 15% validation, and 15% testing segments.
6. **Sequence Preparation:** Create rolling sequences of 60 observations, with the following observation's closing price as the target.
7. **Feature Scaling:** Fit `MinMaxScaler` on the training data and apply it to the validation and test data.
8. **LSTM Model Building:** Build a three-layer LSTM network with dropout and fully connected layers.
9. **Model Training:** Train with Adam, Mean Squared Error (MSE) loss, and weight decay; monitor training and validation loss.
10. **Model Evaluation:** Evaluate test loss and calculate MAE, RMSE, and R² after converting predictions back to the original price scale.
11. **Model Saving & Prediction Check:** Save the model weights and scaler, reload the model, and compare a sample prediction with its actual value.

## 🧠 Model Architecture

The notebook defines an LSTM regression model with:

- **LSTM Layer 1:** 128 hidden units
- **Dropout:** 20%
- **LSTM Layer 2:** 128 hidden units
- **Dropout:** 20%
- **LSTM Layer 3:** 64 hidden units
- **Fully Connected Layers:** 64 → 32 → 1, with ReLU activation between dense layers

**Training configuration:**

- **Optimizer:** Adam
- **Learning Rate:** `0.001`
- **Epochs:** `200`
- **Loss Function:** Mean Squared Error (MSE)
- **Batch Size:** Set in the notebook's `batch_size` variable
- **Device:** CPU [because data was not enough, but added a configuration to switch to GPU as a lot of data becomes available]

## 📊 Model Performance

| **Metric** | **Score** |
| ---------- | --------: |
| Mean Absolute Error (MAE) | **20.6625** |
| Root Mean Squared Error (RMSE) | **25.1258** |
| R² Score | **74.31%** |
| Test Loss (scaled MSE) | **0.0040** |

## 🔮 Future Improvements
- Try different LSTM models and settings to improve prediction score.
- Use MLOps techniques to track experiments, monitor model performance, and manage updates.
- Build and deploy a user-friendly frontend to make stock price predictions easier to use.
- Train the model on more data as new stock market data becomes available.
- Later, integrate this trained model into my [Optivest](https://github.com/neuralarin/Optivest) project for practical use.