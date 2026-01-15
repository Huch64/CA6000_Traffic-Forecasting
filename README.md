# 🚗 Interstate Traffic Forecasting: A Data-Centric SimpleRNN Approach

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project serves as the coursework for **NTU CA6000 (Applied AI Programming)**. 
The objective is to forecast **next-hour traffic volume** on the westbound I-94 interstate highway using historical data. 

Adopting a **Data-Centric AI** approach, this project places heavy emphasis on rigorous data cleaning, anomaly detection, and time-series feature engineering before implementing a **SimpleRNN (Recurrent Neural Network)** to capture sequential dependencies.

## 📂 Dataset
* **Source:** [Metro Interstate Traffic Volume (Kaggle)](https://www.kaggle.com/datasets/anshtanwar/metro-interstate-traffic-volume)
* **Description:** Hourly traffic data collected by MnDOT (2012-2018).
* **Features:** Traffic volume, weather conditions (temp, rain, snow, clouds), and holiday flags.

## 🛠️ Methodology

### 1. Data Diagnosis & Cleaning
* **Anomaly Removal:** Filtered out physically impossible values (e.g., Temperature = 0 Kelvin) and extreme weather outliers.
* **Sparse Data Handling:** Addressed sparse `holiday` data by using the Python `holidays` library to backfill missing holiday flags, ensuring the model recognizes non-working days.
* **Gap Filling:** Handled duplicate timestamps and missing hourly intervals.

### 2. Feature Engineering
* **Cyclical Encoding:** Transformed `Hour`, `Day`, and `Month` into Sine/Cosine pairs to preserve temporal continuity (e.g., hour 23 is close to hour 0).
* **Lag Features:** Created a **24-hour lookback window** (sliding window) as the input sequence for the model.
* **Scaling:** Applied `StandardScaler` to normalize numerical inputs for neural network stability.

### 3. Model Architecture
Three models were implemented to benchmark performance:
1.  **Linear Regression (Baseline):** Establishes a simple linear baseline.
2.  **FCNN (Fully Connected Neural Network):** Captures non-linear relationships but lacks memory.
3.  **SimpleRNN:** utilizes recurrent units to explicitly model the **sequential order** of traffic patterns.

## 💻 Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow / Keras
* **Data Processing:** Pandas, NumPy, Scikit-learn
* **Visualization:** Matplotlib, Seaborn
* **Utilities:** `holidays` library

## 🚀 How to Run

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YourUsername/Traffic-Forecasting-RNN.git](https://github.com/YourUsername/Traffic-Forecasting-RNN.git)
    cd Traffic-Forecasting-RNN
    ```

2.  **Setup Dataset (Important!)**
    * This project requires the **Metro Interstate Traffic Volume** dataset.
    * **Step A:** Download the dataset from Kaggle: [Link to Dataset](https://www.kaggle.com/datasets/anshtanwar/metro-interstate-traffic-volume)
    * **Step B:** Create a folder named `data` in the same directory as the notebook.
    * **Step C:** Unzip the downloaded file and place `Metro_Interstate_Traffic_Volume.csv` inside the `data` folder.
    * *Directory Structure should look like this:*
        ```text
        Traffic-Forecasting-RNN/
        ├── CA6000_Traffic_Forecasting_Code.ipynb
        └── data/
            └── Metro_Interstate_Traffic_Volume.csv
        ```

3.  **Install Dependencies**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn tensorflow holidays
    ```

4.  **Run the Notebook**
    ```bash
    jupyter notebook CA6000_Traffic_Forecasting_Code.ipynb
    ```
    
## 📊 Results & Conclusion
* **Data Quality:** The data-centric approach (fixing holidays and weather anomalies) significantly reduced noise.
* **Model Performance:** The **SimpleRNN** model demonstrated superior ability to capture time-dependent trends (such as rush hour peaks) compared to the non-sequential Linear Regression and FCNN baselines.

---
*Author: Haocheng Wang | NTU CCDS MCAAI*
