# 📊 Global Sales Analytics Dashboard

## 📌 Project Overview

The **Global Sales Analytics Dashboard** is a web-based interactive data analytics application developed using **Streamlit**. It enables users to analyze historical sales data across multiple countries and products using dynamic filters, visualizations, and basic forecasting techniques. The dashboard helps in understanding sales trends, identifying top-performing products, and supporting data-driven business decisions.

---

## 🎯 Objectives

* To analyze sales data across different regions and products
* To provide interactive filtering based on country, product, and date range
* To visualize sales trends using charts and heatmaps
* To perform basic sales forecasting
* To present key performance indicators (KPIs) clearly

---

## 🛠️ Features

* 📌 Country and product-based filtering
* 📅 Date range selection for time-series analysis
* 📈 Monthly sales trend visualization
* 🔮 Sales forecasting using Exponential Moving Average (EMA)
* 📊 Product-wise sales comparison
* 🔥 Country vs Product heatmap
* 📋 Detailed summary report

---

## 🧠 Technology Stack

### Programming Language

* Python

### Libraries & Frameworks

* **Streamlit** – Web-based interactive dashboard
* **Pandas** – Data loading, cleaning, and aggregation
* **NumPy** – Numerical computations
* **Matplotlib** – Data visualization
* **Seaborn** – Advanced statistical visualizations

---

## 📂 Dataset Information

* Format: CSV file
* Time Period: 2020 – 2021
* Attributes include:

  * Order Date
  * Country
  * Product
  * Category
  * Sales
  * Profit
  * Discount
  * Quantity

---

## ▶️ How to Run the Project

### 1️⃣ Install Required Libraries

```bash
pip install streamlit pandas numpy matplotlib seaborn
```

### 2️⃣ Project Structure

```
Sales_Analytics_Dashboard/
│
├── app.py
├── sales_data.csv
└── README.md
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

The dashboard will open automatically in your web browser.

---

## 🧪 Usage Instructions

1. Select one or more countries and products using the filters
2. Choose the desired date range
3. View updated KPIs, charts, and forecasts dynamically
4. Modify filters to explore different insights

---

## 📈 Forecasting Technique

The dashboard uses **Exponential Moving Average (EMA)** for basic sales forecasting. EMA assigns greater weight to recent data points, making it suitable for short-term trend prediction.

---

## ✅ Advantages

* Interactive and user-friendly interface
* Real-time filtering and visualization
* Lightweight and easy to deploy
* No complex backend or database required

---

## 🔮 Future Enhancements

* Integration of advanced forecasting models (ARIMA, Prophet)
* Real-time database connectivity
* Export reports to PDF or Excel
* Role-based user authentication
* Currency normalization and inflation-adjusted analysis

---

## 👨‍🎓 Academic Relevance

This project demonstrates practical applications of:

* Data analytics
* Time-series analysis
* Data visualization
* Web-based dashboard development using Python

---

## 📄 License
MIT License

Copyright (c) 2026 CaptainThunderer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Author: CaptainThunderer

---
