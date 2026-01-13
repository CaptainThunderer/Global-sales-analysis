# 📊 Global Sales Analytics Dashboard(AI-Prediction)

## 📌 Project Overview

The **Global Sales Analytics Dashboard** is a web-based interactive data analytics application developed using **Python and Streamlit**.  
It enables users to explore, analyze, and visualize historical sales data across multiple countries and products using dynamic filters, interactive charts, forecasting techniques, and scenario-based analysis.

The dashboard supports **data-driven business decision-making** by presenting key performance indicators (KPIs), sales trends, forecasting insights, and downloadable analytical reports.

---

## 🎯 Objectives

- Analyze sales performance across countries and products  
- Provide interactive filtering using country, product, and date range  
- Visualize sales trends and comparisons through charts and heatmaps  
- Perform time-series sales forecasting  
- Enable what-if scenario planning for pricing and discount changes  
- Generate and display a detailed analytical report  

---

## 🛠️ Key Features

### 🔎 Interactive Filtering
- Country-wise filtering  
- Product-wise filtering  
- Date range selection  

### 📊 Analytics & Visualization
- KPI cards for **Total Sales, Total Profit, and Average Discount**
- Monthly sales trend line chart
- Product-wise sales bar chart
- Country vs Product heatmap
- Sales distribution using pie charts
- Country vs Product comparative bar chart

### 🔮 Sales Forecasting
- Time-series forecasting using **ARIMA**
- Forecast visualization alongside historical monthly sales
- Adaptive behavior based on available data

### 🎯 What-If Scenario Planning
- Interactive sliders to simulate:
  - Discount changes
  - Price changes
- Real-time projection of:
  - Sales impact
  - Profit impact
- Stable scenario modeling for all filter combinations

### 📄 AI Sales Report
- Dynamically generated **filter-based report**
- Displays directly on the dashboard
- Downloadable report in **TXT format**

---

## 🧠 Technology Stack

### Programming Language
- **Python**

### Libraries & Frameworks
- **Streamlit** – Interactive web dashboard
- **Pandas** – Data processing and aggregation
- **NumPy** – Numerical computations
- **Matplotlib** – Chart visualizations
- **Seaborn** – Statistical visualizations
- **Scikit-learn** – Predictive modeling
- **Statsmodels** – Time-series forecasting (ARIMA)

---

## 📂 Dataset Information

- Format: CSV
- Time Period: **2020 – 2021**
- Attributes:
  - Order Date
  - Country
  - Product
  - Category
  - Sales
  - Profit
  - Discount
  - Quantity

The dataset is dynamically filtered and aggregated based on user selections.

---

## ▶️ How to Run the Project

### 1️⃣ Install Required Libraries

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn statsmodels
````

### 2️⃣ Project Structure

```
Sales_Analytics_Dashboard/
│
├── app_ai.py
├── sales_data.csv
├── README.md
└── requirements.txt
```

### 3️⃣ Run the Application

```bash
streamlit run app_ai.py
```

The dashboard will open automatically in your web browser.

---

## 🧪 Usage Instructions

1. Select countries and products using the sidebar filters
2. Choose a date range for analysis
3. View KPIs, charts, and sales trends dynamically
4. Analyze future sales using forecasting
5. Adjust discount and price sliders for scenario planning
6. View and download the analytical report

---

## 📈 Forecasting Method

The dashboard uses **ARIMA (AutoRegressive Integrated Moving Average)** for monthly sales forecasting.
ARIMA captures historical trends and patterns in time-series data to provide short-term sales predictions.

---

## ✅ Advantages

* User-friendly and interactive interface
* Real-time data filtering and visualization
* Robust handling of sparse and filtered data
* No external database required
* Suitable for academic and portfolio projects

---

## 🔮 Future Enhancements

* Confidence intervals for forecasts
* PDF and Excel report exports
* Seasonal forecasting models (SARIMA, Prophet)
* Cloud deployment (Streamlit Cloud)
* User authentication and role-based access

---

## 👨‍🎓 Academic Relevance

This project demonstrates practical application of:

* Data analytics and visualization
* Time-series analysis
* Scenario-based business analysis
* Python-based dashboard development

Suitable for:

* Data Analytics projects
* Python mini / major projects
* Business Intelligence demonstrations

---

## 📄 License

MIT License

Copyright (c) 2026 CaptainThunderer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

---

## 👤 Author

**CaptainThunderer**
GitHub: [https://github.com/CaptainThunderer](https://github.com/CaptainThunderer)

---
