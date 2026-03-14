# 🌍 Global Sales Analytics Dashboard

An **interactive business intelligence dashboard** for analyzing global sales data, built using **Dash, Plotly, and Python**.
The system allows users to upload a CSV dataset, apply dynamic filters, visualize trends, detect anomalies, forecast future sales, and generate downloadable business reports.

---

# 📊 Features

### 📂 CSV Upload

Users can upload their own sales dataset in CSV format.
The system automatically validates the schema and loads the data for analysis.

Required dataset columns:

* `Order_Date`
* `Country`
* `Product`
* `Sales`
* `Profit`
* `Discount`

---

### 🔎 Dynamic Filtering

The dashboard supports interactive filters:

* Country filter
* Product filter
* Date range selection
* Forecast horizon selection (6 or 12 months)

All charts update instantly based on the selected filters.

---

### 📈 Monthly Sales Trend

Visualizes sales performance over time using interactive Plotly line charts.

Features:

* Smooth line visualization
* Hover analytics
* Optional anomaly markers

---

### 🔮 Sales Forecasting

The system implements **Holt-Winters Double Exponential Smoothing** for forecasting future sales trends.

Forecast parameters:

* Level smoothing factor (α) = 0.3
* Trend smoothing factor (β) = 0.1

Users can forecast sales for:

* 6 months
* 12 months

---

### 🚨 Anomaly Detection

The dashboard detects unusual sales patterns using a **rolling Z-score method**.

Steps:

1. Compute rolling mean and standard deviation
2. Calculate Z-score
3. Flag anomalies where:

[
|Z| > 2.5
]

Anomalies are highlighted in:

* the sales trend chart
* a dedicated anomaly timeline

---

### 📊 KPI Indicators

Key business metrics are displayed at the top of the dashboard:

* Total Sales
* Total Profit
* Average Discount
* Number of Detected Anomalies

---

### 📉 Product Sales Analysis

Bar charts visualize the performance of different products.

This helps identify:

* top selling products
* low performing categories

---

### 🔥 Country vs Product Heatmap

A heatmap visualizes the relationship between countries and product sales.

Special features:

* Dynamic text color based on WCAG luminance calculation
* Accessibility-aware visualization
* Adaptive heatmap height for large datasets

---

### 🥧 Sales Distribution

Interactive pie chart displaying sales share by:

* Country
* Product

Users can toggle the visualization mode dynamically.

---

### 📊 Country vs Product Comparison

Grouped bar charts allow comparison of product performance across different countries.

---

### 📄 Business Summary Report

The dashboard automatically generates a summarized report including:

* total sales
* total profit
* top country
* top product
* timestamp

The report can be downloaded as a `.txt` file.

---

# 🧠 Technologies Used

### Programming Language

* Python

### Frameworks

* Dash
* Dash Bootstrap Components

### Data Analysis

* Pandas
* NumPy
* SciPy

### Visualization

* Plotly
* Plotly Express
* Plotly Graph Objects

### Styling

* Custom CSS animations
* Glassmorphism UI design

---

# 📂 Project Structure

```
Global-sales-analysis
│
├── new.py
├── README.md
├── global_sales_daily_updated.csv
│
└── assets/
    ├── animations.css
    ├── background.css
    ├── drag_drop.js
    ├── liquid_glass.css
    └── style.css
```

The **assets folder** contains custom styles, animations, and UI effects.

---

# ▶️ How to Run the Project

### 1️⃣ Install dependencies

```bash
pip install dash dash-bootstrap-components plotly pandas numpy scipy
```

---

### 2️⃣ Run the application

```bash
python new.py
```

---

### 3️⃣ Open in browser

```
http://127.0.0.1:8050
```

---

# 📊 Example Workflow

1. Upload a CSV dataset
2. Select countries and products
3. Adjust date range filters
4. Analyze sales trends
5. View anomaly detection
6. Forecast future sales
7. Download the business report

---

# 📈 Key Capabilities

✔ Interactive business intelligence dashboard
✔ CSV-driven analytics system
✔ Built-in anomaly detection
✔ Time-series forecasting
✔ Accessible data visualizations
✔ Dynamic filtering system
✔ Downloadable analytical reports

---

# 🎯 Use Cases

This dashboard can be used for:

* sales performance monitoring
* business intelligence analysis
* trend forecasting
* anomaly detection in revenue streams
* strategic decision making

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

Author: [CaptainThunderer](https://github.com/CaptainThunderer)

---
