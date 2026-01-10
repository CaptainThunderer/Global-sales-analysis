# 📊 Global Sales Analytics Dashboard (Tkinter Version)

## 📌 Project Overview

The **Global Sales Analytics Dashboard (Tkinter Version)** is a desktop-based data analytics application developed using **Python and Tkinter**. The application allows users to analyze historical sales data across multiple countries and products through an interactive graphical user interface. It provides visual insights, key performance indicators, and basic forecasting to support data-driven decision-making in an offline environment.

---

## 🎯 Objectives

* To develop a desktop-based sales analytics application using Tkinter
* To allow multi-selection of countries and products using checkboxes
* To analyze sales performance over a selected date range
* To visualize sales trends and comparisons using charts
* To provide a simple forecasting mechanism

---

## 🛠️ Features

* ✔ Country and product selection using checkboxes
* ✔ Date range filtering
* ✔ Key Performance Indicators (Total Sales, Total Profit, Average Discount)
* ✔ Monthly sales trend visualization
* ✔ Sales forecasting using Exponential Moving Average (EMA)
* ✔ Product-wise sales analysis
* ✔ Country vs Product sales heatmap
* ✔ Offline desktop execution

---

## 🧠 Technology Stack

### Programming Language

* Python

### Libraries & Frameworks

* **Tkinter** – Desktop graphical user interface
* **Pandas** – Data loading, filtering, and aggregation
* **NumPy** – Numerical computations
* **Matplotlib** – Chart plotting and embedding in Tkinter
* **Seaborn** – Heatmap and statistical visualizations

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

## ▶️ How to Run the Application

### 1️⃣ Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn
```

*(Tkinter is included by default with Python)*

---

### 2️⃣ Project Structure

```
tkinter_app/
│
├── app_tkinter.py
├── sales_data.csv
└── README_TKINTER.md
```

---

### 3️⃣ Run the Application

```bash
python app_tkinter.py
```

The desktop dashboard window will open.

---

## 🧪 Usage Instructions

1. Select countries and products using checkboxes
2. Enter the start and end dates in `YYYY-MM-DD` format
3. Click **Apply Filters**
4. View updated KPIs and charts
5. Modify selections to explore more insights

---

## 📈 Forecasting Technique

The application uses **Exponential Moving Average (EMA)** for basic sales forecasting. EMA gives higher weight to recent data points, making it suitable for short-term trend analysis.

---

## ✅ Advantages

* Fully offline desktop application
* Simple and intuitive user interface
* Lightweight and easy to deploy
* Suitable for academic and learning purposes

---

## 🔮 Future Enhancements

* Advanced forecasting models (ARIMA, Prophet)
* Export charts and reports to PDF/Excel
* Improved UI styling and themes
* Database connectivity

---

## 📄 License

This project is licensed under the **MIT License**.

Author: [CaptainThunderer](https://github.com/CaptainThunderer)

---

