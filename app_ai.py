import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import zscore
import plotly.express as px
from io import BytesIO
from statsmodels.tsa.arima.model import ARIMA  # For advanced forecasting
from sklearn.ensemble import RandomForestRegressor, IsolationForest  # For predictive modeling and anomaly detection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# ================= UI CONFIG =================
st.set_page_config(
    page_title="Global Sales Analytics Dashboard",
    layout="wide"
)

st.title("🌍 Global Sales Analytics Dashboard")
st.caption("AI-Powered Business Intelligence & Forecasting")

# ================= LOAD DATA =================
df = pd.read_csv("sales_data.csv")
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

# ================= SIDEBAR FILTERS =================
st.sidebar.header("🔎 Filter Options")

# --- Country filter (Select All default) ---
countries = sorted(df["Country"].unique())
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=countries,
    default=countries
)

# --- Product filter (Select All default) ---
products = sorted(df["Product"].unique())
selected_products = st.sidebar.multiselect(
    "Select Products",
    options=products,
    default=products
)

# --- Date range filter (SAFE) ---
min_date = df["Order_Date"].min()
max_date = df["Order_Date"].max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# --- Defensive date handling ---
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = min_date
    end_date = max_date

# ================= APPLY FILTERS =================
filtered_df = df[
    (df["Country"].isin(selected_countries)) &
    (df["Product"].isin(selected_products)) &
    (df["Order_Date"] >= pd.to_datetime(start_date)) &
    (df["Order_Date"] <= pd.to_datetime(end_date))
]

# ================= NO DATA SAFETY CHECK =================
if filtered_df.empty:
    st.warning("⚠️ No data available for the selected filters.")
    st.stop()

# ================= SESSION STATE FOR PERSONALIZATION =================
if "selected_countries_history" not in st.session_state:
    st.session_state.selected_countries_history = []
if "selected_products_history" not in st.session_state:
    st.session_state.selected_products_history = []

st.session_state.selected_countries_history.append(selected_countries)
st.session_state.selected_products_history.append(selected_products)

# ================= KPI SECTION =================
total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
avg_discount = filtered_df["Discount"].mean()

k1, k2, k3 = st.columns(3)
k1.metric("💰 Total Sales", f"₹{total_sales:,.0f}")
k2.metric("📈 Total Profit", f"₹{total_profit:,.0f}")
k3.metric("🏷 Avg Discount", f"{avg_discount:.2f}%")

st.divider()

# =================================================
# 📈 MONTHLY SALES TREND (ADAPTIVE)
# =================================================
st.subheader("📈 Monthly Sales Trend")

monthly_sales = (
    filtered_df
    .groupby("Month")["Sales"]
    .sum()
    .reset_index()
)

if len(monthly_sales) < 2:
    st.info("ℹ️ Not enough months for a trend line. Showing single-month sales.")
    fig1, ax1 = plt.subplots(figsize=(4, 3))
    ax1.bar(monthly_sales["Month"], monthly_sales["Sales"])
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Sales")
    plt.tight_layout()
    st.pyplot(fig1)
else:
    fig1, ax1 = plt.subplots(figsize=(5.5, 3))
    ax1.plot(monthly_sales["Month"], monthly_sales["Sales"], marker="o")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig1)

# =================================================
# 🔮 ADVANCED AI FORECASTING (ENHANCED WITH ARIMA)
# =================================================
st.subheader("🔮 AI Sales Forecast")

if len(monthly_sales) >= 5:  # ARIMA needs more data
    try:
        # Fit ARIMA model (p=1, d=1, q=1 as default; tune as needed)
        model = ARIMA(monthly_sales["Sales"], order=(1, 1, 1))
        model_fit = model.fit()
        forecast_steps = min(3, len(monthly_sales))  # Forecast next 3 months or less
        forecast = model_fit.forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start=monthly_sales["Month"].iloc[-1], periods=forecast_steps+1, freq='M')[1:]
        forecast_df = pd.DataFrame({"Month": forecast_index.strftime('%Y-%m'), "Forecast": forecast})
        
        fig2, ax2 = plt.subplots(figsize=(5.5, 3))
        ax2.plot(monthly_sales["Month"], monthly_sales["Sales"], label="Actual", marker="o")
        ax2.plot(forecast_df["Month"], forecast_df["Forecast"], linestyle="--", label="ARIMA Forecast", marker="x")
        ax2.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
        st.info(f"📊 Forecast Accuracy (MAE on historical data): {mean_absolute_error(monthly_sales['Sales'][:-forecast_steps], model_fit.fittedvalues[:-forecast_steps]):.2f}")
    except Exception as e:
        st.warning(f"⚠️ ARIMA forecasting failed: {str(e)}. Falling back to EMA.")
        monthly_sales["EMA_Forecast"] = monthly_sales["Sales"].ewm(span=3).mean()
        fig2, ax2 = plt.subplots(figsize=(5.5, 3))
        ax2.plot(monthly_sales["Month"], monthly_sales["Sales"], label="Actual", marker="o")
        ax2.plot(monthly_sales["Month"], monthly_sales["EMA_Forecast"], linestyle="--", label="EMA Forecast")
        ax2.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
else:
    st.warning("⚠️ Advanced forecasting requires at least 5 months of data.")

# =================================================
# 🚨 ENHANCED AI ANOMALY DETECTION (ISOLATION FOREST)
# =================================================
st.subheader("🚨 AI Anomaly Detection & Alerts")

if len(monthly_sales) >= 5:
    # Use Isolation Forest for anomaly detection
    iso_forest = IsolationForest(contamination=0.1, random_state=42)  # 10% expected anomalies
    monthly_sales["Anomaly_Score"] = iso_forest.fit_predict(monthly_sales[["Sales"]])
    anomalies = monthly_sales[monthly_sales["Anomaly_Score"] == -1]
    
    if not anomalies.empty:
        st.warning(f"🚨 Anomalies detected in: {', '.join(anomalies['Month'].tolist())}. AI suggests investigating market disruptions or data issues.")
        fig_anom, ax_anom = plt.subplots(figsize=(5.5, 3))
        ax_anom.plot(monthly_sales["Month"], monthly_sales["Sales"], marker="o", label="Sales")
        ax_anom.scatter(anomalies["Month"], anomalies["Sales"], color="red", label="Anomalies", s=100)
        ax_anom.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig_anom)
    else:
        st.success("✅ No anomalies detected by AI.")
else:
    st.info("ℹ️ AI anomaly detection requires at least 5 months of data.")

# =================================================
# 🤖 PREDICTIVE SALES MODELING (RANDOM FOREST REGRESSION)
# =================================================
st.subheader("🤖 Predictive Sales Modeling")

# Prepare data for modeling (encode categorical features)
le_country = LabelEncoder()
le_product = LabelEncoder()
filtered_df["Country_Encoded"] = le_country.fit_transform(filtered_df["Country"])
filtered_df["Product_Encoded"] = le_product.fit_transform(filtered_df["Product"])

features = ["Discount", "Country_Encoded", "Product_Encoded"]
target = "Sales"

if len(filtered_df) >= 10:  # Minimum data for training
    X = filtered_df[features]
    y = filtered_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    predictions = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    st.write(f"**Model Performance:** Mean Absolute Error (MAE): ₹{mae:.2f}")
    
    # Feature Importance
    importance = rf_model.feature_importances_
    fig_imp, ax_imp = plt.subplots(figsize=(4, 3))
    ax_imp.barh(features, importance)
    ax_imp.set_xlabel("Importance")
    ax_imp.set_title("Feature Importance for Sales Prediction")
    plt.tight_layout()
    st.pyplot(fig_imp)
    
    # Prediction Demo: Allow user to input values
    st.subheader("Predict Sales for New Scenario")
    user_discount = st.slider("Discount (%)", 0.0, 50.0, avg_discount)
    user_country = st.selectbox("Country", countries)
    user_product = st.selectbox("Product", products)
    
    user_input = pd.DataFrame({
        "Discount": [user_discount],
        "Country_Encoded": [le_country.transform([user_country])[0]],
        "Product_Encoded": [le_product.transform([user_product])[0]]
    })
    predicted_sales = rf_model.predict(user_input)[0]
    st.metric("Predicted Sales", f"₹{predicted_sales:,.0f}")
else:
    st.warning("⚠️ Predictive modeling requires at least 10 data points.")

# =================================================
# 🎯 DYNAMIC SCENARIO PLANNING & WHAT-IF ANALYSIS (WITH ML PREDICTIONS)
# =================================================
st.subheader("🎯 What-If Scenario Planning with ML Predictions")

discount_change = st.slider("Adjust Discount (%)", -50, 50, 0, help="Simulate discount changes to see ML-predicted impact on sales/profit.")
price_change = st.slider("Adjust Price (%)", -50, 50, 0, help="Simulate price changes (affects sales indirectly via discount).")

original_sales = total_sales
original_profit = total_profit

# Use ML model for projections if available
if len(filtered_df) >= 10 and 'rf_model' in locals():
    # Adjust discount for projection (simulate price change by scaling discount)
    adjusted_discount = avg_discount + discount_change  # New discount level
    adjusted_discount = max(0, min(adjusted_discount, 100))  # Clamp to 0-100%
    
    # Prepare input for ML prediction: Use filtered data with adjusted discount
    prediction_inputs = filtered_df.copy()
    prediction_inputs["Discount"] = adjusted_discount  # Apply adjusted discount to all rows
    
    # Predict sales for each row
    prediction_inputs["Predicted_Sales"] = rf_model.predict(prediction_inputs[features])
    
    # Aggregate predicted sales (average or sum across filtered data)
    projected_sales = prediction_inputs["Predicted_Sales"].sum()  # Or .mean() if per-unit
    
    # Simulate projected profit (assuming profit = sales - cost; adjust if you have cost data)
    # For simplicity, use original profit margin ratio
    profit_margin = original_profit / original_sales if original_sales > 0 else 0
    projected_profit = projected_sales * profit_margin * (1 + price_change / 100)  # Factor in price change
    
    st.info("📊 Projections powered by ML (Random Forest). Adjust sliders to see dynamic predictions.")
else:
    # Fallback to simple calculations if ML not available
    projected_sales = original_sales * (1 + price_change / 100) * (1 - discount_change / 100)
    projected_profit = original_profit * (1 + price_change / 100) * (1 - discount_change / 100)
    st.warning("⚠️ ML model not available (need >=10 data points). Using simple projections.")

col1, col2 = st.columns(2)
col1.metric("Projected Sales", f"₹{projected_sales:,.0f}", delta=f"{((projected_sales - original_sales) / original_sales * 100):.1f}%")
col2.metric("Projected Profit", f"₹{projected_profit:,.0f}", delta=f"{((projected_profit - original_profit) / original_profit * 100):.1f}%")

# --- Dynamic Graph for ML Predictions ---
st.subheader("📊 ML-Powered Scenario Impact Visualization")

# Check for valid data before plotting
if original_sales > 0 and original_profit > 0 and projected_sales >= 0 and projected_profit >= 0:
    # Prepare data for bar chart
    categories = ['Sales', 'Profit']
    original_values = [original_sales, original_profit]
    projected_values = [projected_sales, projected_profit]

    x = np.arange(len(categories))  # Label locations
    width = 0.35  # Bar width

    fig_scenario, ax_scenario = plt.subplots(figsize=(6, 4))
    bars1 = ax_scenario.bar(x - width/2, original_values, width, label='Original (CSV)', color='skyblue')
    bars2 = ax_scenario.bar(x + width/2, projected_values, width, label='ML Projected', color='orange')

    # Add labels and title
    ax_scenario.set_xlabel('Metrics')
    ax_scenario.set_ylabel('Amount (₹)')
    ax_scenario.set_title('Original vs. ML-Projected Sales & Profit')
    ax_scenario.set_xticks(x)
    ax_scenario.set_xticklabels(categories)
    ax_scenario.legend()

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax_scenario.text(bar.get_x() + bar.get_width()/2., height, f'₹{height:,.0f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax_scenario.text(bar.get_x() + bar.get_width()/2., height, f'₹{height:,.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig_scenario)
else:
    st.warning("⚠️ Invalid data for visualization (e.g., zero values). Check filters or data.")
# =================================================
# 📊 PRODUCT-WISE SALES
# =================================================
st.subheader("📊 Product-wise Sales")

product_sales = filtered_df.groupby("Product")["Sales"].sum()

if product_sales.empty:
    st.warning("⚠️ No product sales data available.")
else:
    fig3, ax3 = plt.subplots(figsize=(5.5, 3))
    product_sales.sort_values(ascending=False).plot(kind="bar", ax=ax3)
    ax3.set_xlabel("Product")
    ax3.set_ylabel("Sales")
    plt.tight_layout()
    st.pyplot(fig3)

# =================================================
# 🔥 COUNTRY vs PRODUCT HEATMAP
# =================================================
st.subheader("🔥 Country vs Product Heatmap")

pivot = pd.pivot_table(
    filtered_df,
    values="Sales",
    index="Country",
    columns="Product",
    aggfunc="sum"
)

if pivot.empty:
    st.warning("⚠️ Heatmap not available for selected filters.")
else:
    fig4, ax4 = plt.subplots(figsize=(6, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="coolwarm", ax=ax4)
    plt.tight_layout()
    st.pyplot(fig4)

# =================================================
# 🥧 SALES DISTRIBUTION (PIE CHART)
# =================================================
st.subheader("🥧 Sales Distribution")

pie_choice = st.radio(
    "View Sales Distribution By:",
    ["Country", "Product"],
    horizontal=True
)

if pie_choice == "Country":
    pie_data = filtered_df.groupby("Country")["Sales"].sum()
else:
    pie_data = filtered_df.groupby("Product")["Sales"].sum()

if pie_data.empty:
    st.warning("⚠️ Not enough data for pie chart.")
else:
    fig_pie, ax_pie = plt.subplots(figsize=(3, 3))
    ax_pie.pie(
        pie_data,
        labels=pie_data.index,
        autopct="%1.1f%%",
        startangle=140,
        radius=0.9,
        textprops={"fontsize": 8}
    )
    ax_pie.set_title(
        f"Sales Share by {pie_choice}",
        fontsize=10
    )
    plt.tight_layout()
    st.pyplot(fig_pie)

# =================================================
# 📊 COUNTRY vs PRODUCT (GROUPED BAR CHART)
# =================================================
st.subheader("📊 Country vs Product – Comparative Analysis")

pivot_bar = pd.pivot_table(
    filtered_df,
    values="Sales",
    index="Country",
    columns="Product",
    aggfunc="sum"
)

if pivot_bar.empty:
    st.warning("⚠️ No data available for comparison.")
else:
    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
    pivot_bar.plot(kind="bar", ax=ax_bar)
    ax_bar.set_xlabel("Country")
    ax_bar.set_ylabel("Sales")
    ax_bar.set_title("Country vs Product Sales Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_bar)

# =================================================
# 💡 ENHANCED AI RECOMMENDATIONS
# =================================================
st.subheader("💡 AI-Powered Recommendations")

recent_countries = list(set(st.session_state.selected_countries_history[-1])) if st.session_state.selected_countries_history else []
recent_products = list(set(st.session_state.selected_products_history[-1])) if st.session_state.selected_products_history else []

# Correlation-based suggestions
correlation = filtered_df[["Sales", "Discount"]].corr().iloc[0, 1]
if abs(correlation) > 0.5:
    st.info(f"🤖 AI Insight: Sales and Discount are {'positively' if correlation > 0 else 'negatively'} correlated ({correlation:.2f}). Consider adjusting discounts for better performance.")

if recent_countries:
    st.info(f"Based on your selections, AI recommends comparing {', '.join(recent_countries)} with underperforming regions for growth opportunities.")
if recent_products:
    st.info(f"You've explored {', '.join(recent_products)}—AI suggests checking anomaly detection for these products.")

# =================================================
# 📄 AUTOMATED REPORT GENERATION & EXPORT
# =================================================
st.subheader("📄 Generate & Export Report")

if st.button("Generate PDF Report"):
    report_content = f"""
    Global Sales Analytics Report
    =============================