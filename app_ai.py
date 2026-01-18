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
# 🔮 AI SALES FORECAST (100% SAFE VERSION)
# =================================================
# =================================================
# 🔮 AI SALES FORECAST (ARIMA – CLEAN & RELIABLE)
# =================================================
st.subheader("🔮 AI Sales Forecast (ARIMA)")

try:
    # --- Prepare monthly time series ---
    ts_df = filtered_df.copy()
    ts_df["Order_Date"] = pd.to_datetime(ts_df["Order_Date"])

    ts_monthly = (
        ts_df
        .set_index("Order_Date")
        .resample("M")["Sales"]
        .sum()
    )

    if len(ts_monthly) < 8:
        st.warning("⚠️ At least 8 months of data required for ARIMA forecasting.")
    else:
        # --- Train ARIMA model ---
        model = ARIMA(ts_monthly, order=(1, 1, 1))
        model_fit = model.fit()

        # --- 1️⃣ In-sample fitted values (SKIP first unstable point) ---
        fitted_values = model_fit.predict(
            start=ts_monthly.index[1],
            end=ts_monthly.index[-1],
            typ="levels"
        )

        # --- 2️⃣ Future forecast (NEXT 6 MONTHS) ---
        forecast_steps = 6
        future_forecast = model_fit.forecast(steps=forecast_steps)

        future_dates = pd.date_range(
            start=ts_monthly.index[-1] + pd.offsets.MonthEnd(1),
            periods=forecast_steps,
            freq="M"
        )

        # --- Combine fitted + future forecast ---
        full_forecast = pd.concat(
            [
                fitted_values,
                pd.Series(future_forecast.values, index=future_dates)
            ]
        )

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(7.5, 4))

        ax.plot(
            ts_monthly.index,
            ts_monthly.values,
            marker="o",
            label="Actual"
        )

        ax.plot(
            full_forecast.index,
            full_forecast.values,
            linestyle="--",
            marker="x",
            label="Forecast"
        )

        ax.set_xlabel("Month")
        ax.set_ylabel("Sales")
        ax.set_title("Monthly Sales Forecast (ARIMA)")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)

        # --- Reliability Metric (MAE) ---
        mae = mean_absolute_error(
            ts_monthly.loc[fitted_values.index],
            fitted_values
        )

        st.info(f"📏 Forecast Reliability (MAE on historical data): ₹{mae:,.2f}")

except Exception as e:
    st.error(f"❌ ARIMA forecast failed: {e}")

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
# 🎯 DYNAMIC SCENARIO PLANNING & WHAT-IF ANALYSIS (WITH ML PREDICTIONS)
# =================================================
st.subheader("🎯 What-If Scenario Planning with Predictions")

discount_change = st.slider(
    "Adjust Discount (%)",
    -50, 50, 0
)

price_change = st.slider(
    "Adjust Price (%)",
    -50, 50, 0
)

# Base values
original_sales = total_sales
original_profit = total_profit

# ================= ALWAYS-ON SCENARIO LOGIC =================
# Impact assumptions (simple + stable)
# Positive demand response
discount_impact = 1 + (discount_change / 100) * 0.4
price_impact = 1 - (price_change / 100) * 0.6

# Clamp to avoid unrealistic explosions
discount_impact = max(0.7, discount_impact)
price_impact = max(0.6, price_impact)


projected_sales = original_sales * discount_impact * price_impact

profit_margin = original_profit / original_sales if original_sales > 0 else 0
projected_profit = projected_sales * profit_margin

# ================= DISPLAY RESULTS =================
col1, col2 = st.columns(2)

col1.metric(
    "Projected Sales",
    f"₹{projected_sales:,.0f}",
    delta=f"{((projected_sales - original_sales) / original_sales * 100):.1f}%"
)

col2.metric(
    "Projected Profit",
    f"₹{projected_profit:,.0f}",
    delta=f"{((projected_profit - original_profit) / original_profit * 100):.1f}%"
)


# --- Dynamic Graph for ML Predictions ---
st.subheader("📊Scenario Impact Visualization")

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
    bars2 = ax_scenario.bar(x + width/2, projected_values, width, label='Projected', color='orange')

    # Add labels and title
    ax_scenario.set_xlabel('Metrics')
    ax_scenario.set_ylabel('Amount (₹)')
    ax_scenario.set_title('Original vs. Projected Sales & Profit')
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
    st.success("📊 Scenario projections applied based on pricing & discount impact model")
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

recent_countries = (
    list(set(st.session_state.selected_countries_history[-1]))
    if st.session_state.selected_countries_history else []
)

recent_products = (
    list(set(st.session_state.selected_products_history[-1]))
    if st.session_state.selected_products_history else []
)

recommendations = []

# --- Country-based recommendation ---
if not filtered_df.empty:
    top_country = (
        filtered_df.groupby("Country")["Sales"]
        .sum()
        .idxmax()
    )
    recommendations.append(
        f"🌍 **{top_country}** is your strongest market. Consider increasing inventory or targeted marketing here."
    )

# --- Product-based recommendation ---
top_product = (
    filtered_df.groupby("Product")["Sales"]
    .sum()
    .idxmax()
)
recommendations.append(
    f"📦 **{top_product}** is the top-performing product. Bundling or premium pricing may increase profit."
)

# --- Discount optimization ---
if avg_discount > 30:
    recommendations.append(
        "🏷 High average discounts detected. Reducing discounts slightly could improve profit margins."
    )
else:
    recommendations.append(
        "✅ Discount levels are healthy. Focus on volume growth instead of price cuts."
    )

# --- Anomaly-based advice ---
if 'anomalies' in locals() and not anomalies.empty:
    recommendations.append(
        "🚨 Sales anomalies detected. Investigate supply chain issues, seasonal effects, or data inconsistencies."
    )

# --- Forecast-based guidance ---
if len(monthly_sales) >= 5:
    recommendations.append(
        "🔮 Forecast indicates future demand trends. Align procurement and staffing accordingly."
    )

# --- Display recommendations ---
for rec in recommendations:
    st.success(rec)
# =================================================
# 📄 DETAILED FILTER-AWARE AI REPORT CONTENT
# =================================================

selected_country_text = ", ".join(selected_countries) if selected_countries else "All"
selected_product_text = ", ".join(selected_products) if selected_products else "All"

date_range_text = f"{start_date} to {end_date}"

monthly_summary = (
    monthly_sales
    .sort_values("Sales", ascending=False)
    .head(3)
)

monthly_summary_text = "\n".join(
    [f"- {row['Month']}: ₹{row['Sales']:,.0f}" for _, row in monthly_summary.iterrows()]
)

anomaly_text = (
    ", ".join(anomalies["Month"].tolist())
    if 'anomalies' in locals() and not anomalies.empty
    else "No anomalies detected"
)

ml_status = (
    f"Random Forest Model Trained | MAE: ₹{mae:.2f}"
    if 'mae' in locals()
    else "ML model not trained (insufficient data)"
)

forecast_status = (
    "ARIMA forecasting applied"
    if len(monthly_sales) >= 5
    else "Forecasting skipped (insufficient data)"
)

report_content = f"""
📊 GLOBAL SALES ANALYTICS REPORT
=================================

🔎 FILTERS APPLIED
------------------
Countries Selected : {selected_country_text}
Products Selected  : {selected_product_text}
Date Range         : {date_range_text}

📈 KEY PERFORMANCE INDICATORS
-----------------------------
Total Sales        : ₹{total_sales:,.0f}
Total Profit       : ₹{total_profit:,.0f}
Average Discount   : {avg_discount:.2f}%

🏆 TOP CONTRIBUTORS
--------------------
Top Country        : {top_country if 'top_country' in locals() else 'N/A'}
Top Product        : {top_product if 'top_product' in locals() else 'N/A'}

📅 TOP MONTHS BY SALES
----------------------
{monthly_summary_text}

🚨 ANOMALY INSIGHTS
-------------------
{anomaly_text}

🤖 AI & ML INSIGHTS
-------------------
Forecasting        : {forecast_status}
Predictive Model   : {ml_status}

🎯 STRATEGIC INSIGHTS
---------------------
• Focus marketing on high-performing countries and products
• Optimize discount strategy to balance volume and profit
• Monitor anomaly periods for operational risks
• Use ML projections for pricing and demand planning

Generated on: {pd.Timestamp.now().strftime('%d-%m-%Y %H:%M:%S')}
"""
# =================================================
# 📄 AI SALES REPORT (ON-DASHBOARD VIEW)
# =================================================
st.subheader("📄 AI Sales Report Summary")

with st.expander("📊 Click to View Detailed Report", expanded=False):
    st.text(report_content)

# =================================================
# 📥 DOWNLOAD DETAILED AI REPORT
# =================================================
st.subheader("📥 Download Detailed AI Sales Report")

report_buffer = BytesIO()
report_buffer.write(report_content.encode("utf-8"))
report_buffer.seek(0)

st.download_button(
    label="⬇ Download Filter-Based AI Report (TXT)",
    data=report_buffer,
    file_name="Detailed_AI_Sales_Report.txt",
    mime="text/plain"
)
