import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ================= LOAD DATA =================
df = pd.read_csv("sales_data.csv")
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

# ================= MAIN WINDOW =================
root = tk.Tk()
root.title("Global Sales Analytics Dashboard")
root.geometry("1250x850")

# ================= FILTER FRAME =================
filter_frame = ttk.LabelFrame(root, text="Filters")
filter_frame.pack(fill="x", padx=10, pady=5)

# ================= COUNTRY CHECKBOXES =================
ttk.Label(filter_frame, text="Countries").grid(row=0, column=0, padx=10, sticky="w")

countries = sorted(df["Country"].unique())
country_vars = {}

for i, c in enumerate(countries):
    var = tk.BooleanVar(value=True)
    chk = ttk.Checkbutton(filter_frame, text=c, variable=var)
    chk.grid(row=i+1, column=0, sticky="w")
    country_vars[c] = var

# ================= PRODUCT CHECKBOXES =================
ttk.Label(filter_frame, text="Products").grid(row=0, column=1, padx=10, sticky="w")

products = sorted(df["Product"].unique())
product_vars = {}

for i, p in enumerate(products):
    var = tk.BooleanVar(value=True)
    chk = ttk.Checkbutton(filter_frame, text=p, variable=var)
    chk.grid(row=i+1, column=1, sticky="w")
    product_vars[p] = var

# ================= DATE INPUTS =================
ttk.Label(filter_frame, text="Start Date (YYYY-MM-DD)").grid(row=0, column=2, padx=10)
start_entry = ttk.Entry(filter_frame)
start_entry.insert(0, df["Order_Date"].min().strftime("%Y-%m-%d"))
start_entry.grid(row=1, column=2, padx=10)

ttk.Label(filter_frame, text="End Date (YYYY-MM-DD)").grid(row=0, column=3, padx=10)
end_entry = ttk.Entry(filter_frame)
end_entry.insert(0, df["Order_Date"].max().strftime("%Y-%m-%d"))
end_entry.grid(row=1, column=3, padx=10)

# ================= KPI FRAME =================
kpi_frame = ttk.Frame(root)
kpi_frame.pack(fill="x", padx=10, pady=5)

kpi_sales = ttk.Label(kpi_frame, text="Total Sales: ₹0", font=("Arial", 12, "bold"))
kpi_profit = ttk.Label(kpi_frame, text="Total Profit: ₹0", font=("Arial", 12, "bold"))
kpi_discount = ttk.Label(kpi_frame, text="Avg Discount: 0%", font=("Arial", 12, "bold"))

kpi_sales.pack(side="left", padx=30)
kpi_profit.pack(side="left", padx=30)
kpi_discount.pack(side="left", padx=30)

# ================= CHART FRAME =================
chart_frame = ttk.Frame(root)
chart_frame.pack(fill="both", expand=True, padx=10, pady=10)

canvas_widgets = []

def clear_charts():
    for c in canvas_widgets:
        c.get_tk_widget().destroy()
    canvas_widgets.clear()

# ================= APPLY FILTERS =================
def apply_filters():
    clear_charts()

    try:
        start_date = pd.to_datetime(start_entry.get())
        end_date = pd.to_datetime(end_entry.get())
    except:
        messagebox.showerror("Invalid Date", "Use YYYY-MM-DD format")
        return

    selected_countries = [c for c, v in country_vars.items() if v.get()]
    selected_products = [p for p, v in product_vars.items() if v.get()]

    if not selected_countries or not selected_products:
        messagebox.showwarning("Selection Error", "Select at least one country and one product")
        return

    filtered = df[
        (df["Country"].isin(selected_countries)) &
        (df["Product"].isin(selected_products)) &
        (df["Order_Date"] >= start_date) &
        (df["Order_Date"] <= end_date)
    ]

    if filtered.empty:
        messagebox.showwarning("No Data", "No data available for selected filters")
        return

    # ================= KPIs =================
    kpi_sales.config(text=f"Total Sales: ₹{filtered['Sales'].sum():,.0f}")
    kpi_profit.config(text=f"Total Profit: ₹{filtered['Profit'].sum():,.0f}")
    kpi_discount.config(text=f"Avg Discount: {filtered['Discount'].mean():.2f}%")

    # ================= MONTHLY SALES =================
    monthly = filtered.groupby("Month")["Sales"].sum().reset_index()

    fig1, ax1 = plt.subplots(figsize=(4, 3))
    if len(monthly) < 2:
        ax1.bar(monthly["Month"], monthly["Sales"])
    else:
        ax1.plot(monthly["Month"], monthly["Sales"], marker="o")
        ax1.tick_params(axis="x", rotation=45)

    ax1.set_title("Monthly Sales Trend")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Sales")

    canvas1 = FigureCanvasTkAgg(fig1, master=chart_frame)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side="left", padx=5)
    canvas_widgets.append(canvas1)

    # ================= FORECAST =================
    if len(monthly) >= 3:
        monthly["EMA"] = monthly["Sales"].ewm(span=3).mean()

        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.plot(monthly["Month"], monthly["Sales"], label="Actual")
        ax2.plot(monthly["Month"], monthly["EMA"], linestyle="--", label="Forecast")
        ax2.legend()
        ax2.tick_params(axis="x", rotation=45)
        ax2.set_title("Sales Forecast")

        canvas2 = FigureCanvasTkAgg(fig2, master=chart_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side="left", padx=5)
        canvas_widgets.append(canvas2)

    # ================= PRODUCT SALES =================
    prod_sales = filtered.groupby("Product")["Sales"].sum()

    fig3, ax3 = plt.subplots(figsize=(4, 3))
    prod_sales.sort_values().plot(kind="barh", ax=ax3)
    ax3.set_title("Product-wise Sales")

    canvas3 = FigureCanvasTkAgg(fig3, master=chart_frame)
    canvas3.draw()
    canvas3.get_tk_widget().pack(side="left", padx=5)
    canvas_widgets.append(canvas3)

    # ================= HEATMAP =================
    pivot = pd.pivot_table(
        filtered,
        values="Sales",
        index="Country",
        columns="Product",
        aggfunc="sum"
    )

    fig4, ax4 = plt.subplots(figsize=(4, 3))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="coolwarm", ax=ax4)
    ax4.set_title("Country vs Product")

    canvas4 = FigureCanvasTkAgg(fig4, master=chart_frame)
    canvas4.draw()
    canvas4.get_tk_widget().pack(side="left", padx=5)
    canvas_widgets.append(canvas4)

# ================= APPLY BUTTON =================
apply_btn = ttk.Button(filter_frame, text="Apply Filters", command=apply_filters)
apply_btn.grid(row=1, column=4, padx=20)

# ================= START APP =================
root.mainloop()
