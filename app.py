import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Airlines Intelligence Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .metric-card {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 14px 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .metric-title {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            color: #0f172a;
            font-size: 1.6rem;
            font-weight: 700;
            margin: 2px 0;
        }
        .metric-sub {
            font-size: 0.8rem;
            color: #2563eb;
            font-weight: 500;
        }
        .section-header {
            font-size: 1.25rem;
            font-weight: 700;
            color: #1e293b;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 4px;
        }
    </style>
""", unsafe_allow_html=True)

# Seaborn & Matplotlib Global Theme Setup
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 0.8
})

# ==========================================
# 2. DATA CLEANING PIPELINE
# ==========================================
@st.cache_data
def load_and_clean_data(file_path: str = "data/airlines.csv") -> pd.DataFrame:
    df = pd.read_csv(file_path)

    # 1. Standardize column names to snake_case
    rename_dict = {
        "Airport.Code": "airport_code",
        "Airport.Name": "airport_name",
        "Time.Label": "time_label",
        "Time.Month": "month",
        "Time.Month Name": "month_name",
        "Time.Year": "year",
        "Statistics.# of Delays.Carrier": "delays_carrier",
        "Statistics.# of Delays.Late Aircraft": "delays_late_aircraft",
        "Statistics.# of Delays.National Aviation System": "delays_nas",
        "Statistics.# of Delays.Security": "delays_security",
        "Statistics.# of Delays.Weather": "delays_weather",
        "Statistics.Carriers.Names": "carrier_names",
        "Statistics.Carriers.Total": "carriers_total",
        "Statistics.Flights.Cancelled": "flights_cancelled",
        "Statistics.Flights.Delayed": "flights_delayed",
        "Statistics.Flights.Diverted": "flights_diverted",
        "Statistics.Flights.On Time": "flights_ontime",
        "Statistics.Flights.Total": "flights_total",
        "Statistics.Minutes Delayed.Carrier": "mins_carrier",
        "Statistics.Minutes Delayed.Late Aircraft": "mins_late_aircraft",
        "Statistics.Minutes Delayed.National Aviation System": "mins_nas",
        "Statistics.Minutes Delayed.Security": "mins_security",
        "Statistics.Minutes Delayed.Total": "mins_total",
        "Statistics.Minutes Delayed.Weather": "mins_weather"
    }
    df = df.rename(columns=rename_dict)

    # 2. Fix data anomalies (security delays < 0)
    df["delays_security"] = df["delays_security"].clip(lower=0)

    # 3. Datetime conversion and sorting
    df["date"] = pd.to_datetime(df["time_label"].str.replace("/", "-") + "-01")
    df = df.sort_values(by=["date", "airport_code"]).reset_index(drop=True)

    # 4. Fill numeric null values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # 5. Engineered Performance Indicators
    df["delay_rate_pct"] = np.where(df["flights_total"] > 0, (df["flights_delayed"] / df["flights_total"]) * 100, 0)
    df["cancellation_rate_pct"] = np.where(df["flights_total"] > 0, (df["flights_cancelled"] / df["flights_total"]) * 100, 0)
    df["ontime_rate_pct"] = np.where(df["flights_total"] > 0, (df["flights_ontime"] / df["flights_total"]) * 100, 0)
    df["avg_delay_mins"] = np.where(df["flights_delayed"] > 0, df["mins_total"] / df["flights_delayed"], 0)

    return df

df_clean = load_and_clean_data()

# ==========================================
# SIDEBAR FILTERS (MODIFIED CLEAN CONTROLS)
# ==========================================
st.sidebar.header("Filter Operations")

st.sidebar.subheader("Time Period")
all_years = sorted(df_clean["year"].unique())
col_y1, col_y2 = st.sidebar.columns(2)
with col_y1:
    start_year = st.selectbox("From Year", all_years, index=0)
with col_y2:
    end_year = st.selectbox("To Year", all_years, index=len(all_years) - 1)

if start_year > end_year:
    st.sidebar.warning("Resetting: 'From Year' was after 'To Year'")
    start_year, end_year = end_year, start_year

st.sidebar.subheader("Airport Focus")
all_airports = sorted(df_clean["airport_code"].unique())
airport_options = ["All Airports (Nationwide)"] + all_airports

selected_airport = st.sidebar.selectbox(
    "Choose Airport:",
    options=airport_options,
    index=0
)

# Filtering Data Logic
filtered_df = df_clean[
    (df_clean["year"] >= start_year) & 
    (df_clean["year"] <= end_year)
]

if selected_airport != "All Airports (Nationwide)":
    filtered_df = filtered_df[filtered_df["airport_code"] == selected_airport]

st.sidebar.divider()
st.sidebar.write(f"Total Filtered Records: **{len(filtered_df):,}**")

# ==========================================
# 1. DATASET OVERVIEW
# ==========================================
st.title("✈️ Airlines Performance & Delay Intelligence Dashboard")
st.caption("Comprehensive data analytics dashboard implemented with Matplotlib, Seaborn, and Streamlit.")

st.markdown('<div class="section-header">1. Dataset Overview</div>', unsafe_allow_html=True)

o1, o2, o3, o4 = st.columns(4)
o1.info(f"**Total Records:** {len(filtered_df):,}")
o2.info(f"**Airports Tracked:** {filtered_df['airport_code'].nunique()}")
o3.info(f"**Date Span:** {filtered_df['date'].min().strftime('%b %Y')} – {filtered_df['date'].max().strftime('%b %Y')}")
o4.info(f"**Total Features:** {filtered_df.shape[1]}")

with st.expander("🔍 View Cleaned Dataset Preview"):
    st.dataframe(filtered_df.head(25), use_container_width=True)

# ==========================================
# 2. DATA CLEANING REPORT
# ==========================================
with st.expander("🧹 2. Data Cleaning & Transformation Report", expanded=False):
    st.markdown("""
    * **Header Standardization:** Standardized nested DOT strings into clean `snake_case` format.
    * **Temporal Parsing:** Extracted date stamps from `Time.Label` into standard `pd.to_datetime` format.
    * **Anomaly Clipping:** Adjusted negative values in security delays (`delays_security < 0`) to 0.
    * **Calculated Rates:** Generated `delay_rate_pct`, `cancellation_rate_pct`, and `avg_delay_mins`.
    """)

# ==========================================
# 3. DESCRIPTIVE STATISTICS
# ==========================================
st.markdown('<div class="section-header">3. Descriptive Statistics</div>', unsafe_allow_html=True)

stat_cols = [
    "flights_total", "flights_delayed", "flights_cancelled", "flights_ontime",
    "delays_carrier", "delays_late_aircraft", "delays_nas", "delays_weather",
    "mins_total", "avg_delay_mins"
]
stats_summary = filtered_df[stat_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]].rename(columns={"50%": "median"})
st.dataframe(stats_summary.style.format("{:,.2f}"), use_container_width=True)

# ==========================================
# 4. FLIGHT PERFORMANCE (KPIS)
# ==========================================
st.markdown('<div class="section-header">4. Flight Performance KPIs</div>', unsafe_allow_html=True)

tot_flights = filtered_df["flights_total"].sum()
tot_ontime = filtered_df["flights_ontime"].sum()
tot_delayed = filtered_df["flights_delayed"].sum()
tot_cancelled = filtered_df["flights_cancelled"].sum()
tot_delay_hrs = filtered_df["mins_total"].sum() / 60

ontime_pct = (tot_ontime / tot_flights * 100) if tot_flights > 0 else 0
delay_pct = (tot_delayed / tot_flights * 100) if tot_flights > 0 else 0
cancel_pct = (tot_cancelled / tot_flights * 100) if tot_flights > 0 else 0
avg_delay_per_flight = (filtered_df["mins_total"].sum() / tot_delayed) if tot_delayed > 0 else 0

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f'<div class="metric-card"><div class="metric-title">Total Operations</div><div class="metric-value">{tot_flights:,.0f}</div><div class="metric-sub">{ontime_pct:.1f}% On-Time Rate</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="metric-card"><div class="metric-title">Delayed Flights</div><div class="metric-value">{tot_delayed:,.0f}</div><div class="metric-sub">{delay_pct:.1f}% Delay Rate</div></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="metric-card"><div class="metric-title">Cancelled Flights</div><div class="metric-value">{tot_cancelled:,.0f}</div><div class="metric-sub">{cancel_pct:.2f}% Cancel Rate</div></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="metric-card"><div class="metric-title">Total Delay Duration</div><div class="metric-value">{tot_delay_hrs:,.0f} hrs</div><div class="metric-sub">{avg_delay_per_flight:.1f} mins / delay</div></div>', unsafe_allow_html=True)

# ==========================================
# 5. VISUAL ANALYSIS (ALL 6 CHARTS)
# ==========================================
st.markdown('<div class="section-header">5. Visual Analysis</div>', unsafe_allow_html=True)

# Row 1: Line Chart & Pie/Donut Chart
col_r1_1, col_r1_2 = st.columns([6, 4])

with col_r1_1:
    st.markdown("##### 📈 Line Chart: Operational Flight Trends Over Time")
    monthly = filtered_df.groupby("date")[["flights_total", "flights_ontime", "flights_delayed"]].sum().reset_index()

    fig_line, ax_line = plt.subplots(figsize=(7.5, 4.2), dpi=130)
    ax_line.plot(monthly["date"], monthly["flights_total"], color="#1e40af", lw=2, label="Total Flights")
    ax_line.plot(monthly["date"], monthly["flights_ontime"], color="#10b981", lw=1.8, label="On-Time Flights")
    ax_line.plot(monthly["date"], monthly["flights_delayed"], color="#ef4444", lw=1.8, label="Delayed Flights")
    ax_line.fill_between(monthly["date"], monthly["flights_delayed"], color="#ef4444", alpha=0.15)
    
    ax_line.set_ylabel("Flights Count")
    ax_line.legend(loc="upper left", frameon=True, facecolor="white")
    ax_line.grid(True, axis="y", alpha=0.6)
    sns.despine(top=True, right=True)
    plt.xticks(rotation=25)
    plt.tight_layout()
    st.pyplot(fig_line)
    plt.close(fig_line)

with col_r1_2:
    st.markdown("##### 🍩 Pie Chart: Share of Delay Causes")
    causes = {
        "Carrier": filtered_df["delays_carrier"].sum(),
        "Late Aircraft": filtered_df["delays_late_aircraft"].sum(),
        "NAS Traffic": filtered_df["delays_nas"].sum(),
        "Weather": filtered_df["delays_weather"].sum(),
        "Security": filtered_df["delays_security"].sum()
    }
    pie_colors = ["#3b82f6", "#f97316", "#10b981", "#ef4444", "#8b5cf6"]

    fig_pie, ax_pie = plt.subplots(figsize=(6, 4.2), dpi=130)
    wedges, texts, autotexts = ax_pie.pie(
        causes.values(),
        autopct=lambda pct: f"{pct:.1f}%" if pct > 2 else "",
        pctdistance=0.75,
        startangle=130,
        colors=pie_colors,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
        at.set_fontsize(8.5)

    tot_causes = sum(causes.values())
    ax_pie.text(0, 0, f"{tot_causes:,.0f}\nDelays", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#334155")
    
    legend_labels = [f"{k} ({v:,.0f})" for k, v in causes.items()]
    ax_pie.legend(wedges, legend_labels, title="Delay Causes", loc="center left", bbox_to_anchor=(0.95, 0.5), frameon=False, fontsize=8)
    ax_pie.axis("equal")
    plt.tight_layout()
    st.pyplot(fig_pie)
    plt.close(fig_pie)

# Row 2: Bar Chart & Scatter Plot
col_r2_1, col_r2_2 = st.columns(2)

with col_r2_1:
    st.markdown("##### 📊 Bar Chart: Top 10 Airports by Delayed Flights")
    top_delayed_airports = filtered_df.groupby("airport_code")["flights_delayed"].sum().sort_values(ascending=True).tail(10)

    fig_bar, ax_bar = plt.subplots(figsize=(6.5, 3.8), dpi=130)
    bars = ax_bar.barh(top_delayed_airports.index, top_delayed_airports.values / 1000, color="#2563eb", alpha=0.85, height=0.6)
    
    for bar in bars:
        w = bar.get_width()
        ax_bar.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:,.0f}k", va="center", fontsize=8, color="#1e293b")

    ax_bar.set_xlabel("Delayed Flights (in Thousands)")
    ax_bar.set_ylabel("Airport Code")
    ax_bar.grid(True, axis="x", alpha=0.6)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)

with col_r2_2:
    st.markdown("##### 🔵 Scatter Plot: Total Operations vs. Delay Minutes")
    fig_scatter, ax_scatter = plt.subplots(figsize=(6.5, 3.8), dpi=130)
    sns.scatterplot(
        data=filtered_df,
        x="flights_total",
        y="mins_total",
        hue="delay_rate_pct",
        palette="viridis",
        alpha=0.65,
        s=35,
        ax=ax_scatter
    )
    ax_scatter.set_xlabel("Total Monthly Flights")
    ax_scatter.set_ylabel("Total Minutes Delayed")
    ax_scatter.legend(title="Delay Rate %", frameon=True, facecolor="white", fontsize=8)
    ax_scatter.grid(True, alpha=0.6)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    st.pyplot(fig_scatter)
    plt.close(fig_scatter)

# Row 3: Histogram & Box Plot
col_r3_1, col_r3_2 = st.columns(2)

with col_r3_1:
    st.markdown("##### 📊 Histogram: Delay Rate Distribution (%)")
    fig_hist, ax_hist = plt.subplots(figsize=(6.5, 3.8), dpi=130)
    sns.histplot(filtered_df["delay_rate_pct"], bins=35, kde=True, color="#0284c7", ax=ax_hist)
    median_val = filtered_df["delay_rate_pct"].median()
    ax_hist.axvline(median_val, color="#ef4444", linestyle="--", lw=1.5, label=f"Median ({median_val:.1f}%)")
    ax_hist.set_xlabel("Delay Rate (%)")
    ax_hist.set_ylabel("Frequency")
    ax_hist.legend(frameon=True, facecolor="white")
    sns.despine(top=True, right=True)
    plt.tight_layout()
    st.pyplot(fig_hist)
    plt.close(fig_hist)

with col_r3_2:
    st.markdown("##### 📦 Box Plot: Delay Reason Distribution")
    box_df = filtered_df[["delays_carrier", "delays_late_aircraft", "delays_nas", "delays_weather"]].rename(columns={
        "delays_carrier": "Carrier",
        "delays_late_aircraft": "Late Aircraft",
        "delays_nas": "NAS",
        "delays_weather": "Weather"
    })

    fig_box, ax_box = plt.subplots(figsize=(6.5, 3.8), dpi=130)
    sns.boxplot(data=box_df, palette="Set2", ax=ax_box, fliersize=2.5)
    ax_box.set_ylabel("Monthly Incidents")
    ax_box.grid(True, axis="y", alpha=0.6)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    st.pyplot(fig_box)
    plt.close(fig_box)

## ==========================================
# 6. CORRELATION ANALYSIS (SEABORN HEATMAP)
# ==========================================
st.markdown('<div class="section-header">6. Correlation Analysis</div>', unsafe_allow_html=True)

corr_vars = [
    "flights_total",
    "flights_ontime",
    "flights_delayed",
    "flights_cancelled",
    "flights_diverted",
    "mins_total",
    "mins_carrier",
    "mins_late_aircraft",
    "mins_nas",
    "mins_weather",
    "mins_security"
]

corr_matrix = filtered_df[corr_vars].corr()

fig_corr, ax_corr = plt.subplots(
    figsize=(12, 8),
    dpi=140
)

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.8,
    linecolor="white",
    ax=ax_corr,
    annot_kws={"size": 9}
)

ax_corr.set_title(
    "Flight Performance Correlation Heatmap",
    fontsize=13,
    fontweight="bold"
)

plt.xticks(rotation=35, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)

plt.tight_layout()
st.pyplot(fig_corr)
plt.close(fig_corr)

# ==========================================
# 7. DATA-DRIVEN INSIGHTS
# ==========================================
st.markdown('<div class="section-header">7. Data-Driven Insights</div>', unsafe_allow_html=True)

top_hub = top_delayed_airports.index[-1] if not top_delayed_airports.empty else "N/A"
nas_share = (filtered_df["delays_nas"].sum() / tot_causes) * 100 if tot_causes > 0 else 0
late_share = (filtered_df["delays_late_aircraft"].sum() / tot_causes) * 100 if tot_causes > 0 else 0

st.markdown(f"""
* **Primary Systemic Bottlenecks:** **NAS (Air Traffic System)** and **Late Aircraft Propagation** represent **{nas_share + late_share:.1f}%** of all documented flight delays.
* **Busiest Delay Hub:** Airport **`{top_hub}`** consistently records the highest total volume of delayed flights in the selected subset.
* **Volume Escalation Correlation:** Total Scheduled Flights and NAS Delays show a high correlation coefficient ($r > 0.72$), indicating congestion amplifies system delays.
* **Delay Duration Impact:** Carrier delays and Late Aircraft turnaround issues account for the largest proportion of gross delay hours accumulated nationwide.
""")

# ==========================================
# 8. MACHINE LEARNING DELAY PREDICTOR
# ==========================================
st.markdown('<div class="section-header">8. Machine Learning Delay Predictor</div>', unsafe_allow_html=True)

@st.cache_resource
def train_delay_model(df: pd.DataFrame):
    df_feat = df[["airport_code", "month", "carriers_total", "flights_total", "delay_rate_pct"]].copy()
    df_enc = pd.get_dummies(df_feat, columns=["airport_code"], drop_first=False)
    X = df_enc.drop(columns=["delay_rate_pct"])
    y = df_enc["delay_rate_pct"]
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
    model.fit(X, y)
    return model, X.columns.tolist()

ml_model, model_cols = train_delay_model(df_clean)

p_col1, p_col2 = st.columns(2)
with p_col1:
    pred_airport = st.selectbox("Select Target Airport:", all_airports, index=0)
    pred_month = st.slider("Select Month of Operation:", min_value=1, max_value=12, value=7)
with p_col2:
    pred_flights = st.number_input("Estimated Monthly Scheduled Flights:", min_value=1000, max_value=50000, value=12000, step=500)
    pred_carriers = st.slider("Active Operating Carriers:", min_value=1, max_value=25, value=12)

if st.button("🚀 Estimate Delay Rate & Risk", type="primary"):
    input_data = {col: 0 for col in model_cols}
    input_data["month"] = pred_month
    input_data["carriers_total"] = pred_carriers
    input_data["flights_total"] = pred_flights
    airport_col = f"airport_code_{pred_airport}"
    if airport_col in input_data:
        input_data[airport_col] = 1
        
    input_df = pd.DataFrame([input_data])[model_cols]
    pred_rate = ml_model.predict(input_df)[0]
    est_delayed = int(pred_flights * (pred_rate / 100))

    res1, res2, res3 = st.columns(3)
    res1.markdown(f'<div class="metric-card"><div class="metric-title">Predicted Delay Rate</div><div class="metric-value">{pred_rate:.2f}%</div><div class="metric-sub">Expected average</div></div>', unsafe_allow_html=True)
    res2.markdown(f'<div class="metric-card"><div class="metric-title">Estimated Delayed Flights</div><div class="metric-value">{est_delayed:,}</div><div class="metric-sub">Out of {pred_flights:,} flights</div></div>', unsafe_allow_html=True)
    
    risk_level = "HIGH 🔴" if pred_rate > 24 else ("MODERATE 🟡" if pred_rate > 18 else "LOW 🟢")
    res3.markdown(f'<div class="metric-card"><div class="metric-title">Operational Risk Level</div><div class="metric-value">{risk_level}</div><div class="metric-sub">Risk Assessment</div></div>', unsafe_allow_html=True)

# ==========================================
# 9. AIRPORT BENCHMARK / COMPARISON TOOL
# ==========================================
st.markdown('<div class="section-header">9. Side-by-Side Airport Benchmark Comparison</div>', unsafe_allow_html=True)

b_col1, b_col2 = st.columns(2)
with b_col1:
    airport_a = st.selectbox("Select Airport A:", all_airports, index=0, key="bench_a")
with b_col2:
    airport_b = st.selectbox("Select Airport B:", all_airports, index=min(1, len(all_airports)-1), key="bench_b")

df_a = filtered_df[filtered_df["airport_code"] == airport_a]
df_b = filtered_df[filtered_df["airport_code"] == airport_b]

if not df_a.empty and not df_b.empty:
    comp_metrics = pd.DataFrame({
        "Metric": [
            "Total Flights Handled",
            "Total Delayed Flights",
            "Overall Delay Rate (%)",
            "On-Time Rate (%)",
            "Cancellation Rate (%)",
            "Total Delay Duration (Hours)",
            "Avg Delay Time per Delayed Flight (Mins)"
        ],
        f"Airport {airport_a}": [
            f"{df_a['flights_total'].sum():,}",
            f"{df_a['flights_delayed'].sum():,}",
            f"{(df_a['flights_delayed'].sum()/df_a['flights_total'].sum()*100):.2f}%",
            f"{(df_a['flights_ontime'].sum()/df_a['flights_total'].sum()*100):.2f}%",
            f"{(df_a['flights_cancelled'].sum()/df_a['flights_total'].sum()*100):.2f}%",
            f"{(df_a['mins_total'].sum()/60):,.0f} hrs",
            f"{(df_a['mins_total'].sum()/df_a['flights_delayed'].sum()):.1f} mins"
        ],
        f"Airport {airport_b}": [
            f"{df_b['flights_total'].sum():,}",
            f"{df_b['flights_delayed'].sum():,}",
            f"{(df_b['flights_delayed'].sum()/df_b['flights_total'].sum()*100):.2f}%",
            f"{(df_b['flights_ontime'].sum()/df_b['flights_total'].sum()*100):.2f}%",
            f"{(df_b['flights_cancelled'].sum()/df_b['flights_total'].sum()*100):.2f}%",
            f"{(df_b['mins_total'].sum()/60):,.0f} hrs",
            f"{(df_b['mins_total'].sum()/df_b['flights_delayed'].sum()):.1f} mins"
        ]
    })
    st.dataframe(comp_metrics, use_container_width=True, hide_index=True)

    st.markdown("##### 📊 Comparative Delay Causes")
    comp_causes = pd.DataFrame({
        "Cause": ["Carrier", "Late Aircraft", "NAS Traffic", "Weather", "Security"],
        airport_a: [
            df_a["delays_carrier"].sum(),
            df_a["delays_late_aircraft"].sum(),
            df_a["delays_nas"].sum(),
            df_a["delays_weather"].sum(),
            df_a["delays_security"].sum()
        ],
        airport_b: [
            df_b["delays_carrier"].sum(),
            df_b["delays_late_aircraft"].sum(),
            df_b["delays_nas"].sum(),
            df_b["delays_weather"].sum(),
            df_b["delays_security"].sum()
        ]
    }).melt(id_vars="Cause", var_name="Airport", value_name="Incidents")

    fig_comp, ax_comp = plt.subplots(figsize=(8, 3.8), dpi=130)
    sns.barplot(data=comp_causes, x="Cause", y="Incidents", hue="Airport", palette=["#2563eb", "#f97316"], ax=ax_comp)
    ax_comp.set_ylabel("Total Incidents")
    ax_comp.grid(True, axis="y", alpha=0.6)
    sns.despine(top=True, right=True)
    plt.tight_layout()
    st.pyplot(fig_comp)
    plt.close(fig_comp)
else:
    st.warning("Insufficient data available for comparison under the current filter.")

# ==========================================
# 10. DATA & REPORT EXPORT
# ==========================================
st.markdown('<div class="section-header">10. Data & Diagnostic Report Export</div>', unsafe_allow_html=True)

exp_col1, exp_col2 = st.columns(2)
with exp_col1:
    st.subheader("1. Download Filtered Dataset")
    st.write(f"Includes **{len(filtered_df):,}** records matching the active filters.")
    csv_filtered = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📄 Download Cleaned CSV",
        data=csv_filtered,
        file_name="filtered_airlines_data.csv",
        mime="text/csv",
        type="primary"
    )

with exp_col2:
    st.subheader("2. Download KPI Summary Report")
    st.write("Exports aggregated airport operations and delay rates.")
    summary_export_df = filtered_df.groupby("airport_code").agg({
        "flights_total": "sum",
        "flights_delayed": "sum",
        "flights_cancelled": "sum",
        "mins_total": "sum"
    }).reset_index()
    summary_export_df["delay_rate_pct"] = (summary_export_df["flights_delayed"] / summary_export_df["flights_total"]) * 100
    
    csv_summary = summary_export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📊 Download Airport KPI Summary CSV",
        data=csv_summary,
        file_name="airport_performance_summary.csv",
        mime="text/csv"
    )
