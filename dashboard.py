import streamlit as st
import pandas as pd
import numpy as np

# Prefer Plotly for interactivity; fallback to matplotlib if Plotly is not available.
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    import matplotlib.pyplot as plt
    PLOTLY_OK = False

st.set_page_config(page_title="EM-DAT Natural Disasters Dashboard", layout="wide")
st.title("Natural Disasters & Risk Dashboard")
st.caption("Dynamic filters update all KPIs and charts (Year, Region, Disaster Type, and Impact threshold).")

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_xlsx(file) -> pd.DataFrame:
    df = pd.read_excel(file)

    # Coerce key numeric fields
    num_cols = [
        "Total Deaths", "No. Injured", "No. Affected", "No. Homeless", "Total Affected",
        "Magnitude", "Latitude", "Longitude",
        "Total Damage ('000 US$)", "Total Damage, Adjusted ('000 US$)"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fix rare sentinel-like values in coords/magnitude
    if "Longitude" in df.columns:
        df.loc[df["Longitude"] <= -90, "Longitude"] = np.nan
    if "Latitude" in df.columns:
        df.loc[df["Latitude"] <= -90, "Latitude"] = np.nan
    if "Magnitude" in df.columns:
        df.loc[df["Magnitude"] < 0, "Magnitude"] = np.nan

    # Build start date for convenience (month/day can be missing)
    for c in ["Start Year", "Start Month", "Start Day"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    month = df.get("Start Month", pd.Series([np.nan]*len(df))).fillna(1).astype(int)
    day = df.get("Start Day", pd.Series([np.nan]*len(df))).fillna(1).astype(int)
    year = df.get("Start Year", pd.Series([np.nan]*len(df))).astype("Int64")

    df["Start Date"] = pd.to_datetime(dict(year=year, month=month, day=day), errors="coerce")

    # Helpful derived columns
    df["Year"] = pd.to_numeric(df["Start Year"], errors="coerce")
    df["Month"] = pd.to_numeric(df.get("Start Month"), errors="coerce")
    return df

# Load data from local file
DATA_PATH = "public_emdat_custom_request_2026-01-30_3cd9696b-0b00-4d47-875e-ba58371a5747.xlsx"
try:
    df = load_xlsx(DATA_PATH)
except Exception:
    st.error(
        "Could not load the dataset.\n\n"
        "Please ensure `{DATA_PATH}` is in the same directory as this dashboard."
    )
    st.stop()

# Required columns check
required = ["Start Year", "Disaster Type", "Region", "Country", "Total Deaths", "Total Affected"]
missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

year_min = int(np.nanmin(df["Year"]))
year_max = int(np.nanmax(df["Year"]))
year_range = st.sidebar.slider("Year range", min_value=year_min, max_value=year_max, value=(year_min, year_max), step=1)

regions = sorted([x for x in df["Region"].dropna().unique().tolist()])
types = sorted([x for x in df["Disaster Type"].dropna().unique().tolist()])

sel_regions = st.sidebar.multiselect("Region", options=regions, default=regions)
sel_types = st.sidebar.multiselect("Disaster Type", options=types, default=types)

impact_metric = st.sidebar.selectbox(
    "Primary impact metric",
    options=["Total Deaths", "Total Affected", "No. Injured"],
    index=0
)
if impact_metric not in df.columns:
    impact_metric = "Total Deaths"

min_impact = st.sidebar.number_input(
    f"Minimum {impact_metric} per event (optional)",
    min_value=0.0,
    value=0.0,
    step=1.0
)

# Apply filters
f = df.copy()
f = f[(f["Year"] >= year_range[0]) & (f["Year"] <= year_range[1])]
f = f[f["Region"].isin(sel_regions)]
f = f[f["Disaster Type"].isin(sel_types)]
f = f[pd.to_numeric(f[impact_metric], errors="coerce").fillna(0) >= min_impact]

if len(f) == 0:
    st.warning("No events match the current filters. Adjust filters in the sidebar.")
    st.stop()

# -----------------------------
# KPI row
# -----------------------------
events = int(len(f))
countries = int(f["Country"].nunique())
total_deaths = float(pd.to_numeric(f["Total Deaths"], errors="coerce").fillna(0).sum())
total_affected = float(pd.to_numeric(f["Total Affected"], errors="coerce").fillna(0).sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Events", f"{events:,}")
k2.metric("Countries", f"{countries:,}")
k3.metric("Total Deaths (reported)", f"{total_deaths:,.0f}")
k4.metric("Total Affected (reported)", f"{total_affected:,.0f}")

st.divider()

# -----------------------------
# Charts (2x2 grid)
# -----------------------------
c1, c2 = st.columns(2)
c3, c4 = st.columns(2)

# 1) Events per year
events_by_year = f.groupby("Year").size().reset_index(name="Events").sort_values("Year")
with c1:
    st.subheader("Events per year")
    if PLOTLY_OK:
        fig = px.line(events_by_year, x="Year", y="Events")
        st.plotly_chart(fig, width='stretch')
    else:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(events_by_year["Year"], events_by_year["Events"])
        st.pyplot(fig)

# 2) Top disaster types by frequency
type_counts = f["Disaster Type"].value_counts().head(10).reset_index()
type_counts.columns = ["Disaster Type", "Events"]
type_counts = type_counts.sort_values("Events")
with c2:
    st.subheader("Top disaster types (frequency)")
    if PLOTLY_OK:
        fig = px.bar(type_counts, x="Events", y="Disaster Type", orientation="h")
        st.plotly_chart(fig, width='stretch')
    else:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.barh(type_counts["Disaster Type"], type_counts["Events"])
        st.pyplot(fig)

# 3) Impact by type (selected metric)
impact_by_type = (f.groupby("Disaster Type", as_index=False)[impact_metric]
                    .sum(min_count=1)
                    .sort_values(impact_metric, ascending=False)
                    .head(10))
impact_by_type = impact_by_type.iloc[::-1]
with c3:
    st.subheader(f"Top disaster types by {impact_metric}")
    if PLOTLY_OK:
        fig = px.bar(impact_by_type, x=impact_metric, y="Disaster Type", orientation="h")
        st.plotly_chart(fig, width='stretch')
    else:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.barh(impact_by_type["Disaster Type"], impact_by_type[impact_metric])
        st.pyplot(fig)

# 4) Impact by region (selected metric)
impact_by_region = (f.groupby("Region", as_index=False)[impact_metric]
                      .sum(min_count=1)
                      .sort_values(impact_metric, ascending=False))
impact_by_region = impact_by_region.iloc[::-1]
with c4:
    st.subheader(f"Regional totals — {impact_metric}")
    if PLOTLY_OK:
        fig = px.bar(impact_by_region, x=impact_metric, y="Region", orientation="h")
        st.plotly_chart(fig, width='stretch')
    else:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.barh(impact_by_region["Region"], impact_by_region[impact_metric])
        st.pyplot(fig)

st.divider()

# -----------------------------
# Concentration analysis
# -----------------------------
st.subheader("Concentration: do a few events dominate total deaths?")
deaths = f[["DisNo.", "Event Name", "Country", "Disaster Type", "Year", "Total Deaths"]].copy()
deaths["Total Deaths"] = pd.to_numeric(deaths["Total Deaths"], errors="coerce")
deaths = deaths.dropna(subset=["Total Deaths"]).sort_values("Total Deaths", ascending=False)

if len(deaths) == 0:
    st.info("No reported death values under current filters.")
else:
    total = deaths["Total Deaths"].sum()
    top_n = st.slider("Top-N events to summarize", min_value=5, max_value=50, value=10, step=5)
    top = deaths.head(top_n)
    share = top["Total Deaths"].sum() / total if total else np.nan
    st.write(f"Top {top_n} events account for **{share:.2%}** of reported deaths (under current filters).")

    left, right = st.columns(2)

    with left:
        st.markdown("**Top events (table)**")
        show_cols = ["DisNo.", "Year", "Country", "Disaster Type", "Event Name", "Total Deaths"]
        show_cols = [c for c in show_cols if c in top.columns]
        st.dataframe(top[show_cols], width='stretch')

    with right:
        if PLOTLY_OK:
            fig = px.bar(top[::-1], x="Total Deaths", y="DisNo.", orientation="h",
                         hover_data=[c for c in ["Country","Disaster Type","Year","Event Name"] if c in top.columns])
            fig.update_layout(yaxis_title="Event ID (DisNo.)")
            st.plotly_chart(fig, width='stretch')
        else:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.barh(top["DisNo."].astype(str), top["Total Deaths"])
            st.pyplot(fig)

st.caption("Tip: adjust filters to compare hazard profiles and impacts across time windows and regions.")