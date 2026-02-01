#!/usr/bin/env python3
"""
DSCD 611 Final Project — EM-DAT Natural Disasters EDA

Few Events, Many Lives: Exploratory Data Analysis of Global Disaster Impacts

Run:
  python main.py

Outputs:
- Prints answers and summary tables to terminal
- Saves all visualisations as PNGs in figures/ folder
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
DATASET_PATH = "public_emdat_custom_request_2026-01-30_3cd9696b-0b00-4d47-875e-ba58371a5747.xlsx"
OUTPUT_DIR = "figures"

# Columns that should be numeric
NUMERIC_COLUMNS = [
    # Human impact
    "Total Deaths",
    "No. Injured",
    "No. Affected",
    "No. Homeless",
    "Total Affected",
    # Economic impact (in thousands of USD)
    "Total Damage ('000 US$)",
    "Total Damage, Adjusted ('000 US$)",
    "Insured Damage ('000 US$)",
    "Insured Damage, Adjusted ('000 US$)",
    "Reconstruction Costs ('000 US$)",
    "Reconstruction Costs, Adjusted ('000 US$)",
    # Event characteristics
    "Magnitude",
]

# Major disaster years to annotate on charts
MAJOR_DISASTER_YEARS = [
    2004,  # Indian Ocean tsunami
    2008,  # Cyclone Nargis
    2010,  # Haiti earthquake
    2023,  # Turkey-Syria earthquake
]


def load_data(path: str) -> pd.DataFrame:
    """
    Load EM-DAT Excel export into a pandas DataFrame.

    Args:
        path: Path to the .xlsx file.

    Returns:
        Raw disaster event data.
    """
    return pd.read_excel(path)


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Convert specified columns to numeric, coercing errors to NaN.

    This safely handles cases where numeric fields might contain
    text annotations (a pattern sometimes seen in database exports).

    Args:
        df: Input dataframe.
        cols: Column names to convert.

    Returns:
        DataFrame with converted columns.
    """
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fix_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace suspected sentinel values with NaN.

    Some database exports use placeholder values for missing data.
    This function treats the following as potential sentinels:
    - Longitude/Latitude: Values <= -90 (outside valid coordinate range)
    - Magnitude: Negative values

    Args:
        df: Input dataframe.

    Returns:
        DataFrame with suspected sentinels replaced by NaN.
    """
    sentinel_replacements = {
        "Longitude": lambda x: x <= -90,
        "Latitude": lambda x: x <= -90,
        "Magnitude": lambda x: x < 0,
    }

    for column, condition in sentinel_replacements.items():
        if column in df.columns:
            df.loc[condition(df[column]), column] = np.nan

    return df


def savefig(path: str) -> None:
    """Save current figure to path and close it."""
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    print(f"Using data file: {DATASET_PATH}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # Load and preprocess data
    # =========================================================================
    df = load_data(DATASET_PATH)
    print(f"Loaded {len(df):,} disaster event records")

    # Apply data cleaning
    df = coerce_numeric(df, NUMERIC_COLUMNS)
    df = fix_sentinels(df)

    # Create start date from separate year/month/day columns
    df["Start Month"] = pd.to_numeric(df.get("Start Month"), errors="coerce")
    df["Start Day"] = pd.to_numeric(df.get("Start Day"), errors="coerce")
    df["Start Year"] = pd.to_numeric(df.get("Start Year"), errors="coerce")

    # Fill missing month/day with 1 (default to first of month/year)
    start_month = df["Start Month"].fillna(1).astype(int)
    start_day = df["Start Day"].fillna(1).astype(int)

    df["Start Date"] = pd.to_datetime(
        {
            "year": df["Start Year"].astype("Int64"),
            "month": start_month,
            "day": start_day,
        },
        errors="coerce",
    )

    print("Preprocessing complete")

    # =========================================================================
    # Dataset Overview
    # =========================================================================
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Time range: {int(df['Start Year'].min())} to {int(df['Start Year'].max())}")
    print(f"Countries: {df['Country'].nunique():,}")
    print(f"Regions: {df['Region'].nunique()}")
    print(f"Disaster types: {df['Disaster Type'].nunique()}")
    print("=" * 60)

    # =========================================================================
    # Data Quality Assessment - Missingness Analysis
    # =========================================================================
    print("\nData quality — missingness (top 10 columns by missing rate):")
    missing_pct = (df.isna().mean() * 100).round(2)
    missing_top10 = missing_pct.sort_values(ascending=False).head(10)

    print("=" * 50)
    for col, pct in missing_top10.items():
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"{col[:40]:<40} {bar} {pct:>5.1f}%")

    # Key fields completeness
    print("\n" + "=" * 50)
    print("KEY ANALYSIS FIELDS - COMPLETENESS")
    print("=" * 50)
    key_fields = ["Disaster Type", "Country", "Region", "Start Year", "Total Deaths", "Total Affected"]
    for field in key_fields:
        if field in df.columns:
            complete_pct = 100 - missing_pct[field]
            print(f"{field:<25} {complete_pct:>6.1f}% complete")

    # =========================================================================
    # Q1: Temporal Trends - Disaster Frequency Over Time
    # =========================================================================
    print("\n" + "=" * 60)
    print("Q1) How has the frequency of natural disasters changed over time?")
    print("=" * 60)

    events_by_year = df.groupby("Start Year").size().sort_index()

    print(f"Mean events/year:   {events_by_year.mean():,.0f}")
    print(f"Median events/year: {events_by_year.median():,.0f}")
    print(f"Max events/year:    {events_by_year.max():,.0f} ({int(events_by_year.idxmax())})")
    print(f"Min events/year:    {events_by_year.min():,.0f} ({int(events_by_year.idxmin())})")

    print("\nTop 10 years by event count:")
    for year, count in events_by_year.sort_values(ascending=False).head(10).items():
        print(f"  {int(year)}: {count:,} events")

    # Visualization
    plt.figure(figsize=(12, 5))
    plt.plot(events_by_year.index, events_by_year.values, linewidth=2, color="#2E86AB")
    plt.fill_between(events_by_year.index, events_by_year.values, alpha=0.3, color="#2E86AB")
    plt.title("Number of Recorded Disaster Events per Year", fontweight="bold")
    plt.xlabel("Year")
    plt.ylabel("Number of Events")
    plt.grid(True, alpha=0.3)
    savefig(os.path.join(OUTPUT_DIR, "q1_events_per_year.png"))

    # =========================================================================
    # Q2: Disaster Type Frequency
    # =========================================================================
    print("\n" + "=" * 60)
    print("Q2) Which disaster types occur most frequently worldwide?")
    print("=" * 60)

    type_counts = df["Disaster Type"].value_counts()
    total_events = len(df)

    print(f"{'Disaster Type':<30} {'Count':>10} {'% of Total':>12}")
    print("-" * 55)
    for dtype, count in type_counts.head(15).items():
        pct = count / total_events * 100
        print(f"{dtype:<30} {count:>10,} {pct:>11.1f}%")

    # Visualization - Top 10
    top_types = type_counts.head(10)[::-1]
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_types)))
    plt.barh(top_types.index, top_types.values, color=colors)
    plt.title("Top 10 Disaster Types by Frequency (2000–2025)", fontweight="bold")
    plt.xlabel("Number of Events")
    for i, (dtype, count) in enumerate(top_types.items()):
        plt.text(count + 50, i, f"{count:,}", va="center", fontsize=10)
    savefig(os.path.join(OUTPUT_DIR, "q2_top10_types_by_frequency.png"))

    # =========================================================================
    # Q3: Mortality by Disaster Type
    # =========================================================================
    print("\n" + "=" * 60)
    print("Q3) Which disaster types cause the greatest total deaths?")
    print("=" * 60)

    deaths_by_type = (
        df.groupby("Disaster Type", as_index=False)["Total Deaths"]
        .sum(min_count=1)
        .sort_values("Total Deaths", ascending=False)
    )
    total_deaths_all = deaths_by_type["Total Deaths"].sum()

    print(f"{'Disaster Type':<30} {'Deaths':>15} {'% of Total':>12}")
    print("-" * 60)
    for _, row in deaths_by_type.head(15).iterrows():
        pct = row["Total Deaths"] / total_deaths_all * 100 if total_deaths_all else 0
        print(f"{row['Disaster Type']:<30} {row['Total Deaths']:>15,.0f} {pct:>11.1f}%")

    # Visualization
    top_deaths = deaths_by_type.head(10).iloc[::-1]
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_deaths)))
    plt.barh(top_deaths["Disaster Type"], top_deaths["Total Deaths"], color=colors)
    plt.title("Top 10 Disaster Types by Total Deaths (2000–2025)", fontweight="bold")
    plt.xlabel("Total Deaths")
    for i, (_, row) in enumerate(top_deaths.iterrows()):
        plt.text(row["Total Deaths"] + 5000, i, f"{row['Total Deaths']:,.0f}", va="center", fontsize=9)
    savefig(os.path.join(OUTPUT_DIR, "q3_top10_types_by_total_deaths.png"))

    # =========================================================================
    # Q4: Population Impact by Disaster Type
    # =========================================================================
    print("\n" + "=" * 60)
    print("Q4) Which disaster types affect the largest populations?")
    print("=" * 60)

    affected_by_type = (
        df.groupby("Disaster Type", as_index=False)["Total Affected"]
        .sum(min_count=1)
        .sort_values("Total Affected", ascending=False)
    )
    total_affected_all = affected_by_type["Total Affected"].sum()

    print(f"{'Disaster Type':<30} {'Affected':>20} {'% of Total':>12}")
    print("-" * 65)
    for _, row in affected_by_type.head(15).iterrows():
        pct = row["Total Affected"] / total_affected_all * 100 if total_affected_all else 0
        print(f"{row['Disaster Type']:<30} {row['Total Affected']:>20,.0f} {pct:>11.1f}%")

    # Visualization
    top_affected = affected_by_type.head(10).iloc[::-1]
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(top_affected)))
    plt.barh(top_affected["Disaster Type"], top_affected["Total Affected"], color=colors)
    plt.title("Top 10 Disaster Types by Total Affected (2000–2025)", fontweight="bold")
    plt.xlabel("Total Affected")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"{x/1e9:.1f}B" if x >= 1e9 else f"{x/1e6:.0f}M")
    )
    savefig(os.path.join(OUTPUT_DIR, "q4_top10_types_by_total_affected.png"))

    # =========================================================================
    # Q5: Regional Variation
    # =========================================================================
    print("\n" + "=" * 60)
    print("Q5) How does disaster impact vary by region?")
    print("=" * 60)

    deaths_by_region = (
        df.groupby("Region", as_index=False)["Total Deaths"]
        .sum(min_count=1)
        .sort_values("Total Deaths", ascending=False)
    )
    affected_by_region = (
        df.groupby("Region", as_index=False)["Total Affected"]
        .sum(min_count=1)
        .sort_values("Total Affected", ascending=False)
    )
    events_by_region = (
        df.groupby("Region").size().reset_index(name="Events").sort_values("Events", ascending=False)
    )

    # Regional summary table
    print(f"{'Region':<15} {'Events':>10} {'Deaths':>15} {'Affected':>20}")
    print("-" * 65)
    regional_summary = (
        events_by_region.merge(deaths_by_region, on="Region").merge(affected_by_region, on="Region")
    )
    for _, row in regional_summary.iterrows():
        print(
            f"{row['Region']:<15} {row['Events']:>10,} "
            f"{row['Total Deaths']:>15,.0f} {row['Total Affected']:>20,.0f}"
        )

    # Side-by-side charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    top_deaths_region = deaths_by_region.iloc[::-1]
    axes[0].barh(top_deaths_region["Region"], top_deaths_region["Total Deaths"], color="#C1292E")
    axes[0].set_title("Total Deaths by Region", fontweight="bold")
    axes[0].set_xlabel("Total Deaths")

    top_affected_region = affected_by_region.iloc[::-1]
    axes[1].barh(top_affected_region["Region"], top_affected_region["Total Affected"], color="#F26419")
    axes[1].set_title("Total Affected by Region", fontweight="bold")
    axes[1].set_xlabel("Total Affected")
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e9:.1f}B"))

    savefig(os.path.join(OUTPUT_DIR, "q5_regional_impact.png"))

    # =========================================================================
    # Q6: Impact Concentration
    # =========================================================================
    print("\n" + "=" * 60)
    print("Q6) Are disaster impacts concentrated in a small number of events?")
    print("=" * 60)

    deaths_df = df[["DisNo.", "Country", "Disaster Type", "Start Year", "Total Deaths"]].dropna(
        subset=["Total Deaths"]
    )
    deaths_df = deaths_df.sort_values("Total Deaths", ascending=False).reset_index(drop=True)

    if len(deaths_df) > 0:
        total_deaths = deaths_df["Total Deaths"].sum()

        # Top 10 deadliest events
        top10 = deaths_df.head(10)
        share_top10 = top10["Total Deaths"].sum() / total_deaths

        print(f"Total events with reported deaths: {len(deaths_df):,}")
        print(f"Total deaths recorded: {total_deaths:,.0f}")
        print(f"\nTop 10 events account for {share_top10:.1%} of all deaths")
        print("=" * 70)

        print(f"\n{'Rank':<5} {'Year':<6} {'Country':<25} {'Type':<20} {'Deaths':>12}")
        print("-" * 70)
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            print(
                f"{i:<5} {int(row['Start Year']):<6} {row['Country'][:24]:<25} "
                f"{row['Disaster Type'][:19]:<20} {row['Total Deaths']:>12,.0f}"
            )

        # Two-panel visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart of top 10
        axes[0].bar(range(1, 11), top10["Total Deaths"].values, color="#C1292E")
        axes[0].set_title("Top 10 Deadliest Disaster Events", fontweight="bold")
        axes[0].set_xlabel("Event Rank")
        axes[0].set_ylabel("Total Deaths")
        axes[0].set_xticks(range(1, 11))

        # Cumulative share curve
        deaths_df["cum_share"] = deaths_df["Total Deaths"].cumsum() / total_deaths
        ranks = np.arange(1, len(deaths_df) + 1)

        axes[1].plot(ranks, deaths_df["cum_share"], linewidth=2, color="#2E86AB")
        axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% of deaths")
        axes[1].axhline(y=0.9, color="gray", linestyle=":", alpha=0.5, label="90% of deaths")
        axes[1].set_title("Cumulative Share of Deaths by Event Rank", fontweight="bold")
        axes[1].set_xlabel("Event Rank (sorted by deaths)")
        axes[1].set_ylabel("Cumulative Share of Deaths")
        axes[1].legend()
        axes[1].set_ylim(0, 1.05)

        savefig(os.path.join(OUTPUT_DIR, "q6_impact_concentration.png"))

        # Concentration metrics
        pct_50 = (deaths_df["cum_share"] >= 0.5).idxmax() + 1
        pct_90 = (deaths_df["cum_share"] >= 0.9).idxmax() + 1

        print(f"\nConcentration Metrics:")
        print(f"  • Top {pct_50:,} events ({pct_50/len(deaths_df)*100:.1f}%) account for 50% of deaths")
        print(f"  • Top {pct_90:,} events ({pct_90/len(deaths_df)*100:.1f}%) account for 90% of deaths")
    else:
        print("No death data available for concentration analysis.")

    # =========================================================================
    # Q7: Impact Trends Over Time
    # =========================================================================
    print("\n" + "=" * 60)
    print("Q7) How have disaster impacts changed over time?")
    print("=" * 60)

    deaths_by_year = df.groupby("Start Year")["Total Deaths"].sum(min_count=1).sort_index()
    affected_by_year = df.groupby("Start Year")["Total Affected"].sum(min_count=1).sort_index()

    print("Top 10 years by total deaths:")
    for year, deaths in deaths_by_year.sort_values(ascending=False).head(10).items():
        print(f"  {int(year)}: {deaths:>15,.0f} deaths")

    # Stacked time series charts
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Deaths over time
    axes[0].plot(
        deaths_by_year.index, deaths_by_year.values, linewidth=2, color="#C1292E", marker="o", markersize=4
    )
    axes[0].fill_between(deaths_by_year.index, deaths_by_year.values, alpha=0.3, color="#C1292E")
    axes[0].set_title("Total Deaths per Year", fontweight="bold")
    axes[0].set_ylabel("Deaths")
    axes[0].grid(True, alpha=0.3)

    # Annotate major disaster years
    for year in MAJOR_DISASTER_YEARS:
        if year in deaths_by_year.index:
            axes[0].annotate(
                str(int(year)),
                (year, deaths_by_year[year]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
            )

    # Affected over time
    axes[1].plot(
        affected_by_year.index,
        affected_by_year.values,
        linewidth=2,
        color="#F26419",
        marker="o",
        markersize=4,
    )
    axes[1].fill_between(affected_by_year.index, affected_by_year.values, alpha=0.3, color="#F26419")
    axes[1].set_title("Total Affected per Year", fontweight="bold")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Affected")
    axes[1].grid(True, alpha=0.3)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.0f}M"))

    savefig(os.path.join(OUTPUT_DIR, "q7_impact_trends.png"))

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Saved figures to: {os.path.abspath(OUTPUT_DIR)}")
    print("Done.\n")


if __name__ == "__main__":
    main()
