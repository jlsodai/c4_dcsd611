# Few Events, Many Lives

Exploratory Data Analysis of Global Disaster Impacts using the EM-DAT International Disaster Database.

**DSCD 611 Final Project**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://c4dcsd611.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/jlsodai/c4_dcsd611)

**[Live Dashboard](https://c4dcsd611.streamlit.app/)**     |     **[GitHub Repository](https://github.com/jlsodai/c4_dcsd611)**

## Dataset

| Detail            | Value                                              |
| ----------------- | -------------------------------------------------- |
| **Source**        | [EM-DAT Public Database](https://public.emdat.be/) |
| **Documentation** | [EM-DAT Docs](https://doc.emdat.be/docs/)          |
| **Time Period**   | 2000–2025                                          |
| **Records**       | 16,657 disaster events                             |

## Research Questions

1. How has the frequency of natural disasters changed over time?
2. Which disaster types occur most frequently worldwide?
3. Which disaster types cause the greatest total deaths?
4. Which disaster types affect the largest populations?
5. How does disaster impact vary by geographic region?
6. Are disaster impacts concentrated in a small number of events?
7. How have disaster impacts changed over time?

## Key Findings

- **Floods** are the most frequent disaster type, but **earthquakes** cause the most deaths
- **Asia** bears the highest burden of disaster impacts globally
- A small number of catastrophic events account for nearly **50%** of all recorded deaths

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Jupyter Notebook (Interactive Analysis)
```bash
jupyter notebook notebook.ipynb
```

### Python Script (Generates Figures)
```bash
python main.py
```
Saves visualizations to `figures/` folder.

### Streamlit Dashboard (Interactive Exploration)

**Live:** [https://c4dcsd611.streamlit.app/](https://c4dcsd611.streamlit.app/)

Or run locally:
```bash
streamlit run dashboard.py
```

## Project Structure

```
├── notebook.ipynb    # Jupyter notebook with full analysis
├── main.py           # Python script (saves figures)
├── dashboard.py      # Streamlit interactive dashboard
├── requirements.txt  # Python dependencies
├── figures/          # Generated visualizations
└── *.xlsx            # EM-DAT data export
```

## Requirements

- Python 3.10+
- pandas
- numpy
- matplotlib
- openpyxl
- streamlit (for dashboard)
- plotly (for dashboard)