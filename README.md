# Airlines Performance & Delay Intelligence Dashboard

An interactive **Data Science and Machine Learning dashboard** for analyzing airline flight operations, delays, cancellations, diversions, and delay causes using real-world airline operations data.

Built with **Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and Streamlit**.

---

## Project Overview

Airline operations generate large amounts of data related to flight schedules, delays, cancellations, diversions, airports, carriers, and delay causes.

This project transforms that data into an interactive dashboard that helps users:

- Explore and understand the dataset
- Analyze airline operational performance
- Identify major causes of flight delays
- Compare airport performance
- Study trends over time
- Analyze relationships between operational variables
- Detect unusual delay patterns
- Estimate potential delay rates using Machine Learning

---

## Key Features

### 1. Dataset Overview
- Total records
- Number of airports
- Date range
- Dataset features
- Cleaned dataset preview

### 2. Data Cleaning & Transformation
- Standardized column names
- Converted time information into datetime format
- Handled missing numerical values
- Corrected negative security-delay values
- Created calculated performance indicators

### 3. Descriptive Statistics
Includes:

- Mean
- Median
- Standard deviation
- Minimum and maximum values
- Flight delay statistics
- Cancellation statistics
- Delay duration statistics

### 4. Flight Performance KPIs

The dashboard provides:

- Total flights
- Delayed flights
- Cancelled flights
- On-time flights
- On-time rate
- Delay rate
- Cancellation rate
- Total delay duration
- Average delay duration

### 5. Visual Analysis

The dashboard uses **Matplotlib and Seaborn** for:

- Line charts
- Bar charts
- Scatter plots
- Pie/Donut charts
- Histograms
- Box plots

### 6. Correlation Analysis

A Seaborn correlation heatmap is used to examine relationships between:

- Total flights
- On-time flights
- Delayed flights
- Cancelled flights
- Diversions
- Delay causes
- Total delay minutes
- Delay-duration categories

### 7. Data-Driven Insights

The dashboard analyzes:

- Major delay causes
- Airport delay volumes
- Delay trends over time
- Delay distributions
- Outliers
- Relationships between operational variables

### 8. Machine Learning Delay Predictor

A **Random Forest Regressor** from Scikit-learn is used to estimate flight delay rates.

The prediction interface uses operational inputs such as:

- Airport
- Month
- Number of operating carriers
- Estimated scheduled flights

The dashboard also calculates an estimated number of delayed flights from the predicted delay rate.

### 9. Airport Benchmark Comparison

Users can select two airports and compare:

- Total flights
- Delayed flights
- Delay rate
- On-time rate
- Cancellation rate
- Total delay duration
- Average delay duration
- Delay causes

### 10. Data Export

Users can download:

- Filtered cleaned dataset
- Airport performance summary

in CSV format.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning |
| Streamlit | Interactive dashboard |
| CSV | Dataset and data export |

---

##  Machine Learning

The project uses:

**Random Forest Regression**

to estimate the expected flight delay rate.

### Input Features

- Airport
- Month
- Operating carriers
- Scheduled flights

### Output

- Predicted delay rate
- Estimated delayed flights
- Operational risk category

> The prediction is intended for analytical and educational purposes and should not be interpreted as a production-grade operational forecasting system.

---

##  Project Structure

```text
AirlinesDashboard/
│
├── data/
│   ├── airlines.csv
│   ├── airlines_cleaned.csv
│   ├── airport_summary.csv
│   ├── delay_reason_summary.csv
│   ├── monthly_summary.csv
│   └── yearly_summary.csv
│
├── app.py
├── requirements.txt
└── README.md
