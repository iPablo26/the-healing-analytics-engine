import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

def simulate_dataset(filepath, num_rows=10000):
    """
    Simulates a messy traffic dataset with:
    - 10,000 rows.
    - Missing values (NaNs) in timestamp, traffic_volume, response_time.
    - Formatting issues in timestamp (prefixes, suffixes, alternative formats, corruption).
    - Numerical outliers (negative/extreme traffic volumes and extreme response times).
    """
    print(f"Simulating messy dataset with {num_rows} rows...")
    np.random.seed(42)
    
    # 1. Timestamps
    base_time = pd.Timestamp("2026-06-01 00:00:00")
    timestamps = [base_time + pd.Timedelta(minutes=5 * i) for i in range(num_rows)]
    
    timestamp_strs = []
    for i, ts in enumerate(timestamps):
        r = np.random.rand()
        if r < 0.05:
            # Null value
            timestamp_strs.append(np.nan)
        elif r < 0.10:
            # Format: DD/MM/YYYY HH:MM:SS
            timestamp_strs.append(ts.strftime("%d/%m/%Y %H:%M:%S"))
        elif r < 0.15:
            # Prefix ERR-
            timestamp_strs.append(f"ERR-{ts.strftime('%Y-%m-%d %H:%M:%S')}")
        elif r < 0.20:
            # Suffix (UTC)
            timestamp_strs.append(f"{ts.strftime('%Y-%m-%d %H:%M:%S')} (UTC)")
        elif r < 0.22:
            # Completely unparseable
            timestamp_strs.append("corrupted_timestamp")
        else:
            # Standard YYYY-MM-DD HH:MM:SS
            timestamp_strs.append(ts.strftime("%Y-%m-%d %H:%M:%S"))
            
    # 2. Traffic Volume (Mean=500, Std=100)
    traffic_volume = np.random.normal(500, 100, num_rows)
    # Null values (5%)
    vol_null_indices = np.random.choice(num_rows, int(num_rows * 0.05), replace=False)
    traffic_volume[vol_null_indices] = np.nan
    # Deliberate outliers: High outliers (2%) and Negative outliers (1%)
    vol_outlier_high_indices = np.random.choice(num_rows, int(num_rows * 0.02), replace=False)
    traffic_volume[vol_outlier_high_indices] = np.random.uniform(5000, 10000, len(vol_outlier_high_indices))
    vol_outlier_neg_indices = np.random.choice(num_rows, int(num_rows * 0.01), replace=False)
    traffic_volume[vol_outlier_neg_indices] = np.random.uniform(-1000, -100, len(vol_outlier_neg_indices))
    
    # 3. Response Time (Mean=50ms, Std=10ms)
    response_time = np.random.normal(50, 10, num_rows)
    # Null values (5%)
    resp_null_indices = np.random.choice(num_rows, int(num_rows * 0.05), replace=False)
    response_time[resp_null_indices] = np.nan
    # High outliers (2%)
    resp_outlier_high_indices = np.random.choice(num_rows, int(num_rows * 0.02), replace=False)
    response_time[resp_outlier_high_indices] = np.random.uniform(1000, 3000, len(resp_outlier_high_indices))
    
    df = pd.DataFrame({
        "timestamp": timestamp_strs,
        "traffic_volume": traffic_volume,
        "response_time": response_time
    })
    
    df.to_csv(filepath, index=False)
    print(f"Dataset simulated and saved to {filepath}")

def clean_dataset(filepath):
    """
    Cleans raw dataset:
    - Normalizes and parses datetime formats.
    - Imputes missing numerical values using robust medians.
    - Drops invalid timestamps.
    """
    print("Cleaning dataset...")
    df = pd.read_csv(filepath)
    original_row_count = len(df)
    
    # Initial missing values count
    initial_null_timestamp = df['timestamp'].isna().sum()
    initial_null_volume = df['traffic_volume'].isna().sum()
    initial_null_response = df['response_time'].isna().sum()
    
    # Normalize strings: strip prefix 'ERR-' and suffix ' (UTC)'
    # Note that we use fillna("") first so astype(str) doesn't produce "nan" strings.
    cleaned_timestamps = df['timestamp'].fillna("").astype(str)
    cleaned_timestamps = cleaned_timestamps.str.replace(r"^ERR-", "", regex=True)
    cleaned_timestamps = cleaned_timestamps.str.replace(r"\s*\(UTC\)$", "", regex=True)
    cleaned_timestamps = cleaned_timestamps.replace("", np.nan)
    
    # Parse timestamps
    # pd.to_datetime handles DD/MM/YYYY, YYYY-MM-DD, etc. seamlessly when errors='coerce' is set
    df['cleaned_timestamp'] = pd.to_datetime(cleaned_timestamps, errors='coerce')
    
    # Drop rows where timestamp is unparseable or null
    df_clean = df.dropna(subset=['cleaned_timestamp']).copy()
    rows_dropped_timestamp = original_row_count - len(df_clean)
    
    # Impute missing numerical values using MEDIAN (robust to simulated extreme outliers)
    vol_median = df_clean['traffic_volume'].median()
    resp_median = df_clean['response_time'].median()
    
    imputed_volume_count = df_clean['traffic_volume'].isna().sum()
    imputed_response_count = df_clean['response_time'].isna().sum()
    
    df_clean['traffic_volume'] = df_clean['traffic_volume'].fillna(vol_median)
    df_clean['response_time'] = df_clean['response_time'].fillna(resp_median)
    
    stats = {
        "original_row_count": original_row_count,
        "initial_null_timestamp": initial_null_timestamp,
        "initial_null_volume": initial_null_volume,
        "initial_null_response": initial_null_response,
        "rows_dropped_timestamp": rows_dropped_timestamp,
        "vol_median": vol_median,
        "resp_median": resp_median,
        "imputed_volume_count": imputed_volume_count,
        "imputed_response_count": imputed_response_count,
        "cleaned_row_count": len(df_clean)
    }
    
    print("Dataset cleaned successfully.")
    return df_clean, stats

def detect_anomalies(df_clean):
    """
    Trains an Isolation Forest model to flag the top 5% extreme anomalies.
    """
    print("Training Isolation Forest model...")
    features = ['traffic_volume', 'response_time']
    X = df_clean[features]
    
    # Flag top 5% extreme traffic anomalies
    clf = IsolationForest(contamination=0.05, random_state=42)
    df_clean['is_anomaly'] = clf.fit_predict(X) == -1
    
    anomalies = df_clean[df_clean['is_anomaly']]
    normal = df_clean[~df_clean['is_anomaly']]
    
    anomaly_stats = {
        "num_normal": len(normal),
        "num_anomalies": len(anomalies),
        "anomaly_pct": len(anomalies) / len(df_clean) * 100,
        
        "normal_vol_mean": normal['traffic_volume'].mean(),
        "normal_vol_max": normal['traffic_volume'].max(),
        "normal_vol_min": normal['traffic_volume'].min(),
        "anomaly_vol_mean": anomalies['traffic_volume'].mean(),
        "anomaly_vol_max": anomalies['traffic_volume'].max(),
        "anomaly_vol_min": anomalies['traffic_volume'].min(),
        
        "normal_resp_mean": normal['response_time'].mean(),
        "normal_resp_max": normal['response_time'].max(),
        "anomaly_resp_mean": anomalies['response_time'].mean(),
        "anomaly_resp_max": anomalies['response_time'].max()
    }
    
    print("Model training and anomaly detection completed.")
    return df_clean, anomaly_stats

def generate_plot(df_clean, output_paths):
    """
    Generates a scatter plot of anomalies vs normal traffic.
    """
    print("Generating matplotlib visualization...")
    plt.figure(figsize=(10, 6))
    
    normal = df_clean[~df_clean['is_anomaly']]
    anomalies = df_clean[df_clean['is_anomaly']]
    
    # Plot normal points in Indigo
    plt.scatter(
        normal['traffic_volume'], 
        normal['response_time'], 
        c='#4F46E5', 
        alpha=0.6, 
        label='Normal Traffic', 
        edgecolors='none', 
        s=15
    )
    
    # Plot anomalous points in Red
    plt.scatter(
        anomalies['traffic_volume'], 
        anomalies['response_time'], 
        c='#EF4444', 
        alpha=0.8, 
        label='Anomalies (Top 5%)', 
        edgecolors='black', 
        linewidths=0.5,
        s=30
    )
    
    plt.title('Traffic Anomaly Detection (Isolation Forest Clusters)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Traffic Volume (Requests / Min)', fontsize=11)
    plt.ylabel('Response Time (ms)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=10)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    
    for path in output_paths:
        plt.savefig(path, dpi=300)
        print(f"Visualization saved to {path}")
        
    plt.close()

def generate_report(clean_stats, anomaly_stats, output_paths):
    """
    Generates anomaly_report.md summarizing the pipeline stats and findings.
    """
    print("Generating report summary...")
    report_content = f"""# Anomaly Detection Pipeline Report

## 1. Data Cleaning & Preprocessing Summary

The raw traffic dataset was loaded and cleaned to ensure compatibility with the anomaly detection models. The preprocessing steps performed were:
- **Timestamp Parsing:** Extracted date-times, cleaned malformed formats (e.g., removing prefix `ERR-` and suffix ` (UTC)`), and dropped unparseable entries.
- **Handling Missing Values:** Missing numerical values were imputed using column-wise medians to protect the training process from the influence of extreme simulated outliers.

| Metric | Value |
| :--- | :--- |
| **Original Dataset Rows** | {clean_stats['original_row_count']:,} |
| **Initial Null Timestamps** | {clean_stats['initial_null_timestamp']:,} |
| **Initial Null Traffic Volumes** | {clean_stats['initial_null_volume']:,} |
| **Initial Null Response Times** | {clean_stats['initial_null_response']:,} |
| **Unparseable/Missing Timestamps Dropped** | {clean_stats['rows_dropped_timestamp']:,} |
| **Imputed Traffic Volume Values** | {clean_stats['imputed_volume_count']:,} (using median = {clean_stats['vol_median']:.2f}) |
| **Imputed Response Time Values** | {clean_stats['imputed_response_count']:,} (using median = {clean_stats['resp_median']:.2f}) |
| **Cleaned Dataset Rows** | {clean_stats['cleaned_row_count']:,} |

---

## 2. Anomaly Detection Results

We trained an **Isolation Forest** model to detect multivariate anomalies in our network traffic data. To isolate the top 5% extreme behaviors, we configured the model with a `contamination=0.05` hyperparameter.

- **Total Normal Observations:** {anomaly_stats['num_normal']:,} ({100 - anomaly_stats['anomaly_pct']:.2f}%)
- **Total Anomalous Observations:** {anomaly_stats['num_anomalies']:,} ({anomaly_stats['anomaly_pct']:.2f}%)

### Traffic Features Comparison

The table below contrasts the characteristics of normal traffic versus flagged anomalies:

| Feature & Metric | Normal Data | Anomalous Data |
| :--- | :--- | :--- |
| **Traffic Volume (Mean)** | {anomaly_stats['normal_vol_mean']:.2f} | {anomaly_stats['anomaly_vol_mean']:.2f} |
| **Traffic Volume (Min / Max)** | {anomaly_stats['normal_vol_min']:.2f} / {anomaly_stats['normal_vol_max']:.2f} | {anomaly_stats['anomaly_vol_min']:.2f} / {anomaly_stats['anomaly_vol_max']:.2f} |
| **Response Time (Mean)** | {anomaly_stats['normal_resp_mean']:.2f} ms | {anomaly_stats['anomaly_resp_mean']:.2f} ms |
| **Response Time (Max)** | {anomaly_stats['normal_resp_max']:.2f} ms | {anomaly_stats['anomaly_resp_max']:.2f} ms |

---

## 3. Visualization

Below is the visualization of the data points and identified anomaly clusters:

![Anomaly Clusters](anomaly_clusters.png)
"""
    for path in output_paths:
        with open(path, "w") as f:
            f.write(report_content.strip())
        print(f"Report saved to {path}")

def main():
    raw_path = "traffic_raw.csv"
    report_name = "anomaly_report.md"
    plot_name = "anomaly_clusters.png"
    
    # Artifact directory to replicate output files so they can be embedded/viewed in the walkthrough
    artifact_dir = "/Users/peterokwukogu/.gemini/antigravity-ide/brain/fd3b1164-6ca7-4959-837f-10126fcdb090"
    
    if not os.path.exists(raw_path):
        simulate_dataset(raw_path)
    else:
        print(f"Found existing raw dataset: {raw_path}")
        
    df_clean, clean_stats = clean_dataset(raw_path)
    df_labeled, anomaly_stats = detect_anomalies(df_clean)
    
    # Save output to both root directory and artifact directory
    plot_paths = [plot_name]
    report_paths = [report_name]
    
    if os.path.exists(artifact_dir):
        plot_paths.append(os.path.join(artifact_dir, plot_name))
        report_paths.append(os.path.join(artifact_dir, report_name))
        
    generate_plot(df_labeled, plot_paths)
    generate_report(clean_stats, anomaly_stats, report_paths)
    
    print("Pipeline executed successfully.")

if __name__ == "__main__":
    main()
