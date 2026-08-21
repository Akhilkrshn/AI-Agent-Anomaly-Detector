from datetime import datetime, timedelta
import numpy as np
import pandas as pd

def generate_mock_dataset():
    print("Generating 500,000 enterprise-scale mock banking records...")

    np.random.seed(42)
    n_records = 500_000

    start_date = datetime(2026, 1, 1)
    random_timestamps = [
        start_date + timedelta(seconds=int(s))
        for s in np.random.randint(0, 365 * 86400, size=n_records)
    ]

    df_enterprise = pd.DataFrame(
        {
            "Transaction_ID": [f"TXN_{i:07d}" for i in range(1, n_records + 1)],
            "Timestamp": random_timestamps,
            "Account_ID": np.random.randint(10000000, 99999999, size=n_records),
            "Merchant_Category_Code": np.random.choice(
                [5411, 5732, 5999, 4814, 6011], size=n_records
            ),
            "Amount": np.random.exponential(scale=120.0, size=n_records).round(2),
            "Risk_Score": np.random.beta(a=2, b=10, size=n_records).round(3),
            "Channel": np.random.choice(
                ["Online", "ATM", "POS", "Wire_Transfer"],
                size=n_records,
                p=[0.45, 0.20, 0.30, 0.05],
            ),
            "Source_Branch": np.random.choice(
                ["Retail_North", "Retail_South", "Commercial_HQ", "Merchant_Global"],
                size=n_records,
            ),
        }
    )

    #Injected controlled anomalies
    anomaly_indices = np.random.choice(n_records, size=5051, replace=False)
    df_enterprise.loc[anomaly_indices[:2500], "Amount"] *= np.random.uniform(
        15, 50, size=2500
    )
    df_enterprise.loc[anomaly_indices[2500:], "Risk_Score"] = np.random.uniform(
        0.85, 0.99, size=2551
    )

    #Injected dirty data
    dirty_indices = np.random.choice(n_records, size=3000, replace=False)
    df_enterprise.loc[dirty_indices[:1800], "Amount"] = np.nan
    df_enterprise.loc[dirty_indices[1800:2300], "Risk_Score"] = -1.0
    df_enterprise = pd.concat(
        [df_enterprise, df_enterprise.iloc[dirty_indices[2300:2800]]], ignore_index=True,
    )

    parquet_file = "banking_data.parquet"
    df_enterprise.to_parquet(parquet_file, engine="pyarrow", compression="snappy")

    excel_sample_file = "banking_data_sample.xlsx"
    df_enterprise.head(5000).to_excel(
        excel_sample_file, index=False, engine="openpyxl"
    )

    print("Mock Data generation complete!!!")
    print(f"- Full Parquet File: {parquet_file} ({len(df_enterprise)} records)")
    print(f"- Sample Excel File: {excel_sample_file} (5,000 records)")

if __name__ == "__main__":
    generate_mock_dataset()