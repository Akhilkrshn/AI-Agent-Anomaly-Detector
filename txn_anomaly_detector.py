import os
import pandas as pd
from sklearn.ensemble import IsolationForest

#Universal Ingestion
def load_dataset(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".parquet":
        return pd.read_parquet(file_path, engine="pyarrow")
    elif ext in [".xlsx", ".xls", ".xlsm", ".xlsb"]:
        excel_file = pd.ExcelFile(file_path)
        sheets = [pd.read_excel(excel_file, sheet_name=s) for s in excel_file.sheet_names]
        return pd.concat(sheets, ignore_index=True)
    elif ext == ".csv":
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

#Data Cleaning & Remediation Engine
def auto_clean_data(df, skip_rules=None):
    skip_rules = skip_rules or set()
    report_logs = []
    initial_count = len(df)

    deduped_df = df.copy()
    if "dedup" not in skip_rules:
        deduped_df = deduped_df.drop_duplicates().copy()
        duplicates_removed = initial_count - len(deduped_df)
        if duplicates_removed > 0:
            report_logs.append({
                "id": "dedup",
                "message": f"Removed {duplicates_removed:,} duplicate transaction records."
            })

    if "Amount" in deduped_df.columns and "amount_impute" not in skip_rules:
        missing_amount = deduped_df["Amount"].isnull().sum()
        if missing_amount > 0:
            median_amt = deduped_df["Amount"].median()
            deduped_df["Amount"] = deduped_df["Amount"].fillna(median_amt)
            report_logs.append({
                "id": "amount_impute",
                "message": f"Imputed {missing_amount:,} missing 'Amount' entries using median (${median_amt:.2f})."
            })

    if "Risk_Score" in deduped_df.columns and "risk_correct" not in skip_rules:
        out_of_range = (deduped_df["Risk_Score"] < 0) | (deduped_df["Risk_Score"] > 1)
        is_missing = deduped_df["Risk_Score"].isna()
        invalid_risk = out_of_range | is_missing
        invalid_risk_count = invalid_risk.sum()

        if invalid_risk_count > 0:
            median_risk = deduped_df.loc[~invalid_risk, "Risk_Score"].median()

            if pd.isna(median_risk):
                #Entire column is invalid/missing -- no valid values to derive a median from.
                #Fall back to the dataset-wide midpoint since Risk_Score is defined on [0, 1].
                median_risk = 0.5
                report_logs.append({
                    "id": "risk_correct_fallback",
                    "message": "All 'Risk_Score' entries were invalid or missing; no valid median existed, so 0.5 was used as a neutral fallback."
                })

            deduped_df.loc[invalid_risk, "Risk_Score"] = median_risk
            report_logs.append({
                "id": "risk_correct",
                "message": f"Corrected {invalid_risk_count:,} invalid or missing 'Risk_Score' entries to median ({median_risk:.4f})."
            })

    return deduped_df, report_logs