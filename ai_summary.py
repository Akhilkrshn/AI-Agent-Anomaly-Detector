import os
import re
import json
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def sanitize_text_field(text: str) -> str:
    if not text:
        return ""
    text = text.replace("`", "")
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\*{1,3}(\S.*?\S|\S)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(\S.*?\S|\S)_{1,3}", r"\1", text)
    text = re.sub(r"[\*_]+", " ", text)
    text = re.sub(r"(\d)\s+%", r"\1%", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _safe_get_str(source: dict, key: str, default: str = "Not provided.") -> str:
    value = source.get(key, default) if isinstance(source, dict) else default
    if not isinstance(value, str):
        value = str(value) if value else default
    return sanitize_text_field(value)

def generate_executive_metrics(df: pd.DataFrame) -> dict:
    total_records = len(df)
    total_anomalies = int(df["Is_Anomaly"].sum()) if "Is_Anomaly" in df.columns else 0
    anomaly_rate_pct = round((total_anomalies / total_records * 100), 2) if total_records > 0 else 0.0

    anomalous_exposure = 0.0
    if "Is_Anomaly" in df.columns and "Amount" in df.columns:
        anomalous_exposure = float(df.loc[df["Is_Anomaly"], "Amount"].sum())

    avg_risk_clean = 0.0
    avg_risk_anomaly = 0.0
    if "Is_Anomaly" in df.columns and "Risk_Score" in df.columns:
        clean_scores = df.loc[~df["Is_Anomaly"], "Risk_Score"]
        anomaly_scores = df.loc[df["Is_Anomaly"], "Risk_Score"]
        if not clean_scores.empty:
            avg_risk_clean = round(float(clean_scores.mean()), 4)
        if not anomaly_scores.empty:
            avg_risk_anomaly = round(float(anomaly_scores.mean()), 4)

    risk_concentration_factor = round(avg_risk_anomaly / avg_risk_clean, 2) if avg_risk_clean > 0 else 0.0

    return {
        "total_records": total_records,
        "total_anomalies": total_anomalies,
        "anomaly_rate_pct": anomaly_rate_pct,
        "anomalous_exposure": anomalous_exposure,
        "avg_risk_clean": avg_risk_clean,
        "avg_risk_anomaly": avg_risk_anomaly,
        "risk_concentration_factor": risk_concentration_factor,
    }

def generate_ai_executive_report(df: pd.DataFrame, audit_logs: list):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return False, "GROQ_API_KEY environment variable is missing. Add it to your .env file."

    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    metrics = generate_executive_metrics(df)
    
    formatted_metrics = (
        f"- Total Records Evaluated: {metrics['total_records']:,}\n"
        f"- Total Anomalies Flagged: {metrics['total_anomalies']:,}\n"
        f"- Anomaly Rate: {metrics['anomaly_rate_pct']}%\n"
        f"- Total Fiscal Anomalous Exposure: ${metrics['anomalous_exposure']:,.2f}\n"
        f"- Average Risk Score (Clean Transactions): {metrics['avg_risk_clean']:.4f}\n"
        f"- Average Risk Score (Anomalous Transactions): {metrics['avg_risk_anomaly']:.4f}\n"
        f"- Risk Concentration Factor (Anomalous Avg / Clean Avg): {metrics['risk_concentration_factor']:.2f}x"
    )

    audit_summary_lines = []
    for log in audit_logs:
        if isinstance(log, dict):
            msg = log.get("message", str(log))
            audit_summary_lines.append(f"- {msg}")
        else:
            audit_summary_lines.append(f"- {str(log)}")
    audit_summary = "\n".join(audit_summary_lines) if audit_summary_lines else "No remediation operations performed."

    samples_markdown = ""
    if "Is_Anomaly" in df.columns:
        anomalies_df = df[df["Is_Anomaly"]]
        sample_subset = anomalies_df.head(10)
        samples_markdown = sample_subset.to_string(index=False) if not sample_subset.empty else "No anomalies detected."

    prompt = f"""
You are the Chief Risk Officer (CRO) drafting a rigorous executive risk assessment for the Board of Directors. Respond with a single JSON object only -- no preamble, no markdown code fences, no text outside the JSON object.

### SYSTEM AUDIT METRICS (pre-formatted -- copy these figures EXACTLY as written, digit-for-digit, do not recalculate or rewrite them):
{formatted_metrics}

### DATA REMEDIATION & PREPROCESSING LOGS:
{audit_summary}

### HIGH-EXPOSURE ANOMALIES SAMPLE:
{samples_markdown}

---
Return a JSON object with EXACTLY this structure:

{{
  "executive_overview": "Comprehensive multi-paragraph narrative (use \\n\\n between paragraphs) covering aggregate transaction volume, total records evaluated, exact anomaly count, anomaly rate, and total fiscal exposure, with business context.",
  "key_financial_findings": {{
    "financial_exposure": "Exact fiscal exposure magnitude, its percentage ratio to total volume, and operational significance of high-value outliers.",
    "risk_score_divergence": "Contrast baseline vs anomalous risk profiles, including the risk concentration factor.",
    "high_risk_concentrations": "Structural risk patterns across channels and branches contributing to high-value outliers."
  }},
  "data_quality_remediation": "Technical summary of automated data ingestion preprocessing, missing-value handling, and out-of-range corrections.",
  "actionable_recommendations": {{
    "targeted_branch_audits": "Prioritized operational inspections for high-exposure branches.",
    "monitoring_rule_refinements": "Calibrated threshold adjustments for high-risk channels.",
    "continuous_governance": "Formal governance cadence for ongoing audit validation and control enhancements."
  }}
}}

STRICT RULES:
1. Output valid JSON only -- no markdown, no backticks, no commentary before or after the JSON object.
2. Every string value must be plain text -- never use asterisks or underscores for bold or italics.
3. ALL numbers -- record counts, dollar amounts, percentages, risk scores, ratios -- must be written as numeric digits, copied exactly from the SYSTEM AUDIT METRICS section. NEVER spell out numbers in words.
4. Always include the dollar sign for currency and the percent sign for percentages, with NO space between the number and the percent sign.
"""
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw_content = response.choices[0].message.content or ""
        cleaned_content = re.sub(r"^```[a-zA-Z]*\n?|```$", "", raw_content.strip())

        try:
            report_data = json.loads(cleaned_content)
        except json.JSONDecodeError:
            return False, "The AI response could not be parsed as valid JSON. Please try regenerating the summary."

        if not isinstance(report_data, dict):
            return False, "The AI response was not a valid JSON object. Please try regenerating the summary."

        findings = report_data.get("key_financial_findings")
        if not isinstance(findings, dict):
            findings = {}

        recommendations = report_data.get("actionable_recommendations")
        if not isinstance(recommendations, dict):
            recommendations = {}

        final_report = f"""## 1. Executive Overview
{_safe_get_str(report_data, "executive_overview", "")}

## 2. Key Financial Findings
- Financial Exposure: {_safe_get_str(findings, "financial_exposure")}
- Risk Score Divergence: {_safe_get_str(findings, "risk_score_divergence")}
- High-Risk Concentrations: {_safe_get_str(findings, "high_risk_concentrations")}

## 3. Data Quality & Remediation
{_safe_get_str(report_data, "data_quality_remediation", "")}

## 4. Actionable Recommendations
- Targeted Branch Audits: {_safe_get_str(recommendations, "targeted_branch_audits")}
- Monitoring Rule Refinements: {_safe_get_str(recommendations, "monitoring_rule_refinements")}
- Continuous Governance: {_safe_get_str(recommendations, "continuous_governance")}
"""
        return True, final_report.strip()

    except Exception as e:
        return False, f"Groq API request failed: {str(e)}"