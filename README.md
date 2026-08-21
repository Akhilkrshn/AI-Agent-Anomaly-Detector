### Financial Anomaly & Data Quality Audit Agent
An enterprise-grade financial transaction auditing system and interactive AI Agent. Built with Python, Streamlit, and PyArrow, this application ingests multi-format transaction records at scale (500,000+ rows), automatically cleans dirty data, detects multi-variate statistical anomalies using Isolation Forest (ML), and provides interactive human-in-the-loop audit controls.

### Key Features
1. **Universal File Compatibility:** Ingests `.parquet`, `.xlsx`, `.csv` datasets.
2. **Auto-Data Remediation:** Imputes missing values via dataset medians, purges duplicate records, and logs cleaning steps.
3. **ML Anomaly Detection:** Utilizes `IsolationForest` with customizable contamination sliders.
4. **Dual Exporter Engine:** Exports cleaned data and audit records to `.parquet` and `.xlsx`.
5. **Human-in-the-Loop Governance:** Allows financial auditors to inspect flagged anomalies, override false positives, and edit records dynamically.
6. **AI Risk Narrative Summary:** Integrates Groq API (`openai/gpt-oss-120b`) to generate automated executive summaries and downloadable `.docx` reports.


### Quickstart Guide

1. Prerequisites
**Python 3.10 to 3.12 Recommended** (or Python 3.14 with updated dependencies).

2. Installation
```bash
git clone https://github.com/YOUR_USERNAME/AI-Agent-Anomaly-Detector.git
cd AI-Agent-Anomaly-Detector

#Create virtual environment
python -m venv venv
venv\Scripts\activate      #Windows
#source venv/bin/activate  #MacOS/Linux

#Install dependencies
pip install -r requirements.txt
```

3. Data Generation (Optional)
Generate the 500k synthetic enterprise ledger:
```bash
python dataset_generation.py
```

4. Launch Application
Always launch the web interface via the Streamlit CLI:
```bash
streamlit run app.py
```

### Technical Stack Dependency Table

| Area | Package | Function |
|---|---|---|
| **Interface** | `streamlit` | Native UI components, tabbed tables, interactive charts, and state management |
| **Data Engine** | `pandas`, `pyarrow`, `openpyxl` | High-throughput ingestion, Parquet processing, Excel reporting |
| **Machine Learning** | `scikit-learn` | Outlier detection via `IsolationForest` |
| **AI Assessment** | `groq` | Executive audit narrative generation via LLM client |
| **Document Export** | `python-docx` | Automated `.docx` audit report generation |
| **Environment** | `python-dotenv` | Secure API key management |