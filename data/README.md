# Data directory

The raw dataset is **not committed** to keep the repository lightweight and
reproducible.

- `raw/` – downloaded UCI archive and `diabetic_data.csv` (created by
  `src/data/download.py` or `scripts/run_pipeline.py`).
- `interim/` – reserved for intermediate artifacts.
- `processed/modeling_dataset.parquet` – raw records plus the engineered
  target and train/validation/test split labels (created by training).

Dataset: **Diabetes 130-US Hospitals for Years 1999-2008**
- Source: <https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>
- DOI: 10.24432/C5230J
- License: CC BY 4.0

Regenerate everything with:

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```
