# Loading `.env` into your scripts

This directory contains a `.env` file with the `GOOGLE_API_KEYS` environment variable
(the keys flagged for deletion), comma-separated.

---

## Option 1 — PowerShell (one-liner before running)

```powershell
# Load .env and export every variable into the current shell session
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]*)=(.*)$') {
        [System.Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), 'Process')
    }
}
python check_project_id_v2.py
```

---

## Option 2 — `python-dotenv` (recommended for scripts)

1. Install the package once:
   ```powershell
   pip install python-dotenv
   ```

2. Add these two lines at the **top** of any script (e.g. `check_project_id_v2.py` or `match_keys_to_projects.py`):
   ```python
   from dotenv import load_dotenv
   load_dotenv()          # reads .env from the current working directory
   ```

3. Run normally — the variable is loaded automatically:
   ```powershell
   python check_project_id_v2.py
   python match_keys_to_projects.py
   ```

> **Tip**: Always run from the directory that contains `.env`, or pass the path explicitly:
> ```python
> load_dotenv(dotenv_path=r"C:\Users\mrdat\PycharmProjects\pan-theory\adHoc\kasa\customer_match\.env")
> ```

---

## Option 3 — PyCharm run configuration

1. Open **Run → Edit Configurations…**
2. Select your script configuration.
3. Click **"Load variables from file"** (the folder icon next to *Environment variables*).
4. Browse to `.env` in this directory.
5. PyCharm will inject all variables automatically when you run/debug.

---

## `.env` format reminder

```
GOOGLE_API_KEYS=key1,key2,key3,...
```

The scripts split on `,` and strip whitespace, so order doesn't matter.

---

> ⚠️ **Do not commit `.env` to Git.**
> Make sure `.env` is listed in `.gitignore`.

# Monthly GA4 Data Extraction Pipeline

This directory contains a pipeline to extract, visualize, and summarize GA4 data across multiple properties (VNA, Vinpearl, VinWonders) — all confirmed **Google Analytics 360** accounts.

## Properties

| Property | ID | GA4 360? |
|---|---|---|
| VNA | 237200408 | ✅ Yes |
| Vinpearl | 258003657 | ✅ Yes |
| VinWonders | 318969518 | ✅ Yes |

---

## Check if Properties are GA4 360

**Requirement:**
```powershell
pip install google-analytics-admin
```

**Run:**
```powershell
python check_360_properties.py
```

This uses the GA4 Admin API and reads the `service_level` field:
- `1` = `GOOGLE_ANALYTICS_STANDARD` (free tier)
- `2` = `GOOGLE_ANALYTICS_360` (paid tier — higher export limits, unsampled data)

**Credentials used:** `bubbly-cascade-398303-5f3dd0a21703.json` (service account with GA Viewer access)

---

## Monthly Report Pipeline

To generate monthly reports (e.g., for April), run the 3 scripts in order:

### Step 1 — Extract Data
```powershell
python ga4_april_batched_reports.py
```
Authenticates with `bubbly-cascade-398303-5f3dd0a21703.json`, fetches data in 3-day batches across 5 report types (traffic, pages, hardware, events, revenue) for all 3 properties. Exports to `april_2026_[property]_[type]_full.csv`.

### Step 2 — Visualize Data
```powershell
python visualize_april_batched.py
```
Reads the exported CSVs and generates charts (PNG).

### Step 3 — Summarize with Gemini
```powershell
python summarize_april_batched.py
```
Sends visuals/data to the Gemini API and generates an AI summary report.
> ⚠️ Requires `GOOGLE_API_KEY` environment variable to be set (see `.env` section above).

---

## To add a new month

1. Copy the 3 scripts and rename `march` / `april` → new month name.
2. Update the date range inside `ga4_[month]_batched_reports.py` (e.g., `2026-05-01` to `2026-05-31`).
3. Run Steps 1 → 2 → 3 above.
