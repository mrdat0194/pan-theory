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
