---
name: google-sheets-automation
description: Activates whenever working with Google Drive, Google Sheets, Google APIs, oauth, gspread, or quick_sheet pipelines.
---
# Google APIs & Spreadsheet Automation

When managing Google APIs and spreadsheets:
- **Security:** Securely handle oauth flows and do not hardcode credentials. Ensure `users_config.pkl` and similar files are safely managed.
- **Rate Limits:** Space out API calls and adhere to Google's rate limits to avoid quota blocks.
- **Best Practices:** Use `gspread` or `quick_sheet` wrappers correctly for bulk updates rather than cell-by-cell loops.
