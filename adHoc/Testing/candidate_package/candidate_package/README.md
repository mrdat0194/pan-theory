# EDB Data Scientist Technical Assessment
**Role:** Data Scientist — Economic Development Board, Singapore Government
**Duration:** 90 minutes | **Total marks:** 100 pts

---

## What's in this package

```
assessment/
├── README.md                        ← You are here
├── EDB_Technical_Assessment.docx   ← Full assessment brief (read this first)
├── companies.csv                    ← Dataset for Part A
├── partA.ipynb                      ← Part A starter notebook
├── partC.md                         ← Part C written answers (fill in)
└── partB/
    ├── app.py                       ← Part B FastAPI skeleton
    ├── requirements.txt
    ├── .env.example                 ← Copy to .env and add your API key
    ├── README.md                    ← Part B documentation template (fill in)
    └── docs/
        ├── enterprise_development_grant.txt
        ├── workforce_development.txt
        └── innovation_and_ip.txt
```

---

## Quick setup (do this before the clock starts)

### Prerequisites
- Python 3.11+
- pip
- An OpenAI API key (for Part B)

### 1 — Install Part A dependencies
```bash
pip install jupyter pandas numpy scikit-learn xgboost matplotlib seaborn
jupyter notebook partA.ipynb
```

### 2 — Install Part B dependencies
```bash
cd partB
pip install -r requirements.txt
cp .env.example .env        # then edit .env and paste your OpenAI API key
uvicorn app:app --reload    # server starts at http://localhost:8000
```

### 3 — Part C (no install needed)
Open `partC.md` in any text editor and replace the `[YOUR ANSWER]` placeholders.

---

## Submission
When time is called, zip the entire folder and send to your invigilator:
```bash
zip -r assessment_<your_name>.zip assessment/
```

Submit with:
- `partA.ipynb` — all cells run, output visible
- `partB/app.py` + `partB/README.md` — completed
- `partC.md` — all answers filled in

---

## Notes
- Internet access is permitted. Documentation and package references are fine.
- AI coding assistants (Copilot, Cursor, ChatGPT, etc.) are **not permitted**.
- Partial solutions are accepted — submit whatever you have.
- If you hit environment issues in the first 10 minutes, ask the invigilator for a Codespaces link.
