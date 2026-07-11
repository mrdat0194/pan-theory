# EDB Data Scientist Assessment — Part C
## Short-answer & Code Review

**Time allowed:** 20 minutes  
**Total marks:** 20 pts

Write your answers directly in this file. Replace each `[YOUR ANSWER]` placeholder.

---

## C1 · Conceptual Questions  *(10 pts — 2 pts each)*

---

**Q1.** What is the key difference between RAG (Retrieval-Augmented Generation) and fine-tuning an LLM?
In what situation would you choose each approach?

> [YOUR ANSWER]

---

**Q2.** A churn prediction model that was performing well has seen its F1 score drop from 0.78 to 0.61 after 3 months in production — without any code changes.
What are the first two things you would investigate, and why?

> [YOUR ANSWER]

---

**Q3.** You are designing the chunking strategy for a RAG system over 50 lengthy PDF policy documents.
Compare **fixed-size chunking** vs **semantic/paragraph-aware chunking** — what are the tradeoffs?

> [YOUR ANSWER]

---

**Q4.** Your training dataset has 1,000,000 rows but only 500 are positive labels (churn = 1).
Name two concrete techniques you would use to handle this class imbalance and explain the tradeoff of each.

> [YOUR ANSWER]

---

**Q5.** You are about to expose an internal LLM-powered chatbot as a public-facing API.
Describe one security risk specific to LLM applications (not a generic web security issue) and how you would mitigate it.

> [YOUR ANSWER]

---

## C2 · Code Review  *(10 pts — 2.5 pts per bug)*

The Python snippet below has **exactly 4 bugs**. For each bug:
1. Identify the line(s) involved
2. Explain what is wrong and why it is harmful
3. Write the corrected code

```python
# File: model_pipeline.py

import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from openai import OpenAI

client = OpenAI()

def load_company_data(company_name: str, db_path: str) -> pd.DataFrame:
    """Load company records from the internal database."""
    conn = sqlite3.connect(db_path)
    # BUG 1 — somewhere in the next two lines
    query = f"SELECT * FROM companies WHERE name = '{company_name}'"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def build_features(df: pd.DataFrame):
    """Scale numeric features and return train/test splits."""
    features = ['headcount', 'years_with_edb', 'num_schemes_enrolled',
                'annual_grant_sgd', 'last_interaction_days']
    target = 'churned'

    X = df[features]
    y = df[target]

    # BUG 2 — somewhere in the next four lines
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, scaler


def ask_policy_question(question: str, context_chunks: list[str]) -> str:
    """Ask a question using retrieved policy chunks as context."""
    full_context = "\n\n".join(context_chunks)

    # BUG 3 — somewhere in the next eight lines
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an EDB policy expert. Answer using only the context provided."},
            {"role": "user", "content": f"Context:\n{full_context}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content


def save_predictions(predictions: list, output_path: str) -> None:
    """Write predictions to CSV."""
    # BUG 4 — somewhere in the next three lines
    pd.DataFrame(predictions).to_csv(output_path)
    print(f"Saved {len(predictions)} predictions to {output_path}")
```

---

### Your answers

**Bug 1**
- Lines involved: 
- What is wrong: 
- Fixed code:
```python
# your fix here
```

---

**Bug 2**
- Lines involved: 
- What is wrong: 
- Fixed code:
```python
# your fix here
```

---

**Bug 3**
- Lines involved: 
- What is wrong: 
- Fixed code:
```python
# your fix here
```

---

**Bug 4**
- Lines involved: 
- What is wrong: 
- Fixed code:
```python
# your fix here
```
