# Global vs. Local Explanations: The Map Is Not the Directions

> **The feature that matters most on average is not always the feature that matters for *you*.**

A small, reproducible demo showing why a global feature-importance chart can mislead when you use it to explain an individual prediction.

![Global vs local SHAP view](global_vs_local.png)

---

## The idea in 30 seconds

Think of a navigation app.

| | Navigation app | Explainable AI |
|---|---|---|
| **Global** | The city map: which roads are busiest overall | Feature importance: which features matter most on average |
| **Local** | Turn-by-turn directions for *your* trip | A SHAP explanation for *one* patient |

The busiest road in the city is not on every route. In the same way, the most important feature overall is not the main reason behind every single prediction.

```mermaid
flowchart LR
    A["Global view<br/>'worst perimeter' is the<br/>#1 feature on average"] -->|"does it hold for<br/>each patient?"| B{"Check every<br/>patient"}
    B -->|"116 of 171"| C["Yes: top driver"]
    B -->|"55 of 171"| D["No: something else<br/>drives the prediction"]
    style A fill:#2F5D8A,color:#fff
    style C fill:#2F5D8A,color:#fff
    style D fill:#C8553D,color:#fff
```

## Key result

A random forest trained on the Wisconsin breast cancer dataset reaches **95.3% accuracy**. SHAP says `worst perimeter` is the most important feature globally. Checking it patient by patient (171 test patients):

```mermaid
pie showData title Is "worst perimeter" the top driver for each patient?
    "Yes, #1 driver" : 116
    "No, ranked #2 or #3" : 34
    "No, ranked #4 or lower" : 21
```

- **68%** of patients: the global top feature really is their top driver.
- **32%** (55 patients): it is not.
- **12%** (21 patients): it is not even in their top 3.

Using the global chart to explain an individual patient would be wrong about a third of the time.

## How it works

```mermaid
flowchart TD
    A[Load breast cancer data<br/>569 tumours, 30 features] --> B[Split 70% train / 30% test]
    B --> C[Train Random Forest<br/>300 trees]
    C --> D[Compute SHAP values<br/>per patient, per feature]
    D --> E[Global: average the<br/>absolute SHAP values]
    D --> F[Local: rank features<br/>inside each patient]
    E --> G[Find the #1 global feature]
    G --> H[Where does it rank<br/>for each patient?]
    F --> H
    H --> I[Plot both views side by side]
```

**SHAP** gives every feature a score for every patient: how much that feature pushed the prediction up or down. Two ways to read those scores:

- **Global:** average the absolute scores over all patients. One ranking for the whole population.
- **Local:** rank the scores within one patient. A different ranking for each person.

## Reading the chart

| Panel | What it shows |
|---|---|
| **Left (the map)** | Top 10 features by mean absolute SHAP value. `worst perimeter` (blue) is first. |
| **Right (the directions)** | For each patient, the rank of `worst perimeter` in their own explanation. The blue bar is "it is the top driver". The red bars are "it is not". |

## Why it happens

`worst perimeter`, `worst radius` and `worst area` all measure tumour size, so they are highly correlated. The model can share credit between them differently from patient to patient. The global average hides that, and the local view shows it.

## Takeaways

1. A global ranking describes the **population**, not the **individual**.
2. For decisions about one person (a patient, a loan applicant), look at **local** explanations.
3. Report both: global for understanding the model, local for justifying a specific prediction.

## Run it yourself

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install shap scikit-learn matplotlib numpy pandas
python global_vs_local_demo.py
```

Output:

```
acc 0.953
global top feature: worst perimeter
NOT top-1 locally: 55 of 171
not in top-3: 21 of 171
saved global_vs_local.png
```

## Files

| File | Purpose |
|---|---|
| `global_vs_local_demo.py` | Trains the model, computes SHAP, builds the figure |
| `global_vs_local.png` | The generated figure |

## Stack

Python · scikit-learn · SHAP · matplotlib

> **Note:** This is an educational demo on a public dataset, not a medical tool.
