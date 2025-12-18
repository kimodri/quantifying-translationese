# 📝 Quantifying Translationese in Filipino
## ⚠️ Branching Policy: Development & Review
**Do not push directly to the `main` branch.** All changes must be made via feature branches and merged **after peer review**.

Example:
`feat/<name>/<task>`
>`feat/kim/google-translate-paws`
>`feat/kim/google-translate-bcopa`

This repository supports the research paper: ***Translationese in Filipino: A Quantitative Study of Word Order Bias in Machine-Translated Benchmarks.***

---

## 🎯 Project Objective
The core objective is to quantitatively measure the presence and extent of **Translationese** (specifically, bias toward canonical word order) in English benchmarks translated into Filipino by various Machine Translation (MT) systems.

## 🚀 Progress Snapshot
* **Data Translation:** MT assignments and initial translation runs are underway.
* **Word Order Analysis:** We are close to completing the first key measurement: determining the baseline percentage of Canonical Word Order (KA) versus Non-Canonical Word Order (DKA) in the translated data.

---

## 🗓️ Next Steps & Assigned Tasks
Before translation, each entry must be tokenized (per sentence) and only then translate.

| # | Task | Assigned To | MT Tool | Target Due Date | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Machine Translation (MT)**: Translate assigned English columns into Filipino. | Anthea | Google Translate | 18 | In Progress |
| | | Bryan | Azure Translate (Bing) | 18 | In Progress |
| | | Marvin | DeepL | 18 | In Progress |
| | | JP | Helsinki's Opus (Helsinki-NLP/opus-mt-en-tl) | 18 | In Progress |
| **2** | **Word Order Analysis**: Implement a robust POS Tagger, apply to the translations, and calculate the **KA vs. DKA** percentage. | All (Focus on Data Analysis Leads) | N/A | 21 | Pending |
| **3** | **Checkpoint 1 Documentation**: Record all findings and analysis for the first research question (Quantifying the KA/DKA ratio). | All | N/A | 21 | Pending |

### 🛠️ Technical Guidance for Step 1 (Translation)

> **API Consumption Note:** When using your assigned MT API, remember to first convert the **pandas Series** of English text into a **Python list**. This list will serve as the input argument for your translation function. Assess the API key/rate limit capabilities to determine if you should feed the entire list at once or process it in smaller batches. (See my implementation in `../data-translation/google/translate_paws.ipynb`)

>Remember your API have limits so be careful when using them

### 📊 Metric Definition (Step 2)

* **KA (Karaniwang Ayos - Canonical Order):** Verb-Subject-Complement (e.g., *Bumili ako ng libro.* - *Bought I a book.*)
* **DKA (Di-Karaniwang Ayos - Non-Canonical Order):** Subject-Verb-Complement (e.g., *Ako ay bumili ng libro.* - *I bought a book.*)
