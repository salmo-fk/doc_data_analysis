# 🩺 Medical Chatbot Analytics Dashboard

## 📌 Overview
An interactive, self-contained analytics dashboard that explores the relationship between **diseases, symptoms, severity, and precautions**.  
It is designed for:
- Fast exploratory analysis (clinical/educational context)
- Reproducibility
- Easy sharing via a single HTML file or via GitHub Pages

🔗 **Live Demo (GitHub Pages):** [👉 Open Dashboard](https://<ton_user>.github.io/<ton_repo>/report_full_with_selector_STYLED.html)

---

## 📑 Table of Contents
1. [Datasets](#-datasets)
2. [Pipeline](#-pipeline)
3. [Features](#-features)
4. [Dashboard Walkthrough](#-dashboard-walkthrough)
5. [Technical Stack](#-technical-stack)
6. [Reproducibility](#-reproducibility)
7. [Screenshots](#-screenshots)
8. [Repository Structure](#-repository-structure)
9. [Limitations](#-limitations)
10. [Roadmap](#-roadmap)

---

## 📂 Datasets
- `dataset.csv` → Disease → Symptoms (wide format: Symptom_1 … Symptom_17)  
- `symptom-severity.csv` → Symptom → Severity weight (1–5)  
- `symptom_description.csv` → Disease → Description  
- `symptom_precaution.csv` → Disease → Up to four precautions  

---

## 🔄 Pipeline
1. **Pivot to long format**: Convert `Symptom_1 … Symptom_17` into a single column (Disease, Symptom).  
2. **Normalization**: lowercasing, trimming, replacing spaces with underscores.  
3. **Joins**:
   - Disease ↔ Description  
   - Disease ↔ Precautions  
   - Symptom ↔ Severity weight  
4. **Feature Engineering**:
   - Per-disease metrics (unique symptoms, mean/max severity)  
   - Symptom coverage (how many diseases each symptom appears in)  
   - Optional taxonomy: grouping into families (e.g., Respiratory, Dermatological, Neurological, etc.)  

---

## 📊 Features
- **Disease Profile Selector** (choose a disease → description, precautions, symptoms by severity)  
- **Global Analyses**:
  - Severity distribution (multi-color by weight 1–5)  
  - Top diseases by # symptoms (bar + severity coloring)  
  - Disease × Symptom heatmap  
  - Correlation between # symptoms and average severity  
  - Symptom co-occurrence heatmap  
- **Symptom Families (optional taxonomy)**:
  - Treemap (family → symptom count, color = severity)  
  - Stacked bars per disease  

---

## 🖥️ Dashboard Walkthrough
- [ ] Disease Profile Selector  
- [ ] Severity Distribution  
- [ ] Top Diseases by Symptoms  
- [ ] Disease × Symptom Heatmap  
- [ ] Symptom Co-occurrence Heatmap  
- [ ] Symptom Families Treemap  

---

## ⚙️ Technical Stack
- **Python** (Pandas, NumPy)  
- **Plotly** (interactive visualizations)  
- **HTML/CSS** (standalone report)  
- **GitHub Pages** (for easy online access)  

---

## 🔁 Reproducibility
- Deterministic outputs given same inputs & taxonomy  
- Companion notebooks/scripts provided (`/notebooks/` or `/src/`)  
- No external services required (Plotly served via CDN)  

---

## 📁 Repository Structure

/data/ # Input CSVs
/notebooks/ # Data prep & visualization notebooks
/src/ # Scripts
/outputs/ # Generated HTML + CSVs

---

## ⚠️ Limitations
- Severity weights are **ordinal 1–5**, not continuous medical measures.  
- Taxonomy is heuristic; replace with vetted ontology (SNOMED CT, ICD) for production.  
- Co-occurrence is dataset-level, **not patient-level**.  
- This tool is for **exploration/education only**, not for diagnosis.  

---

## 🚀 Roadmap
- [ ] Add search bar & compare view  
- [ ] Export profiles (PNG/CSV)  
- [ ] Confidence intervals on severity summaries  
- [ ] Community-driven taxonomy file (PRs welcome)  

---

## 📎 Links
- **Dashboard**: [GitHub Pages Demo](https://<ton_user>.github.io/<ton_repo>/report_full_with_selector_STYLED.html)  
- **Repository**: [GitHub Repo](https://github.com/<ton_user>/<ton_repo>)  
