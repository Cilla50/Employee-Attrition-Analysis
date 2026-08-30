# Employee Attrition Analysis, ABC Manufacturing Ltd

**AnalystLab Africa | Data Science Internship Programme, Week 1**
**Project: Business Understanding & Data Exploration**

## Overview

This project investigates employee attrition at ABC Manufacturing Ltd on behalf of AnalystLab Africa Consulting. The goal of this phase is to understand the business problem, explore the workforce dataset, and generate evidence-based insights before any predictive modelling is considered.

## Dataset

[IBM HR Analytics – Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (Kaggle), 1,470 employee records, 35 features.

## Business Questions

1. What does the company's workforce look like?
2. Which departments have the highest employee attrition?
3. Does age influence attrition?
4. Does monthly income affect retention?
5. Does overtime influence attrition?
6. Which job roles experience the highest turnover?
7. Which variables appear important for future predictive modelling?

## Repository Contents

| File | Description |
|---|---|
| `attrition_analysis.ipynb` | Full Jupyter notebook, data loading, inspection, and visualisations |
| `Business_Understanding_Report.docx` | Business context, impact of attrition, and rationale for this engagement |
| `Dataset_Inspection_Report.docx` | Data structure, quality checks, and descriptive statistics |
| `Data_Visualisation_Report.docx` | Charts and interpretations (bar, histogram, boxplot, pie) |
| `Business_Insights_Report.docx` | Key workforce observations and attrition risk factors |
| `Reflection_Report.docx` | Personal reflection on the week's learning |
| `WA_Fn-UseC_-HR-Employee-Attrition.csv` | Source dataset |

## Key Findings

- **Overtime** is one of the strongest attrition drivers: 30.5% attrition among employees who work overtime vs. 10.4% among those who don't.
- **Income** matters: the lowest income quartile has a 29.3% attrition rate vs. 10.3% in the highest quartile.
- **Age** matters: employees aged 18–25 leave at 34.8%, compared to just 9.2% for the 36–45 bracket.
- **Job role** matters: Sales Representatives have the highest attrition rate (39.8%) of any role.
- **Department**: Sales has the highest departmental attrition rate (20.6%), ahead of HR (19.0%) and R&D (13.8%).

## Tools Used

Python, Pandas, Matplotlib, Seaborn, Google Colab

## Author

Priscilla Arhin, Data Science Intern, AnalystLab Africa

---

## Week 2: Feature Engineering & Data Preprocessing for Machine Learning

Building on Week 1, this phase cleans, transforms, encodes, scales, and engineers features to
produce a machine-learning-ready dataset for Week 3 model development.

### Additional Files

| File | Description |
|---|---|
| `Week2_Feature_Engineering.ipynb` | Full preprocessing notebook (cleaning → feature engineering → encoding → scaling) |
| `Week2_Business_Understanding_Report.docx` | Why preprocessing matters for this engagement |
| `Data_Preprocessing_Report.docx` | Full documentation of the preprocessing pipeline |
| `cleaned_dataset.csv` | Cleaned dataset (1470 × 31), constant/ID columns removed, still human-readable |
| `ml_ready_dataset.csv` | Machine-learning-ready dataset (1470 × 53), encoded and scaled |

### Pipeline Summary

| Stage | Shape | Notes |
|---|---|---|
| Raw dataset | 1470 × 35 | As downloaded from Kaggle |
| Cleaned dataset | 1470 × 31 | Dropped 3 constant columns + 1 ID column |
| Feature-engineered | 1470 × 36 | Added 5 new derived features |
| ML-ready (final) | 1470 × 53 | One-hot encoding expanded categoricals; numeric features standardised |

### Engineered Features

- **AverageSatisfaction**, composite of 4 related satisfaction scores
- **TenureRatio**, YearsAtCompany / TotalWorkingYears (removed in Week 3, see below)
- **IncomePerExperienceYear**, MonthlyIncome / TotalWorkingYears
- **AgeGroup**, age bracket matching Week 1's EDA (18-25, 26-35, 36-45, 46-55, 56-60)
- **PromotionStagnationRatio**, YearsSinceLastPromotion / YearsAtCompany (removed in Week 3, see below)

---

## Week 3: Advanced Data Analysis, Statistical Validation & Feature Engineering

Building on the cleaned Week 2 dataset, this phase conducts deeper exploratory analysis, formal
statistical hypothesis testing, additional feature engineering, and feature evaluation, producing
the Final Modelling Dataset for Week 4 machine learning development.

### Additional Files

| File | Description |
|---|---|
| `Week3_Advanced_EDA.ipynb` | Advanced EDA notebook, 13 visualisations covering correlation, multivariate, and target-variable analysis |
| `Project_Continuity_Summary.docx` | Recap connecting Week 1 to Week 3, including open questions carried forward |
| `Statistical_Analysis_Summary.docx` | 4 formal hypothesis tests (Chi-Square, 2x Mann-Whitney U, Kruskal-Wallis) with full interpretation |
| `Feature_Engineering_Documentation.docx` | 4 new engineered features with justification and evidence |
| `Feature_Evaluation_Summary.docx` | Correlation/redundancy analysis and retain/remove decisions for all 9 engineered features |
| `Business_Insights_Recommendations_Report.docx` | Executive summary, findings, recommendations, limitations, next steps |
| `Data_Dictionary.docx` | Full column-by-column reference for the Final Modelling Dataset |
| `Final_Modelling_Dataset.csv` | Final ML-ready dataset (1470 × 54), Week 4-ready |
| `requirements.txt` | Python package dependencies for this project |

### Key Statistical Findings

- **Department x OverTime**: not significant (Chi-Square, p = 0.9543), overtime is evenly spread across departments, ruling it out as the explanation for Sales' higher attrition.
- **Income by Attrition**: significant (Mann-Whitney U, p < 0.0001), median income 3,202 (left) vs 5,204 (stayed).
- **Age by Attrition**: significant (Mann-Whitney U, p < 0.0001), median age 32 (left) vs 36 (stayed).
- **Income across Job Role**: significant (Kruskal-Wallis, p < 0.0001), Sales Representative has the lowest median income (2,579) of any role in the company, plausibly explaining its highest attrition rate (39.8%, from Week 1).

### New Engineered Features (Week 3)

- **HighRiskProfile**, binary flag for employees who are simultaneously young (<=35), lower-paid (below median), and less experienced (<=10 years); 27.0% attrition rate vs 11.4% otherwise
- **LogMonthlyIncome**, log transform of MonthlyIncome, reduces skewness from 1.37 to 0.29
- **JobHopRate**, NumCompaniesWorked / (TotalWorkingYears + 1)
- **IncomeBand**, Low/Medium/High/Very High quartile bins of MonthlyIncome (reporting only, excluded from modelling dataset)

### Feature Evaluation Outcome

Two Week 2 engineered features were removed from the Final Modelling Dataset based on statistical
evidence of negligible relationship with Attrition:

- **TenureRatio** (r = -0.016 with Attrition)
- **PromotionStagnationRatio** (r = 0.03 with Attrition; contradicted by Week 3 EDA)

### Pipeline Summary (Updated)

| Stage | Shape | Notes |
|---|---|---|
| Raw dataset | 1470 x 35 | As downloaded from Kaggle |
| Cleaned dataset | 1470 x 31 | Dropped 3 constant columns + 1 ID column |
| Week 2 feature-engineered | 1470 x 36 | Added 5 new derived features |
| Week 3 feature-engineered | 1470 x 40 | Added 4 more derived features |
| **Final Modelling Dataset** | **1470 x 54** | 2 low-value Week 2 features removed; remaining features encoded and scaled |

---

## Week 4: HealthConnect Experience Lab, Data Science Track (New Project)

Starting Week 4, the internship transitions into the AnalystLab Africa Experience Lab, a shared,
multi-track project called HealthConnect Clinic. This is a separate project from the Weeks 1-3
Employee Attrition Analysis above; the attrition project is not being continued or replaced, both
remain part of the internship portfolio.

**Business problem:** HealthConnect Clinic experiences a high rate of missed appointments
(no-shows). The project explores how data, machine learning, and Generative AI can reduce missed
appointments and improve the patient support experience. Interns contribute from the perspective of
their assigned track; this portfolio covers the **Data Science** track.

### Files

| File | Description |
|---|---|
| `ML_Problem_Definition_Document.docx` | Problem definition, data assessment, target variable decision, candidate features, modelling approach |
| `Week4_Project_Summary.docx` | Concise summary of Week 4 work and proposed Week 5 focus |
| `Week4_Initial_Data_Exploration.ipynb` | Supporting notebook: data quality checks and candidate feature exploration |

### Key Week 4 Findings

- Dataset: 5,000 appointment records, 18 columns, 1,696 unique patients.
- Target variable outcome has 3 categories: No-Show (48.5%), Attended (46.3%), Cancelled (5.3%).
  Proposed approach: binary classification (No-Show vs Attended), excluding Cancelled appointments.
- Strongest candidate features identified: `booking_lead_days` (24.8% to 60.5% no-show rate as lead
  time increases) and `previous_no_shows` (43.5% to 68.8% as prior no-shows increase).
- `reminder_channel`'s 27.3% missingness is structural (no reminder sent = no channel), not a data
  quality issue. `distance_to_clinic_km` and `waiting_time_minutes` have limited genuine missing
  values requiring a handling strategy.
- `waiting_time_minutes` was checked for potential data leakage; near-identical distributions across
  all outcomes suggest it is an estimated value, not an actual recorded wait time, pending
  confirmation.

### Proposed Week 5 Focus

Confirm the Cancelled-appointment handling decision, finalise a missing-data strategy, run
multivariate/statistical analysis on currently weak features, and begin baseline model comparison.
