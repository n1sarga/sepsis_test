# MIMIC-III Sepsis-3 EDA Dashboard

A Streamlit dashboard for the EDA produced in the project notebook `MIMICIII_Sepsis3_EDA.ipynb`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Open Streamlit Community Cloud.
2. Create a new app from the `n1sarga/sepsis_test` GitHub repository.
3. Select the `main` branch.
4. Set the main file path to `app.py`.
5. Deploy.

The app reads the existing processed files from the repository `data/` folder. No database or external API is required.

## Dashboard sections

- Overview
- Cohort
- Eligibility audit
- Class imbalance
- Missingness
- Temporal feature coverage
- Clinical trajectories
- Iteration 2 review

## Important note

The current 4-hour and 8-hour temporal datasets contain very small positive Sepsis-3 groups after temporal eligibility filtering. The dashboard presents this limitation explicitly and should not be interpreted as a clinically validated system.
