from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="MIMIC-III Sepsis-3 EDA Dashboard",
    page_icon="🩺",
    layout="wide",
)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def locate_data_dir():
    candidates = [
        Path.cwd() / "data",
        Path.cwd().parent / "data",
        Path("/mount/src/sepsis_test/data"),      # Streamlit Community Cloud
        Path("/kaggle/working/sepsis_test/data"),
    ]

    for p in candidates:
        if (p / "dataset_manifest.json").exists():
            return p.resolve()

    # Search one level below current working directory.
    for p in Path.cwd().glob("**/dataset_manifest.json"):
        return p.parent.resolve()

    return None


@st.cache_data(show_spinner=False)
def load_project_data(data_dir_str):
    data_dir = Path(data_dir_str)
    interim = data_dir / "interim"
    processed = data_dir / "processed"

    manifest = json.loads((data_dir / "dataset_manifest.json").read_text())
    audit_path = data_dir / "onset_eligibility_audit.json"
    onset_audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}

    cohort = pd.read_csv(
        interim / "mimic_iii_sepsis3_cohort.csv",
        low_memory=False,
    )

    for col in [
        "admittime",
        "dischtime",
        "intime",
        "outtime",
        "sepsis_onset_time",
    ]:
        if col in cohort.columns:
            cohort[col] = pd.to_datetime(cohort[col], errors="coerce")

    tensors = {}
    coverage = {}

    for h in [4, 8]:
        tensor_path = processed / f"timeseries_before_onset_{h}h.npz"
        coverage_path = processed / f"feature_coverage_before_onset_{h}h.csv"

        if tensor_path.exists():
            z = np.load(tensor_path, allow_pickle=False)
            tensors[h] = {k: z[k] for k in z.files}
            z.close()

        if coverage_path.exists():
            coverage[h] = pd.read_csv(coverage_path)

    return manifest, onset_audit, cohort, tensors, coverage


def eligible_cases(cohort, horizon, input_hours=12):
    c = cohort[cohort.sepsis_label == 1].copy()
    onset = c.sepsis_onset_time
    end = onset - pd.to_timedelta(horizon, unit="h")
    start = end - pd.to_timedelta(input_hours, unit="h")

    ok = (
        onset.notna()
        & onset.between(c.intime, c.outtime, inclusive="both")
        & start.ge(c.intime)
        & end.le(c.outtime)
    )
    return int(ok.sum())


def fmt_pct(x):
    return f"{100*x:.2f}%"


DATA_DIR = locate_data_dir()

# -----------------------------
# Header
# -----------------------------
st.title("Early Sepsis Prediction — EDA Dashboard")
st.caption(
    "Interactive dashboard for the processed MIMIC-III Sepsis-3 temporal dataset "
    "used in Iteration 2."
)

if DATA_DIR is None:
    st.error(
        "Project data could not be found. Deploy this app inside the repository "
        "that contains the `data/` folder, or place the `data/` folder beside app.py."
    )
    st.stop()

manifest, onset_audit, cohort, tensors, coverage = load_project_data(str(DATA_DIR))

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")
selected_horizon = st.sidebar.selectbox(
    "Prediction horizon",
    sorted(tensors.keys()) if tensors else [4, 8],
    format_func=lambda x: f"{x} hours",
)

st.sidebar.markdown("---")
st.sidebar.write("**Data location**")
st.sidebar.code(str(DATA_DIR), language=None)

page = st.sidebar.radio(
    "Dashboard section",
    [
        "Overview",
        "Cohort",
        "Eligibility",
        "Class Imbalance",
        "Missingness",
        "Temporal Coverage",
        "Clinical Trajectories",
        "Iteration 2 Review",
    ],
)

# -----------------------------
# Overview
# -----------------------------
if page == "Overview":
    st.subheader("Project Dataset Overview")

    total = len(cohort)
    sepsis = int((cohort.sepsis_label == 1).sum())
    controls = int((cohort.sepsis_label == 0).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", f"{total:,}")
    c2.metric("Sepsis-3", f"{sepsis:,}")
    c3.metric("Controls", f"{controls:,}")
    c4.metric("Sepsis prevalence", fmt_pct(sepsis / total))

    st.markdown("### Processed temporal datasets")
    rows = []

    for h, d in tensors.items():
        x = d["x"].astype(float)
        y = d["y"].astype(int)
        rows.append(
            {
                "Horizon": f"{h}h",
                "Tensor shape": str(tuple(x.shape)),
                "Patients": len(y),
                "Sepsis-3": int(y.sum()),
                "Controls": int((y == 0).sum()),
                "Positive prevalence": fmt_pct(y.mean()),
                "Missing after forward fill": fmt_pct(np.isnan(x).mean()),
                "All-missing windows": int(np.isnan(x).all(axis=(1, 2)).sum()),
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.info(
        "The initial cohort is approximately balanced, but the temporal eligibility "
        "criteria remove most positive patients. This is the central data-quality "
        "finding from Iteration 2."
    )

    st.markdown("### Current design")
    st.write(
        f"Each model input contains **{manifest.get('input_hours', 12)} hourly observations** "
        "and **22 clinical features**. The processed repository currently contains "
        "4-hour and 8-hour prediction datasets."
    )

# -----------------------------
# Cohort
# -----------------------------
elif page == "Cohort":
    st.subheader("Cohort Integrity and Composition")

    integrity = pd.DataFrame(
        {
            "Measure": [
                "Rows",
                "Unique patients",
                "Unique ICU stays",
                "Duplicate patients",
                "Duplicate ICU stays",
                "Sepsis-3",
                "Controls",
                "Sepsis prevalence",
                "Age <18",
                "Positive cases missing onset",
            ],
            "Value": [
                len(cohort),
                cohort.subject_id.nunique(),
                cohort.icustay_id.nunique(),
                cohort.subject_id.duplicated().sum(),
                cohort.icustay_id.duplicated().sum(),
                int((cohort.sepsis_label == 1).sum()),
                int((cohort.sepsis_label == 0).sum()),
                cohort.sepsis_label.mean(),
                int((cohort.age < 18).sum()) if "age" in cohort else np.nan,
                int(
                    (
                        (cohort.sepsis_label == 1)
                        & cohort.sepsis_onset_time.isna()
                    ).sum()
                ),
            ],
        }
    )

    st.dataframe(integrity, use_container_width=True, hide_index=True)

    counts = (
        cohort.sepsis_label
        .map({0: "Non-sepsis", 1: "Sepsis-3"})
        .value_counts()
        .rename_axis("Class")
        .reset_index(name="Patients")
    )

    fig = px.bar(
        counts,
        x="Class",
        y="Patients",
        text_auto=",",
        title="Initial Sepsis-3 cohort class distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    if {"age", "gender", "los_icu_days", "hospital_expire_flag"}.intersection(cohort.columns):
        st.markdown("### Descriptive statistics")

        rows = []
        for label, g in cohort.groupby("sepsis_label"):
            row = {
                "Group": "Sepsis-3" if label else "Non-sepsis",
                "N": len(g),
            }

            if "age" in g:
                row["Age median"] = g.age.median()
                row["Age Q1"] = g.age.quantile(0.25)
                row["Age Q3"] = g.age.quantile(0.75)

            if "gender" in g:
                female = g.gender.astype(str).str.upper().isin(["F", "FEMALE"])
                row["Female %"] = 100 * female.mean()

            if "los_icu_days" in g:
                row["ICU LOS median (days)"] = g.los_icu_days.median()

            if "hospital_expire_flag" in g:
                row["Mortality %"] = 100 * g.hospital_expire_flag.mean()

            rows.append(row)

        st.dataframe(
            pd.DataFrame(rows).round(2),
            use_container_width=True,
            hide_index=True,
        )

# -----------------------------
# Eligibility
# -----------------------------
elif page == "Eligibility":
    st.subheader("Sepsis Onset Timing and Temporal Eligibility")

    st.latex(r"t_{end}=t_{onset}-h")
    st.latex(r"t_{start}=t_{end}-12h")

    st.write(
        "For a 12-hour input window, the 4h, 8h and 12h prediction settings "
        "generally require sepsis onset at least 16h, 20h and 24h after ICU admission."
    )

    pos = cohort[cohort.sepsis_label == 1].copy()
    pos["onset_from_icu_hours"] = (
        pos.sepsis_onset_time - pos.intime
    ).dt.total_seconds() / 3600

    fig = px.histogram(
        pos,
        x="onset_from_icu_hours",
        nbins=70,
        title="Sepsis-3 onset relative to ICU admission",
        labels={"onset_from_icu_hours": "Hours from ICU admission to onset"},
    )

    for x, label in [(0, "ICU admission"), (16, "4h + 12h"), (20, "8h + 12h"), (24, "12h + 12h")]:
        fig.add_vline(x=x, line_dash="dash")
        fig.add_annotation(x=x, y=1, yref="paper", text=label, showarrow=False, yshift=12)

    st.plotly_chart(fig, use_container_width=True)

    elig = pd.DataFrame(
        {
            "Horizon": ["4h", "8h", "12h"],
            "Eligible Sepsis-3": [
                eligible_cases(cohort, 4),
                eligible_cases(cohort, 8),
                eligible_cases(cohort, 12),
            ],
        }
    )

    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.dataframe(elig, use_container_width=True, hide_index=True)
    with c2:
        fig2 = px.bar(
            elig,
            x="Horizon",
            y="Eligible Sepsis-3",
            text_auto=True,
            title="Positive eligibility by prediction horizon",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.warning(
        "The eligibility audit shows that most Sepsis-3 patients do not have enough "
        "pre-onset history for a full 12-hour observation window at the planned horizons."
    )

# -----------------------------
# Class imbalance
# -----------------------------
elif page == "Class Imbalance":
    st.subheader("Class Distribution After Temporal Filtering")

    rows = []
    for h, d in tensors.items():
        y = d["y"].astype(int)
        rows.append(
            {
                "Horizon": f"{h}h",
                "Patients": len(y),
                "Sepsis-3": int(y.sum()),
                "Controls": int((y == 0).sum()),
                "Positive prevalence %": 100 * y.mean(),
            }
        )

    tab = pd.DataFrame(rows)
    st.dataframe(tab.round(3), use_container_width=True, hide_index=True)

    melted = tab.melt(
        id_vars="Horizon",
        value_vars=["Controls", "Sepsis-3"],
        var_name="Class",
        value_name="Samples",
    )

    fig = px.bar(
        melted,
        x="Horizon",
        y="Samples",
        color="Class",
        barmode="group",
        title="Class distribution after temporal filtering",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.error(
        "The severe imbalance is introduced after temporal filtering. "
        "It is not representative of the original 11,302-patient cohort."
    )

# -----------------------------
# Missingness
# -----------------------------
elif page == "Missingness":
    st.subheader(f"Feature Coverage and Missingness — {selected_horizon}h")

    d = tensors[selected_horizon]
    x = d["x"].astype(float)
    y = d["y"].astype(int)

    if selected_horizon in coverage:
        cov = coverage[selected_horizon].copy()

        long = cov.melt(
            id_vars="feature",
            value_vars=["raw_coverage", "post_ffill_coverage"],
            var_name="Stage",
            value_name="Coverage",
        )
        long["Stage"] = long["Stage"].map(
            {
                "raw_coverage": "Raw observed",
                "post_ffill_coverage": "After forward fill",
            }
        )

        fig = px.bar(
            long,
            x="Coverage",
            y="feature",
            color="Stage",
            barmode="group",
            orientation="h",
            title=f"Feature coverage — {selected_horizon}h",
            range_x=[0, 1],
        )
        st.plotly_chart(fig, use_container_width=True)

    miss = np.isnan(x).mean(axis=(1, 2))
    all_missing = np.isnan(x).all(axis=(1, 2))

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean missingness", fmt_pct(miss.mean()))
    c2.metric("Median missingness", fmt_pct(np.median(miss)))
    c3.metric("All-missing windows", int(all_missing.sum()))

    miss_df = pd.DataFrame(
        {
            "Missing fraction": miss,
            "Class": np.where(y == 1, "Sepsis-3", "Non-sepsis"),
        }
    )

    fig2 = px.histogram(
        miss_df,
        x="Missing fraction",
        nbins=30,
        title="Patient-level missingness",
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.box(
        miss_df,
        x="Class",
        y="Missing fraction",
        points=False,
        title="Missingness by class",
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.info(
        "Forward filling is performed only within each patient window. "
        "Remaining missing values must be imputed using training-only statistics."
    )

# -----------------------------
# Temporal coverage
# -----------------------------
elif page == "Temporal Coverage":
    st.subheader(f"Temporal Feature Coverage — {selected_horizon}h")

    d = tensors[selected_horizon]
    names = d["feature_names"].astype(str).tolist()
    observed = np.isfinite(d["x"].astype(float)).mean(axis=0).T

    fig = go.Figure(
        data=go.Heatmap(
            z=observed,
            x=[f"Hour {i}" for i in range(1, 13)],
            y=names,
            zmin=0,
            zmax=1,
            colorbar=dict(title="Observed fraction"),
        )
    )
    fig.update_layout(
        title=f"Temporal feature coverage — {selected_horizon}h",
        xaxis_title="Observation hour",
        yaxis_title="Feature",
        height=700,
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Clinical trajectories
# -----------------------------
elif page == "Clinical Trajectories":
    st.subheader(f"Temporal Clinical Trajectories — {selected_horizon}h")

    d = tensors[selected_horizon]
    names = d["feature_names"].astype(str).tolist()
    x = d["x"].astype(float)
    y = d["y"].astype(int)

    default_features = [
        f for f in [
            "heart_rate",
            "resp_rate",
            "spo2",
            "temperature_c",
            "sbp",
            "glucose",
            "lactate",
            "wbc",
        ] if f in names
    ]

    feature = st.selectbox(
        "Clinical feature",
        names,
        index=names.index(default_features[0]) if default_features else 0,
    )

    j = names.index(feature)
    hours = np.arange(1, 13)
    traj = []

    for label, class_name in [(0, "Non-sepsis"), (1, "Sepsis-3")]:
        arr = x[y == label, :, j]
        mean = np.nanmean(arr, axis=0)
        for hour, value in zip(hours, mean):
            traj.append(
                {
                    "Observation hour": hour,
                    "Mean value": value,
                    "Class": class_name,
                    "N": len(arr),
                }
            )

    traj_df = pd.DataFrame(traj)

    fig = px.line(
        traj_df,
        x="Observation hour",
        y="Mean value",
        color="Class",
        markers=True,
        title=f"{feature.replace('_', ' ').title()} — {selected_horizon}h",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "These trajectories are exploratory. The positive groups are very small, "
        "so apparent differences should not be treated as stable population estimates."
    )

# -----------------------------
# Review
# -----------------------------
elif page == "Iteration 2 Review":
    st.subheader("Review and Evaluation of Iteration 2")

    st.success(
        "The pipeline successfully produced structured 12 × 22 temporal tensors "
        "for the 4-hour and 8-hour prediction settings."
    )

    st.markdown(
        """
        ### What worked
        - The cohort was reduced to unique first ICU stays.
        - The initial cohort contains 11,302 patients and is approximately balanced.
        - Twenty-two clinical variables were organised into 12 hourly time steps.
        - Within-window forward filling preserved incomplete temporal records.
        - The eligibility audit was completed before expensive model training.

        ### What the iteration revealed
        The major limitation appears after temporal-window filtering. Only a very
        small number of Sepsis-3 patients have enough pre-onset history to satisfy
        the 12-hour observation window plus the prediction gap. As a result, the
        4-hour and 8-hour temporal datasets become extremely imbalanced, while the
        12-hour setting does not produce a usable positive cohort.

        ### Consequence for Iteration 3
        Accuracy cannot be interpreted alone. Model training should also report
        precision, recall, F1, AUROC, AUPR, confusion matrices, cross-validation
        variation, and confidence intervals. Class-weighted training can be tested,
        but weighting cannot replace the lack of positive-patient diversity.
        """
    )

    st.markdown("### Repository metadata")
    st.json(
        {
            "input_hours": manifest.get("input_hours"),
            "requested_horizons_hours": manifest.get("requested_horizons_hours"),
            "omitted_12h": manifest.get("omitted_12h"),
            "cohort_patients": manifest.get("cohort_patients"),
            "label_definition": manifest.get("label_definition"),
            "status": manifest.get("status"),
        }
    )
