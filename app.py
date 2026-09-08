from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title='Sepsis Prediction Dashboard', page_icon='🩺', layout='wide')


def find_data():
    candidates = [
        Path.cwd() / 'data',
        Path('/mount/src/sepsis_test/data'),
    ]
    for p in candidates:
        if (p / 'dataset_manifest.json').exists():
            return p
    for p in Path.cwd().glob('**/dataset_manifest.json'):
        return p.parent
    return None


def find_results():
    candidates = [
        Path.cwd() / 'results' / 'kaggle_executed',
        Path('/mount/src/sepsis_test/results/kaggle_executed'),
        Path.cwd() / 'results',
        Path('/mount/src/sepsis_test/results'),
    ]
    for p in candidates:
        if (p / 'model_metrics.csv').exists():
            return p
    return None


@st.cache_data
def load_data(p):
    p = Path(p)
    cohort = pd.read_csv(p / 'interim/mimic_iii_sepsis3_cohort.csv', low_memory=False)
    manifest = json.loads((p / 'dataset_manifest.json').read_text())
    tensors = {}
    for h in [4, 8]:
        q = p / f'processed/timeseries_before_onset_{h}h.npz'
        if q.exists():
            z = np.load(q, allow_pickle=False)
            tensors[h] = {k: z[k] for k in z.files}
            z.close()
    return cohort, manifest, tensors


@st.cache_data
def load_results(p):
    p = Path(p)
    names = {
        'metrics': 'model_metrics.csv',
        'pred': 'heldout_predictions.csv',
        'cv': 'cross_validation_metrics.csv',
        'cv_summary': 'cross_validation_summary.csv',
        'bootstrap': 'bootstrap_confidence_intervals.csv',
        'xcase': 'xai_case.csv',
        'local': 'local_shap_feature_contributions.csv',
        'local_displayed': 'local_shap_top15_displayed.csv',
        'temporal': 'temporal_shap.csv',
        'temporal_displayed': 'temporal_shap_first5_displayed.csv',
        'attn': 'attention_matrix.csv',
        'attn_hour': 'attention_hour_importance.csv',
        'inventory': 'artifact_inventory.csv',
        'cohort_summary': 'cohort_input_summary.csv',
        'baseline': 'prior_baseline_metrics.csv',
    }
    out = {}
    for key, filename in names.items():
        q = p / filename
        if q.exists():
            out[key] = pd.read_csv(q)
    return out


DATA = find_data()
RES = find_results()
R = load_results(str(RES)) if RES else {}

st.title('Early Sepsis Prediction — Research Dashboard')
st.caption(
    'MIMIC-III Sepsis-3 exploratory analysis, executed model results and explainable AI. '
    'Research prototype only; not a clinical decision tool.'
)

if DATA is None:
    st.error('The repository data/ folder could not be found.')
    st.stop()

cohort, manifest, tensors = load_data(str(DATA))

page = st.sidebar.radio(
    'Section',
    [
        'Overview',
        'Eligibility',
        'Missingness',
        'Model Performance',
        'Cross-validation',
        'Bootstrap Uncertainty',
        'Patient / XAI Case',
        'Local SHAP',
        'Temporal SHAP',
        'Attention Heatmap',
        'Result Availability',
    ],
)

if page == 'Overview':
    total = len(cohort)
    sep = int((cohort.sepsis_label == 1).sum())
    ctl = total - sep
    a, b, c, d = st.columns(4)
    a.metric('Patients', f'{total:,}')
    b.metric('Sepsis-3', f'{sep:,}')
    c.metric('Controls', f'{ctl:,}')
    d.metric('Initial prevalence', f'{100 * sep / total:.1f}%')

    rows = []
    for h, x in tensors.items():
        y = x['y'].astype(int)
        rows.append({
            'Horizon': f'{h}h',
            'Tensor shape': str(x['x'].shape),
            'Patients': len(y),
            'Sepsis': int(y.sum()),
            'Controls': int((y == 0).sum()),
            'Positive prevalence %': 100 * y.mean(),
            'Missing %': 100 * np.isnan(x['x']).mean(),
        })
    st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True, hide_index=True)
    st.info(
        'The initial cohort is approximately balanced, but the 12-hour observation-window '
        'and prediction-gap eligibility rules leave very few positive patients in the 4h and 8h temporal datasets.'
    )

    if 'metrics' in R:
        st.markdown('### Executed experiment snapshot')
        m = R['metrics']
        best = m.sort_values(['horizon', 'aupr'], ascending=[True, False]).groupby('horizon').head(1)
        st.dataframe(
            best[['horizon', 'model', 'accuracy', 'precision', 'recall', 'f1', 'auroc', 'aupr', 'test_positives']].round(4),
            use_container_width=True,
            hide_index=True,
        )

elif page == 'Eligibility':
    for col in ['intime', 'outtime', 'sepsis_onset_time']:
        if col in cohort:
            cohort[col] = pd.to_datetime(cohort[col], errors='coerce')
    pos = cohort[cohort.sepsis_label == 1].copy()
    pos['onset_hours'] = (pos.sepsis_onset_time - pos.intime).dt.total_seconds() / 3600
    st.plotly_chart(
        px.histogram(pos, x='onset_hours', nbins=70, title='Sepsis onset relative to ICU admission'),
        use_container_width=True,
    )
    rows = []
    for h in [4, 8, 12]:
        end = pos.sepsis_onset_time - pd.to_timedelta(h, unit='h')
        start = end - pd.to_timedelta(12, unit='h')
        ok = pos.sepsis_onset_time.notna() & start.ge(pos.intime) & end.le(pos.outtime)
        rows.append({'Horizon': f'{h}h', 'Eligible positives': int(ok.sum())})
    tab = pd.DataFrame(rows)
    st.dataframe(tab, use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(tab, x='Horizon', y='Eligible positives', text_auto=True), use_container_width=True)

elif page == 'Missingness':
    h = st.selectbox('Horizon', sorted(tensors))
    x = tensors[h]['x'].astype(float)
    y = tensors[h]['y'].astype(int)
    miss = np.isnan(x).mean((1, 2))
    a, b, c = st.columns(3)
    a.metric('Mean missingness', f'{100 * miss.mean():.1f}%')
    b.metric('Median missingness', f'{100 * np.median(miss):.1f}%')
    c.metric('All-missing windows', int(np.isnan(x).all((1, 2)).sum()))
    df = pd.DataFrame({'Missing fraction': miss, 'Class': np.where(y == 1, 'Sepsis-3', 'Non-sepsis')})
    st.plotly_chart(px.histogram(df, x='Missing fraction', color='Class', nbins=30), use_container_width=True)

elif page == 'Model Performance':
    df = R.get('metrics')
    if df is None:
        st.warning('Executed model_metrics.csv was not found.')
        st.stop()

    h = st.selectbox('Horizon', sorted(df.horizon.unique()))
    v = df[df.horizon == h].copy()
    cols = [c for c in [
        'model', 'accuracy', 'precision', 'recall', 'specificity', 'f1',
        'balanced_accuracy', 'mcc', 'auroc', 'aupr', 'brier', 'test_positives', 'positive_weight'
    ] if c in v.columns]
    st.dataframe(v[cols].round(4), use_container_width=True, hide_index=True)

    metric = st.selectbox('Compare metric', [c for c in ['recall', 'f1', 'auroc', 'aupr', 'precision', 'balanced_accuracy'] if c in v.columns])
    st.plotly_chart(
        px.bar(v, x='model', y=metric, title=f'{metric.upper()} by model', text_auto='.3f'),
        use_container_width=True,
    )

    st.warning(
        'Accuracy is not sufficient for interpretation here because the held-out test sets contain only '
        f"{int(v.test_positives.iloc[0]) if 'test_positives' in v else 'very few'} positive cases. "
        'Recall, F1, AUROC and especially AUPR should be interpreted together with uncertainty estimates.'
    )

elif page == 'Cross-validation':
    cv = R.get('cv_summary')
    if cv is None:
        st.warning('Only the displayed CV summary was not available in the repository.')
        st.stop()
    h = st.selectbox('Horizon', sorted(cv.horizon.unique()))
    v = cv[cv.horizon == h].copy()
    st.dataframe(v.round(4), use_container_width=True, hide_index=True)

    metric = st.selectbox('CV metric', ['auroc', 'aupr', 'recall', 'f1', 'balanced_accuracy'])
    mean_col = f'{metric}_mean'
    std_col = f'{metric}_std'
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=v['model'],
        y=v[mean_col],
        error_y=dict(type='data', array=v[std_col]),
        name=metric.upper(),
    ))
    fig.update_layout(title=f'Five-fold CV {metric.upper()} mean ± SD', yaxis_title=metric.upper())
    st.plotly_chart(fig, use_container_width=True)

elif page == 'Bootstrap Uncertainty':
    ci = R.get('bootstrap')
    if ci is None:
        st.warning('bootstrap_confidence_intervals.csv was not found.')
        st.stop()
    h = st.selectbox('Horizon', sorted(ci.horizon.unique()))
    metric = st.selectbox('Metric', sorted(ci.metric.unique()))
    v = ci[(ci.horizon == h) & (ci.metric == metric)].copy()
    v['midpoint'] = (v['ci_2.5'] + v['ci_97.5']) / 2
    v['err_minus'] = v['midpoint'] - v['ci_2.5']
    v['err_plus'] = v['ci_97.5'] - v['midpoint']
    st.dataframe(v[['model', 'metric', 'ci_2.5', 'ci_97.5', 'valid_bootstrap_samples', 'test_positives']].round(4), use_container_width=True, hide_index=True)

    fig = go.Figure(go.Scatter(
        x=v['midpoint'],
        y=v['model'],
        mode='markers',
        error_x=dict(type='data', symmetric=False, array=v['err_plus'], arrayminus=v['err_minus']),
    ))
    fig.update_layout(title=f'Bootstrap 95% intervals — {metric.upper()} ({h}h)', xaxis_title=metric.upper(), yaxis_title='Model')
    st.plotly_chart(fig, use_container_width=True)
    st.info('Wide intervals reflect the extremely small positive test samples and indicate substantial uncertainty.')

elif page == 'Patient / XAI Case':
    xcase = R.get('xcase')
    if xcase is None or xcase.empty:
        st.warning('xai_case.csv was not found.')
        st.stop()
    r = xcase.iloc[0]
    a, b, c, d = st.columns(4)
    a.metric('Subject ID', str(int(r.subject_id)))
    b.metric('ICU stay', str(int(r.icustay_id)))
    c.metric('Horizon', f'{int(r.horizon)}h')
    d.metric('Predicted risk', f'{100 * r.predicted_probability:.2f}%')
    st.write('**Model:**', r.model)
    st.write('**True label:**', 'Sepsis-3' if int(r.true_label) == 1 else 'Non-sepsis')

    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=100 * r.predicted_probability,
        number={'suffix': '%'},
        title={'text': 'Predicted Sepsis Risk'},
        gauge={'axis': {'range': [0, 100]}},
    ))
    st.plotly_chart(fig, use_container_width=True)

    if 'pred' not in R:
        st.info(
            'The downloaded Kaggle notebook did not embed the full held-out prediction table, '
            'so this dashboard displays the recoverable XAI patient rather than an arbitrary patient selector.'
        )

elif page == 'Local SHAP':
    df = R.get('local')
    source_note = 'Full Local SHAP output'
    if df is None:
        df = R.get('local_displayed')
        source_note = 'Top 15 Local SHAP rows recovered from the executed Kaggle notebook'
    if df is None:
        st.warning('No Local SHAP output was recoverable.')
        st.stop()

    if 'xcase' in R:
        r = R['xcase'].iloc[0]
        st.write(
            f"**Explained patient:** {int(r.subject_id)} | **Horizon:** {int(r.horizon)}h | "
            f"**Model:** {r.model} | **Risk:** {100 * r.predicted_probability:.2f}%"
        )
    st.caption(source_note)

    v = df.sort_values('absolute_contribution', ascending=False).head(15).copy()
    v['Direction'] = np.where(v.shap_contribution >= 0, 'Increases predicted risk', 'Decreases predicted risk')
    st.plotly_chart(
        px.bar(
            v.sort_values('shap_contribution'),
            x='shap_contribution',
            y='feature',
            color='Direction',
            orientation='h',
            title='Local Kernel SHAP',
        ),
        use_container_width=True,
    )
    st.dataframe(v.round(6), use_container_width=True, hide_index=True)
    st.info('SHAP explains model behaviour; it does not establish clinical causation.')

elif page == 'Temporal SHAP':
    df = R.get('temporal')
    full = True
    if df is None:
        df = R.get('temporal_displayed')
        full = False
    if df is None:
        st.warning('No Temporal SHAP output was recoverable.')
        st.stop()

    if full and df.prefix_hour.nunique() > 1:
        top = df.groupby('feature').absolute_contribution.mean().nlargest(12).index
        heat = df[df.feature.isin(top)].pivot(index='feature', columns='prefix_hour', values='shap_contribution').loc[top]
        fig = go.Figure(go.Heatmap(
            z=heat.values,
            x=[f'Hour {int(x)}' for x in heat.columns],
            y=heat.index,
            colorbar={'title': 'SHAP'},
        ))
        fig.update_layout(title='Temporal SHAP: contribution changes as more hours become available', height=650)
        st.plotly_chart(fig, use_container_width=True)
        feature = st.selectbox('Feature', list(top))
        st.plotly_chart(
            px.line(df[df.feature == feature], x='prefix_hour', y='shap_contribution', markers=True, title=f'Temporal SHAP — {feature}'),
            use_container_width=True,
        )
    else:
        st.warning(
            'Only the first displayed Temporal SHAP rows were recoverable from the downloaded notebook, '
            'so a complete 12-hour temporal heatmap cannot be reconstructed yet.'
        )
        st.dataframe(df.round(6), use_container_width=True, hide_index=True)

elif page == 'Attention Heatmap':
    df = R.get('attn')
    if df is None:
        st.warning(
            'Transformer attention output is unavailable. The executed Kaggle run captured 0 attention maps, '
            'so attention_matrix.csv and attention_hour_importance.csv were not generated.'
        )
        st.info('This is reported transparently as an incomplete XAI component rather than displaying fabricated attention values.')
        st.stop()

    if df.columns[0].lower().startswith('unnamed'):
        df = df.set_index(df.columns[0])
    A = df.astype(float)
    fig = go.Figure(go.Heatmap(
        z=A.values,
        x=[f'Hour {i}' for i in range(1, A.shape[1] + 1)],
        y=[f'Hour {i}' for i in range(1, A.shape[0] + 1)],
        zmin=0,
        colorbar={'title': 'Attention'},
    ))
    fig.update_layout(title='Mean Transformer attention across layers and heads', xaxis_title='Key hour', yaxis_title='Query hour', height=650)
    st.plotly_chart(fig, use_container_width=True)
    if 'attn_hour' in R:
        st.plotly_chart(px.bar(R['attn_hour'], x='hour', y='mean_attention_received', title='Attention received by each observation hour'), use_container_width=True)
    st.info('Attention indicates where the Transformer allocated attention. It is not causal evidence.')

elif page == 'Result Availability':
    st.subheader('Recovered Kaggle Result Files')
    if RES:
        st.write('**Loaded result directory:**', str(RES))
    else:
        st.warning('No result directory was found.')
        st.stop()

    rows = []
    for key, obj in R.items():
        rows.append({
            'Result key': key,
            'Rows': len(obj),
            'Columns': len(obj.columns),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown('### Current limitations')
    st.write(
        'The executed notebook allowed recovery of model metrics, CV summary, bootstrap confidence intervals, '
        'one XAI case, the displayed Local SHAP top-15 rows and the first displayed Temporal SHAP rows. '
        'The full held-out prediction table, full cross-validation fold table and full Temporal SHAP CSV were not embedded in the downloaded notebook. '
        'Transformer attention extraction also failed in that run, so no attention matrix is currently available.'
    )
