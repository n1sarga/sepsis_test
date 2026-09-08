from pathlib import Path
import json,numpy as np,pandas as pd,streamlit as st,plotly.express as px,plotly.graph_objects as go
st.set_page_config(page_title='Sepsis Prediction Dashboard',page_icon='🩺',layout='wide')

def find_data():
 for p in [Path.cwd()/'data',Path('/mount/src/sepsis_test/data')]:
  if (p/'dataset_manifest.json').exists(): return p
 for p in Path.cwd().glob('**/dataset_manifest.json'): return p.parent
 return None

def find_results():
 for p in [Path.cwd()/'results',Path('/mount/src/sepsis_test/results')]:
  if p.exists(): return p
 return None
@st.cache_data
def load_data(p):
 p=Path(p); c=pd.read_csv(p/'interim/mimic_iii_sepsis3_cohort.csv',low_memory=False); m=json.loads((p/'dataset_manifest.json').read_text()); T={}
 for h in [4,8]:
  q=p/f'processed/timeseries_before_onset_{h}h.npz'
  if q.exists():
   z=np.load(q,allow_pickle=False); T[h]={k:z[k] for k in z.files}; z.close()
 return c,m,T
@st.cache_data
def load_results(p):
 p=Path(p); names={'metrics':'model_metrics.csv','pred':'heldout_predictions.csv','cv':'cross_validation_metrics.csv','xcase':'xai_case.csv','local':'local_shap_feature_contributions.csv','temporal':'temporal_shap.csv','attn':'attention_matrix.csv','attn_hour':'attention_hour_importance.csv'}; out={}
 for k,n in names.items():
  q=p/n
  if q.exists(): out[k]=pd.read_csv(q)
 return out
DATA=find_data(); RES=find_results(); R=load_results(str(RES)) if RES else {}
st.title('Early Sepsis Prediction — Research Dashboard')
st.caption('MIMIC-III Sepsis-3 EDA, held-out predictions and explainable AI. Research prototype only; not a clinical decision tool.')
if DATA is None: st.error('data/ folder not found.'); st.stop()
cohort,manifest,tensors=load_data(str(DATA))
page=st.sidebar.radio('Section',['Overview','Eligibility','Missingness','Model Performance','Patient Predictions','Local SHAP','Temporal SHAP','Attention Heatmap'])

if page=='Overview':
 total=len(cohort); sep=int((cohort.sepsis_label==1).sum()); ctl=total-sep; a,b,c,d=st.columns(4); a.metric('Patients',f'{total:,}'); b.metric('Sepsis-3',f'{sep:,}'); c.metric('Controls',f'{ctl:,}'); d.metric('Prevalence',f'{100*sep/total:.1f}%')
 rows=[]
 for h,x in tensors.items():
  y=x['y'].astype(int); rows.append({'Horizon':f'{h}h','Tensor':str(x['x'].shape),'Sepsis':int(y.sum()),'Controls':int((y==0).sum()),'Missing %':100*np.isnan(x['x']).mean()})
 st.dataframe(pd.DataFrame(rows).round(2),use_container_width=True,hide_index=True)
 st.info('The original cohort is approximately balanced, but temporal-window eligibility creates severe class imbalance.')

elif page=='Eligibility':
 for col in ['intime','outtime','sepsis_onset_time']:
  if col in cohort: cohort[col]=pd.to_datetime(cohort[col],errors='coerce')
 pos=cohort[cohort.sepsis_label==1].copy(); pos['onset_hours']=(pos.sepsis_onset_time-pos.intime).dt.total_seconds()/3600
 st.plotly_chart(px.histogram(pos,x='onset_hours',nbins=70,title='Sepsis onset relative to ICU admission'),use_container_width=True)
 rows=[]
 for h in [4,8,12]:
  end=pos.sepsis_onset_time-pd.to_timedelta(h,unit='h'); start=end-pd.to_timedelta(12,unit='h'); ok=pos.sepsis_onset_time.notna()&start.ge(pos.intime)&end.le(pos.outtime); rows.append({'Horizon':f'{h}h','Eligible positives':int(ok.sum())})
 tab=pd.DataFrame(rows); st.dataframe(tab,use_container_width=True,hide_index=True); st.plotly_chart(px.bar(tab,x='Horizon',y='Eligible positives',text_auto=True),use_container_width=True)

elif page=='Missingness':
 h=st.selectbox('Horizon',sorted(tensors)); x=tensors[h]['x'].astype(float); y=tensors[h]['y'].astype(int); miss=np.isnan(x).mean((1,2)); a,b,c=st.columns(3); a.metric('Mean missingness',f'{100*miss.mean():.1f}%'); b.metric('Median missingness',f'{100*np.median(miss):.1f}%'); c.metric('All-missing windows',int(np.isnan(x).all((1,2)).sum())); df=pd.DataFrame({'Missing fraction':miss,'Class':np.where(y==1,'Sepsis-3','Non-sepsis')}); st.plotly_chart(px.histogram(df,x='Missing fraction',color='Class',nbins=30),use_container_width=True)

elif page=='Model Performance':
 df=R.get('metrics')
 if df is None: st.warning('Run the Kaggle experiment and extract streamlit_results.zip into results/.'); st.stop()
 h=st.selectbox('Horizon',sorted(df.horizon.unique())); v=df[df.horizon==h]; cols=[c for c in ['model','imbalance_strategy','accuracy','precision','recall','specificity','f1','balanced_accuracy','auroc','aupr','brier'] if c in v]; st.dataframe(v[cols].round(4),use_container_width=True,hide_index=True); metric=st.selectbox('Metric',[c for c in ['recall','f1','auroc','aupr','precision','balanced_accuracy'] if c in v]); st.plotly_chart(px.bar(v,x='model',y=metric,color='imbalance_strategy' if 'imbalance_strategy' in v else None,barmode='group',title=f'{metric.upper()} by model'),use_container_width=True)
 if 'cv' in R:
  st.markdown('### Five-fold cross-validation'); cv=R['cv']; st.dataframe(cv[cv.horizon==h].round(4),use_container_width=True,hide_index=True)

elif page=='Patient Predictions':
 df=R.get('pred')
 if df is None: st.warning('Held-out predictions not found in results/.'); st.stop()
 h=st.selectbox('Horizon',sorted(df.horizon.unique())); m=st.selectbox('Model',sorted(df[df.horizon==h].model.unique())); v=df[(df.horizon==h)&(df.model==m)]
 if 'imbalance_strategy' in v:
  s=st.selectbox('Imbalance strategy',sorted(v.imbalance_strategy.unique())); v=v[v.imbalance_strategy==s]
 v=v.sort_values('predicted_probability',ascending=False); st.dataframe(v.head(100),use_container_width=True,hide_index=True); sid=st.selectbox('Inspect patient',v.subject_id.astype(str)); r=v[v.subject_id.astype(str)==sid].iloc[0]; a,b,c=st.columns(3); a.metric('Predicted risk',f'{100*r.predicted_probability:.2f}%'); b.metric('True label','Sepsis-3' if int(r.true_label) else 'Non-sepsis'); c.metric('ICU stay',str(int(r.icustay_id))); st.plotly_chart(go.Figure(go.Indicator(mode='gauge+number',value=100*r.predicted_probability,number={'suffix':'%'},gauge={'axis':{'range':[0,100]}})),use_container_width=True)

elif page=='Local SHAP':
 df=R.get('local')
 if df is None: st.warning('Enable RUN_LOCAL_SHAP=True in Kaggle and export results.'); st.stop()
 if 'xcase' in R:
  r=R['xcase'].iloc[0]; st.write(f"**Explained patient:** {int(r.subject_id)} | **Horizon:** {int(r.horizon)}h | **Risk:** {100*r.predicted_probability:.2f}%")
 v=df.sort_values('absolute_contribution',ascending=False).head(15); v['Direction']=np.where(v.shap_contribution>=0,'Increases risk','Decreases risk'); st.plotly_chart(px.bar(v.sort_values('shap_contribution'),x='shap_contribution',y='feature',color='Direction',orientation='h',title='Local Kernel SHAP'),use_container_width=True); st.dataframe(v.round(4),use_container_width=True,hide_index=True); st.info('SHAP explains model behaviour; it does not establish clinical causation.')

elif page=='Temporal SHAP':
 df=R.get('temporal')
 if df is None: st.warning('Enable RUN_TEMPORAL_SHAP=True in Kaggle and export results.'); st.stop()
 top=df.groupby('feature').absolute_contribution.mean().nlargest(12).index; heat=df[df.feature.isin(top)].pivot(index='feature',columns='prefix_hour',values='shap_contribution').loc[top]; fig=go.Figure(go.Heatmap(z=heat.values,x=[f'Hour {int(x)}' for x in heat.columns],y=heat.index,colorbar={'title':'SHAP'})); fig.update_layout(title='Temporal SHAP: contribution changes as more hours become available',height=650); st.plotly_chart(fig,use_container_width=True); f=st.selectbox('Feature',list(top)); st.plotly_chart(px.line(df[df.feature==f],x='prefix_hour',y='shap_contribution',markers=True,title=f'Temporal SHAP — {f}'),use_container_width=True)

elif page=='Attention Heatmap':
 df=R.get('attn')
 if df is None: st.warning('Enable RUN_ATTENTION=True for a Transformer model and export results.'); st.stop()
 if df.columns[0].lower().startswith('unnamed'): df=df.set_index(df.columns[0]); A=df.astype(float); fig=go.Figure(go.Heatmap(z=A.values,x=[f'Hour {i}' for i in range(1,A.shape[1]+1)],y=[f'Hour {i}' for i in range(1,A.shape[0]+1)],zmin=0,colorbar={'title':'Attention'})); fig.update_layout(title='Mean Transformer attention across layers and heads',xaxis_title='Key hour',yaxis_title='Query hour',height=650); st.plotly_chart(fig,use_container_width=True)
 if 'attn_hour' in R: st.plotly_chart(px.bar(R['attn_hour'],x='hour',y='mean_attention_received',title='Attention received by each observation hour'),use_container_width=True)
 st.info('Attention indicates where the Transformer allocated attention. It is not causal evidence.')
