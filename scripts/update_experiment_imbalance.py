from pathlib import Path
import textwrap
import nbformat as nbf

p = Path('notebooks/MIMICIII_Sepsis3_Experiment_Kaggle.ipynb')
nb = nbf.read(p, as_version=4)

# Add Zhang et al. class-imbalance rationale.
for c in nb.cells:
    if c.cell_type == 'markdown' and c.source.startswith('# Early Sepsis Prediction Experiment'):
        if 'Class-imbalance experiment based on Zhang et al. (2025)' not in c.source:
            c.source += textwrap.dedent(r'''

## Class-imbalance experiment based on Zhang et al. (2025)
Zhang et al. addressed class imbalance using balanced class weights. This notebook therefore compares **unweighted BCE** with **weighted BCE** so the effect of imbalance handling can be measured directly. Improvement is judged mainly using Recall, F1, AUPR, balanced accuracy, and confusion matrices rather than accuracy alone.

> Class weighting can improve minority-class detection, but it cannot create new positive patients or remove the small-sample limitation in the current 4h/8h datasets.
''')

# Configuration.
for c in nb.cells:
    if c.cell_type == 'code' and ('MODE=' in c.source or 'MODE = ' in c.source) and 'HORIZONS' in c.source:
        if 'IMBALANCE_STRATEGIES' not in c.source:
            c.source += '\nIMBALANCE_STRATEGIES=["unweighted","weighted"]\nPRIMARY_IMBALANCE_STRATEGY="weighted"\n'

# Insert rationale after existing class imbalance / preprocessing area.
if not any(c.cell_type == 'markdown' and 'Why compare unweighted and weighted loss?' in c.source for c in nb.cells):
    target = None
    for i,c in enumerate(nb.cells):
        if c.cell_type == 'markdown' and ('Class imbalance' in c.source or 'Weighted BCE' in c.source):
            target = i + 1
            break
    if target is None:
        target = 8
    nb.cells.insert(target, nbf.v4.new_markdown_cell(textwrap.dedent(r'''
### Why compare unweighted and weighted loss?

Zhang et al. (2025) used a balanced weighting strategy in which minority classes receive larger weights during optimisation. In this binary experiment:

\[
w_+=\frac{N_{negative}}{N_{positive}}
\]

Weighted BCE:

\[
\mathcal{L}_{weighted}=-\frac{1}{N}\sum_i\left[w_+y_i\log(p_i)+(1-y_i)\log(1-p_i)\right]
\]

The unweighted model uses ordinary BCE. Comparing both versions provides an ablation study that directly tests whether weighting improves Sepsis-3 detection. The weighted model should not be judged by accuracy alone; Recall, F1, AUPR, balanced accuracy, Precision and Specificity are examined together.
''').strip()))

# Replace training utilities, handling either compact or expanded notebook versions.
for c in nb.cells:
    if c.cell_type == 'code' and ('def train_one(' in c.source) and ('BCEWithLogitsLoss' in c.source):
        # Keep whichever prediction helper exists before train_one.
        pred_part = c.source.split('def train_one(',1)[0]
        c.source = pred_part + textwrap.dedent('''
        def train_one(name,xtr,ytr,xv,yv,epochs,imbalance_strategy="weighted"):
            # Support both build() and build_model() notebook variants.
            builder = build if "build" in globals() else build_model
            model = builder(name,xtr.shape[2]).to(DEVICE)
            pos=max(1,int(ytr.sum())); neg=max(1,int((ytr==0).sum())); pos_weight=neg/pos

            if imbalance_strategy=="weighted":
                loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight],dtype=torch.float32,device=DEVICE))
            elif imbalance_strategy=="unweighted":
                loss_fn=nn.BCEWithLogitsLoss()
            else:
                raise ValueError("imbalance_strategy must be unweighted or weighted")

            lr = LEARNING_RATE if "LEARNING_RATE" in globals() else LR
            patience_limit = EARLY_STOPPING_PATIENCE if "EARLY_STOPPING_PATIENCE" in globals() else PATIENCE
            opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=WEIGHT_DECAY)
            loader=DataLoader(TensorDataset(torch.tensor(xtr,dtype=torch.float32),torch.tensor(ytr,dtype=torch.float32)),batch_size=BATCH_SIZE,shuffle=True)
            best=None; best_val=float("inf"); wait=0; hist=[]

            for epoch in range(1,epochs+1):
                model.train(); losses=[]
                for xb,yb in loader:
                    xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad(); logits=model(xb); loss=loss_fn(logits,yb)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach().cpu()))

                model.eval()
                with torch.no_grad():
                    vl=float(loss_fn(model(torch.tensor(xv,dtype=torch.float32,device=DEVICE)),torch.tensor(yv,dtype=torch.float32,device=DEVICE)).cpu())
                tl=float(sum(losses)/len(losses)); hist.append({"epoch":epoch,"train_loss":tl,"val_loss":vl})
                print(f"{name:17s} {imbalance_strategy:10s} epoch={epoch:02d} train={tl:.4f} val={vl:.4f}")

                if vl < best_val - 1e-6:
                    best_val=vl; best=model.state_dict(); wait=0
                else:
                    wait += 1
                if wait >= patience_limit:
                    break

            if best is not None:
                model.load_state_dict(best)
            return model,pd.DataFrame(hist),pos_weight
        ''').strip()
        break

# Replace CV utility when present.
for c in nb.cells:
    if c.cell_type == 'code' and 'def run_cv(' in c.source:
        c.source = textwrap.dedent('''
        def run_cv(h,name,d,dev_idx,epochs,imbalance_strategy="weighted"):
            x=d["x"].astype("float32"); y=d["y"].astype(int); xd,yd=x[dev_idx],y[dev_idx]
            if __import__('numpy').bincount(yd,minlength=2).min() < CV_FOLDS:
                print("CV skipped: too few minority samples."); return pd.DataFrame()
            skf=StratifiedKFold(CV_FOLDS,shuffle=True,random_state=SEED); rows=[]
            for fold,(tr,va) in enumerate(skf.split(xd,yd),1):
                if 'fit_preprocessor' in globals():
                    imp,sc=fit_preprocessor(xd[tr]); xtr=transform_x(xd[tr],imp,sc); xv=transform_x(xd[va],imp,sc)
                else:
                    imp,sc=fit_prep(xd[tr]); xtr=tx(xd[tr],imp,sc); xv=tx(xd[va],imp,sc)
                m,_,_=train_one(name,xtr,yd[tr],xv,yd[va],epochs,imbalance_strategy)
                predictor = predict_scores if 'predict_scores' in globals() else predict
                pack = metrics if 'metrics' in globals() else metric_pack
                s=predictor(m,xv); r=pack(yd[va],s)
                r.update({"horizon":h,"model":name,"imbalance_strategy":imbalance_strategy,"fold":fold,"validation_positives":int(yd[va].sum())}); rows.append(r)
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            return pd.DataFrame(rows)
        ''').strip()

# Locate main experiment cell and replace it.
for c in nb.cells:
    if c.cell_type == 'code' and ('result_rows' in c.source or 'res=[]' in c.source) and ('for h' in c.source) and ('MODELS' in c.source):
        c.source = textwrap.dedent('''
        epochs = PILOT_EPOCHS if MODE=="pilot" else FINAL_EPOCHS
        predictor = predict_scores if 'predict_scores' in globals() else predict
        pack = metrics if 'metrics' in globals() else metric_pack
        dataset_dict = processed if 'processed' in globals() else proc

        trained={}; histories={}; result_rows=[]; pred_frames=[]; cv_frames=[]

        for h,pdata in dataset_dict.items():
            trained[h]={}; histories[h]={}
            for strategy in IMBALANCE_STRATEGIES:
                trained[h][strategy]={}; histories[h][strategy]={}
                for name in MODELS:
                    print("\\n"+"="*80+f"\\n{h}h — {name} — {strategy}\\n"+"="*80)
                    m,hist,w=train_one(name,pdata["x_train"],pdata["y_train"],pdata["x_val"],pdata["y_val"],epochs,strategy)
                    trained[h][strategy][name]=m; histories[h][strategy][name]=hist
                    s=predictor(m,pdata["x_test"]); r=pack(pdata["y_test"],s)
                    r.update({"horizon":h,"model":name,"imbalance_strategy":strategy,"analysis":"threshold_0.5","positive_weight_if_weighted":w,"test_positives":int(pdata["y_test"].sum())})
                    result_rows.append(r)

                    if MODE=="final" and 'run_cv' in globals():
                        d=(clean[h] if 'clean' in globals() else data[h])
                        cv=run_cv(h,name,d,splits[h]["dev"],epochs,strategy)
                        if len(cv): cv_frames.append(cv)

        results_df=pd.DataFrame(result_rows)
        display(results_df.sort_values(["horizon","model","imbalance_strategy"]).round(4))
        results_df.to_csv((OUTPUT_DIR if 'OUTPUT_DIR' in globals() else OUT)/"model_metrics.csv",index=False)
        cv_df=pd.concat(cv_frames,ignore_index=True) if cv_frames else pd.DataFrame()
        if len(cv_df): cv_df.to_csv((OUTPUT_DIR if 'OUTPUT_DIR' in globals() else OUT)/"cross_validation_metrics.csv",index=False)
        ''').strip()
        break

# Add an explicit imbalance comparison table before confusion matrices / result plots.
if not any(c.cell_type == 'markdown' and c.source.startswith('## Class-imbalance strategy comparison') for c in nb.cells):
    insert_at = len(nb.cells)-3
    for i,c in enumerate(nb.cells):
        if c.cell_type == 'markdown' and ('Confusion matrices' in c.source or 'ROC' in c.source):
            insert_at=i; break
    nb.cells.insert(insert_at, nbf.v4.new_markdown_cell(textwrap.dedent(r'''
## Class-imbalance strategy comparison

For every model and horizon, compare unweighted and weighted training. The main quantities are:

\[
\Delta Recall = Recall_{weighted}-Recall_{unweighted}
\]

\[
\Delta F1 = F1_{weighted}-F1_{unweighted}
\]

\[
\Delta AUPR = AUPR_{weighted}-AUPR_{unweighted}
\]

Positive values indicate improved minority-class detection, but any reduction in Precision or Specificity must also be reported.
''').strip()))
    nb.cells.insert(insert_at+1, nbf.v4.new_code_cell(textwrap.dedent('''
    rows=[]
    for h in sorted(results_df.horizon.unique()):
        for name in MODELS:
            b=results_df[(results_df.horizon==h)&(results_df.model==name)].set_index("imbalance_strategy")
            if {"unweighted","weighted"}.issubset(b.index):
                rows.append({
                    "horizon":h,"model":name,
                    "recall_unweighted":b.loc["unweighted","recall"],"recall_weighted":b.loc["weighted","recall"],"delta_recall":b.loc["weighted","recall"]-b.loc["unweighted","recall"],
                    "f1_unweighted":b.loc["unweighted","f1"],"f1_weighted":b.loc["weighted","f1"],"delta_f1":b.loc["weighted","f1"]-b.loc["unweighted","f1"],
                    "aupr_unweighted":b.loc["unweighted","aupr"],"aupr_weighted":b.loc["weighted","aupr"],"delta_aupr":b.loc["weighted","aupr"]-b.loc["unweighted","aupr"],
                    "precision_unweighted":b.loc["unweighted","precision"],"precision_weighted":b.loc["weighted","precision"],
                    "specificity_unweighted":b.loc["unweighted","specificity"],"specificity_weighted":b.loc["weighted","specificity"]
                })
    imbalance_comparison_df=pd.DataFrame(rows)
    display(imbalance_comparison_df.round(4))
    imbalance_comparison_df.to_csv((OUTPUT_DIR if 'OUTPUT_DIR' in globals() else OUT)/"imbalance_strategy_comparison.csv",index=False)
    ''').strip()))

# Update result-analysis guidance.
for c in nb.cells:
    if c.cell_type == 'markdown' and ('Result-analysis framework' in c.source or 'Result analysis' in c.source):
        if 'Imbalance-handling analysis' not in c.source:
            c.source += textwrap.dedent('''

### Imbalance-handling analysis
Compare the weighted and unweighted version of every model. Do not state that class weighting “solved” imbalance merely because Recall increased. Examine Precision, Recall, F1, AUPR, balanced accuracy, cross-validation variation, and confidence intervals together. With the current small number of positive patients, even a large apparent gain may be unstable.
''')

# Ensure SHAP uses the weighted model when nested model dictionary exists.
for c in nb.cells:
    if c.cell_type == 'code' and 'RUN_LOCAL_SHAP' in c.source and 'trained[' in c.source:
        c.source = c.source.replace('trained[h][name]', 'trained[h][PRIMARY_IMBALANCE_STRATEGY][name]')

# Validate Python cells.
for i,c in enumerate(nb.cells):
    if c.cell_type == 'code':
        compile(c.source, f'<cell {i}>', 'exec')

nbf.validate(nb)
nbf.write(nb,p)
print('Updated', p)
