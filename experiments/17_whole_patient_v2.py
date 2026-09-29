import pandas as pd, numpy as np
def load():
    X=pd.read_csv('data/X_train.csv');y=pd.read_csv('data/y_train.csv');T=pd.read_csv('data/X_test.csv')
    v=X.merge(y,on='Index')
    return v,T
def pfit(t,s,deg=1):
    m=~np.isnan(s)
    n=m.sum()
    out=np.full((len(t),3),np.nan)  # fitted, slope, intercept
    if n==0: return out,n
    if n==1 or np.ptp(t[m])<1e-6:
        out[:,0]=np.nanmean(s[m]); return out,n
    c=np.polyfit(t[m],s[m],1)
    out[:,0]=np.polyval(c,t);out[:,1]=c[0];out[:,2]=np.polyval(c,t.min())
    return out,n
def build(df):
    df=df.sort_values(['patient_id','age']).copy()
    df['dd']=df['age']-df['age_at_diagnosis']
    df['visit']=df.groupby('patient_id').cumcount()
    df['nvis']=df.groupby('patient_id')['age'].transform('size')
    df['t0']=df['age']-df.groupby('patient_id')['age'].transform('min')
    df['trel']=df['age']-df.groupby('patient_id')['age'].transform('mean')
    df['has_off']=df.off.notna().astype(float); df['has_on']=df.on.notna().astype(float)
    df['pat_ledd_mean']=df.groupby('patient_id').ledd.transform('mean')
    df['ledd_ff']=df.groupby('patient_id').ledd.transform(lambda s:s.interpolate(limit_direction='both'))
    feats={}
    for col in ['off','on','ledd']:
        fit=np.full((len(df),3),np.nan);cnt=np.zeros(len(df))
        for pid,idx in df.groupby('patient_id').indices.items():
            t=df['age'].values[idx];s=df[col].values[idx].astype(float)
            o,n=pfit(t,s);fit[idx]=o;cnt[idx]=n
        df[f'{col}_pfit']=fit[:,0];df[f'{col}_pslope']=fit[:,1];df[f'{col}_pint']=fit[:,2];df[f'{col}_pn']=cnt
        g=df.groupby('patient_id')[col]
        df[f'{col}_pmean']=g.transform('mean');df[f'{col}_pmax']=g.transform('max');df[f'{col}_pmin']=g.transform('min')
        df[f'{col}_pstd']=g.transform('std')
        df[f'{col}_resid']=df[col]-df[f'{col}_pfit']
        # neighbours
        df[f'{col}_prev']=g.transform(lambda s:s.ffill().shift(1))
        df[f'{col}_next']=g.transform(lambda s:s.bfill().shift(-1))
    # rolling centered mean of off
    for w in [3,5]:
        df[f'off_roll{w}']=df.groupby('patient_id').off.transform(lambda s:s.rolling(w,center=True,min_periods=1).mean())
        df[f'on_roll{w}']=df.groupby('patient_id').on.transform(lambda s:s.rolling(w,center=True,min_periods=1).mean())
    df['onoff_ratio_p']=df['on_pmean']/df['off_pmean']
    df['gene']=df['gene'].fillna('NA').astype('category');df['cohort']=df['cohort'].astype('category')
    return df

def fitcols(df,col,degs=(1,2),key='age'):
    for deg in degs:
        fit=np.full(len(df),np.nan)
        for pid,idx in df.groupby('patient_id').indices.items():
            t=df[key].values[idx];s=df[col].values[idx].astype(float);m=~np.isnan(s)
            if m.sum()>deg+1 and np.ptp(t[m])>0.5:
                fit[idx]=np.polyval(np.polyfit(t[m]-t.mean(),s[m],deg),t-t.mean())
            elif m.sum()>0: fit[idx]=np.nanmean(s[m]) if deg==1 else np.nan
        df[f'{col}_fit{deg}']=fit
    return df
def build2(df):
    df=build(df)
    df['est_off']=df.off+6
    df['est_on']=df.on/0.49
    df['est']=np.where(df.off.notna(),df.est_off,df.est_on)
    df['est_avg']=df[['est_off','est_on']].mean(axis=1)
    for c in ['est','est_avg','off']:
        df=fitcols(df,c)
    df['est_pmean']=df.groupby('patient_id').est.transform('mean')
    df['n_off_ratio']=df.off_pn/df.nvis
    return df
import pandas as pd,numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor as H
from sklearn.model_selection import GroupKFold
v,T=load()
tr=build2(v).sort_index();te=build2(T.assign(target=np.nan)).sort_index()
y=tr.target.values;splits=list(GroupKFold(5).split(tr,y,tr.patient_id))
V=['off','on','ledd','time_since_intake_on','time_since_intake_off','age','dd','age_at_diagnosis','sexM','gene','cohort','ledd_ff']
S0={'e_off':(['off','time_since_intake_off','ledd','ledd_ff','dd','age'],'off'),
    'e_on':(['on','time_since_intake_on','ledd','ledd_ff','dd','age'],'on'),
    'e_vis':(V,None)}
mk=lambda:H(max_iter=400,learning_rate=0.05,categorical_features='from_dtype',random_state=0)
for name,(F,req) in S0.items():
    mtr=tr[req].notna().values if req else np.ones(len(tr),bool)
    mte=te[req].notna().values if req else np.ones(len(te),bool)
    o=np.full(len(tr),np.nan)
    for a,b in splits:
        a=a[mtr[a]];m=mk().fit(tr.iloc[a][F],y[a]);o[b]=m.predict(tr.iloc[b][F])
    o[~mtr]=np.nan;tr[name]=o
    m=mk().fit(tr[mtr][F],y[mtr]);p=m.predict(te[F]);p[~mte]=np.nan;te[name]=p
for d in (tr,te):
    for c in S0:
        fitcols(d,c,(1,2));d[c+'_pm']=d.groupby('patient_id')[c].transform('mean')
F=[c for c in tr.columns if c not in ['Index','patient_id','target']]
mk1=lambda s:H(max_iter=2000,learning_rate=0.03,max_leaf_nodes=31,min_samples_leaf=40,l2_regularization=1,categorical_features='from_dtype',random_state=s)
preds=[]
for s in [0,1,2]:
    preds.append(mk1(s).fit(tr[F],y).predict(te[F]))
te['oof']=np.mean(preds,axis=0)
fitcols(te,'oof',(2,))
te['target']=te.oof_fit2.fillna(te.oof).clip(0,132)
sub=pd.read_csv('data/sample_submission.csv')[['Index']].merge(te[['Index','target']],on='Index',how='left')
assert sub.target.notna().all() and len(sub)==11013
sub.to_csv('submission_17_whole_patient_v2.csv',index=False);print(sub.target.describe())
