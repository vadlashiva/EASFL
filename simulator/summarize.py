import json, numpy as np
R=json.load(open('results.json'))
ms=lambda a:(np.mean(a),np.std(a))
def fmt(a,d=3):
    a=[x for x in a if x is not None]
    if not a: return "n/a(dead<200)"
    m,s=ms(a); return f"{m:.{d}f}±{s:.{d}f}"
print('== detection at r200')
for name,rows in [('CL',[{'final':m} for m in R['central']]),('Local',[{'final':m} for m in R['local']])]+[(sc,[r for r in R['learn'] if r['scheme']==sc]) for sc in ['FL-flat','FL-LEACH','EASFL']]:
    line=name+' '+' '.join(k+'='+fmt([r['final'][k] for r in rows]) for k in ['acc','prec','rec','f1','fpr','auc'])
    if 'mbit' in rows[0]: line+=f" Mbit={fmt([r['mbit'] for r in rows],1)} E={fmt([r['energy'] for r in rows],2)} alive={fmt([r['alive'] for r in rows],1)}"
    print(line)
print('== curves (mean f1, auc) every 20 rounds')
for sc in ['FL-flat','FL-LEACH','EASFL']:
    rows=[r for r in R['learn'] if r['scheme']==sc]
    C=np.array([r['curve'] for r in rows])  # seeds x T x 3
    print(sc,[(int(C[0,i,0]),round(C[:,i,1].mean(),3),round(C[:,i,2].mean(),3)) for i in range(1,20,2)])
print('== malicious')
for m in [0.0,0.1,0.2,0.3]:
    for sc,rob in [('FL-LEACH',None),('EASFL',False),('EASFL',None)]:
        rows=[r for r in R['mal'] if r['scheme']==sc and r['kw']['mal_frac']==m and r['kw'].get('robust',None)==rob]
        print(m,sc,rob,'f1',fmt([r['final']['f1'] for r in rows]),'auc',fmt([r['final']['auc'] for r in rows]))
print('== loss')
for p in [0.0,0.1,0.2,0.3]:
    for sc in ['FL-LEACH','EASFL']:
        rows=[r for r in R['loss'] if r['scheme']==sc and r['kw']['p_loss']==p]
        print(p,sc,'f1',fmt([r['final']['f1'] for r in rows]),'auc',fmt([r['final']['auc'] for r in rows]))
print('== lifetime')
for r0 in R['life']: pass
keys=[]
for r in R['life']:
    k=(r['scheme'],json.dumps(r['kw'],sort_keys=True))
    if k not in keys: keys.append(k)
for k in keys:
    rows=[r for r in R['life'] if (r['scheme'],json.dumps(r['kw'],sort_keys=True))==k]
    print(k,'FND',fmt([r['fnd'] for r in rows],0),'HND',fmt([r['hnd'] for r in rows],0),'LND',fmt([r['lnd'] for r in rows],0),'E200',fmt([r['e200'] for r in rows],2))
print('== rate')
for S in [10,30,60,120,240]:
    for ea in [None,True]:
        rows=[r for r in R['rate'] if r['kw']['samples_per_round']==S and r['kw'].get('ea_ch')==ea]
        print(S,ea,'FND',fmt([r['fnd'] for r in rows],0),'HND',fmt([r['hnd'] for r in rows],0),'E200',fmt([r['e200'] for r in rows],2))
