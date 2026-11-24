import pickle, sys, bisect, collections
d=sys.argv[1]; core=sys.argv[2] if len(sys.argv)>2 else '0'
pe, ham = pickle.load(open(d+'/pe.pkl','rb'))
mm = sorted(pe[(core,'TensorMatrix')])
thr=[(t,t+du) for t,du,ty,k in sorted(ham[core]) if ty==1 and du<200000]
ts=[a for a,b in thr]
def inthr(t):
    i=bisect.bisect_right(ts,t)-1
    return i>=0 and t<thr[i][1]
tot=collections.Counter(); th=collections.Counter()
for x in mm:
    tot[x[2]]+=x[1]
    if inthr(x[0]): th[x[2]]+=x[1]
print('line: throttled-dur / total-dur (us)')
for k,v in th.most_common(20): print(f'  {k}: {v/1e3:.0f} / {tot[k]/1e3:.0f}  ({v/tot[k]*100:.0f}%)')
