"""attribute each throttle interval to the largest PE gap in the 6.8us before it: (line after gap, waited engine, line before gap)"""
import pickle, sys, bisect, collections, re
d=sys.argv[1]; core=sys.argv[2] if len(sys.argv)>2 else '0'
pe, ham = pickle.load(open(d+'/pe.pkl','rb'))
mm = sorted(pe[(core,'TensorMatrix')]); allpe = sorted(mm + pe[(core,'Tensor')])
mg=[]
for x0,du,*_ in mm:
    x1=x0+du
    if mg and x0<=mg[-1][1]: mg[-1][1]=max(mg[-1][1],x1)
    else: mg.append([x0,x1])
st=[m[0] for m in mg]
def busy(a,b):
    s=0; i=max(0,bisect.bisect_left(st,a)-1)
    while i<len(mg) and mg[i][0]<b:
        s+=max(0,min(b,mg[i][1])-max(a,mg[i][0])); i+=1
    return s
ast=[x[0] for x in allpe]
thr=[(t,t+du) for t,du,ty,k in sorted(ham[core]) if ty==1 and du<200000]
cost=collections.Counter(); cnt=collections.Counter(); gl=collections.Counter()
for a,b in thr:
    i0=bisect.bisect_left(ast,a-6827); i1=bisect.bisect_left(ast,a+500)
    best=(0,None)
    for i in range(max(i0,1),min(i1+1,len(allpe))):
        g=allpe[i][0]-(allpe[i-1][0]+allpe[i-1][1])
        if g>best[0]: best=(g,i)
    if best[1] is None: key=('?',)
    else:
        i=best[1]; w=re.findall(r'S\[\d+\] \((\w+)\)', allpe[i][3]) or re.findall(r'(q\w+|\$S\[\d+\])', allpe[i][3])
        key=(allpe[i-1][2]+'->'+allpe[i][2], ','.join(w[:2]))
    cost[key]+=busy(a,b)/2; cnt[key]+=1; gl[key]+=best[0]
tot=sum(cost.values())
print(f'est. throttle cost {tot/1e3:.0f} us ({len(thr)} intervals)')
for k,v in cost.most_common(20): print(f'  {k}: {v/1e3:.1f} us ({cnt[k]}, mean gap {gl[k]/cnt[k]/1e3:.1f} us)')
