import pickle,sys,collections
d=sys.argv[1]; c=sys.argv[2] if len(sys.argv)>2 else '0'
pe,ham=pickle.load(open(d+'/pe.pkl','rb'))
a=sorted(pe[(c,'TensorMatrix')])
# per line: count, sum of issue interval to next instr (occupancy)
occ=collections.Counter(); n=collections.Counter()
for i in range(len(a)-1):
    dt=a[i+1][0]-a[i][0]
    if dt<5000: occ[a[i][2]]+=dt; n[a[i][2]]+=1
tot=sum(occ.values())
for k,v in occ.most_common(30): print(f'{k:>6} {v/1e3:8.1f} us  n={n[k]:6d}  per={v/n[k]:6.0f} ns  {a[[x[2] for x in a].index(k)][3][:50]}')
print('total',tot/1e3)
