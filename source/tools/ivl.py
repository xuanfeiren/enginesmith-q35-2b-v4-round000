import pickle,sys,collections
d=sys.argv[1]; line=sys.argv[2]; c=sys.argv[3] if len(sys.argv)>3 else '0'
pe,ham=pickle.load(open(d+'/pe.pkl','rb'))
a=sorted(pe[(c,'TensorMatrix')])
h=collections.Counter()
for i in range(len(a)-1):
    if a[i][2]==line and a[i+1][2]==line:
        dt=a[i+1][0]-a[i][0]; h[min(dt//10*10,1000)]+=1
tot=sum(h.values())
for k in sorted(h):
    if h[k]>tot*0.005: print(k,h[k])
print('dur sample',[x[1] for x in a if x[2]==line][:10])
