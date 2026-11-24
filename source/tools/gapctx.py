import pickle,sys,collections,re
d=sys.argv[1]; core=sys.argv[2]; line=sys.argv[3]; sem=sys.argv[4]
pe,ham=pickle.load(open(d+'/pe.pkl','rb'))
a=sorted(pe[(core,'TensorMatrix')]+pe[(core,'Tensor')])
t0=a[0][0]
c=collections.Counter(); n=collections.Counter()
for i in range(1,len(a)):
    g=a[i][0]-(a[i-1][0]+a[i-1][1])
    if g>1000 and a[i][2]==line and sem in a[i][3]:
        c[a[i-1][2]]+=g; n[a[i-1][2]]+=1
        if len(sys.argv)>5: print(f'{(a[i][0]-t0)/1e3:9.1f} gap {g/1e3:5.1f} prev {a[i-1][2]}')
for k,v in c.most_common(): print('prev',k,v/1e3,n[k])
