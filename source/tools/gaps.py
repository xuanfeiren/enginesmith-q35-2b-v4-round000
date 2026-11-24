import pickle, sys, collections, re
d=sys.argv[1]; core=sys.argv[2] if len(sys.argv)>2 else '0'; thr=float(sys.argv[3]) if len(sys.argv)>3 else 1500
pe, ham = pickle.load(open(d+'/pe.pkl','rb'))
allpe = sorted(pe[(core,'TensorMatrix')] + pe[(core,'Tensor')])
c=collections.Counter(); n=collections.Counter()
for i in range(1,len(allpe)):
    g = allpe[i][0]-(allpe[i-1][0]+allpe[i-1][1])
    if g>thr:
        w = re.findall(r'S\[\d+\] \((\w+)\)', allpe[i][3]) or re.findall(r'(\w+Dynamic\w*|\$S\[\d+\])', allpe[i][3])
        key=(allpe[i][2], ','.join(w) if w else allpe[i][3][:30])
        c[key]+=g; n[key]+=1
for k,v in c.most_common(30): print(f'{k}: {v/1e3:.1f} us ({n[k]})')
