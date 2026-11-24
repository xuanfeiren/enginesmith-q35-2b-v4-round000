import pickle,sys
d=sys.argv[1]; line=sys.argv[2]
pe,ham=pickle.load(open(d+'/pe.pkl','rb'))
for c in '01':
    a=sorted(pe[(c,'TensorMatrix')]+pe[(c,'Tensor')])
    xs=[x for x in a if x[2]==line]
    t0=a[0][0]; 
    print(c, 'span',(a[-1][0]+a[-1][1]-t0)/1e3, 'line first',(xs[0][0]-t0)/1e3,'last',(xs[-1][0]+xs[-1][1]-t0)/1e3, 'n',len(xs), 'busy', sum(x[1] for x in xs)/1e3)
