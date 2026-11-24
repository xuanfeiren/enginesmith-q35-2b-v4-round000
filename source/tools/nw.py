import pickle,sys
d=sys.argv[1]; core=sys.argv[2]
ins=pickle.load(open(d+'/all.pkl','rb'))
ins=sorted([x for x in ins if x[3]==core])
t0=ins[0][0]
last109=None; tot=0; n=0
for x in ins:
    if x[4]=='109' and x[6].startswith('DMA'): last109=x[0]
    if x[4]=='130' and x[6]=='ACTIVATE' or (x[4]=='130' and x[6]=='TENSOR_SCALAR'):
        if last109 is not None:
            w=(x[0]-last109)/1e3; tot+=w; n+=1
            print(f'{(x[0]-t0)/1e3:9.1f} {x[2]} wait-from-dma {w:6.1f}')
        last109=None
print('total',tot,n)
