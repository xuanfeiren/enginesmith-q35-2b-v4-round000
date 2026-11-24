import json,sys,pickle,os
d=sys.argv[1]; core=sys.argv[2]; tc=float(sys.argv[3])*1e3; w=float(sys.argv[4])*1e3
pk=d+'/all.pkl'
if not os.path.exists(pk):
    j=json.load(open(d+'/ntff.json'))
    ins=[(x['timestamp'],x['duration'],x['label'],x['fully_qualified_subgraph'][-1],x.get('nki_source_location','').split(':')[-1],x.get('operands','')[:70], x.get('opcode','')) for x in j['instruction']]
    pickle.dump(ins,open(pk,'wb'))
ins=pickle.load(open(pk,'rb'))
ins=[x for x in ins if x[3]==core]
t0=min(x[0] for x in ins)
for x in sorted(ins):
    t=x[0]-t0
    if tc-w<=t<=tc+w and x[2] not in ('TensorMatrix','Tensor'):
        print(f'{t/1e3:9.2f} {x[1]/1e3:6.2f} {x[2]:12s} L{x[4]:>5} {x[6]} {x[5][:60]}')
