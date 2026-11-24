"""ham.py DIR [core]: throttle (ham type 1) intervals vs PE activity; attributes each throttle to the PE gap before it."""
import json, sys, os, pickle, collections, bisect
d = sys.argv[1]
core = sys.argv[2] if len(sys.argv) > 2 else '0'
pk = os.path.join(d, 'pe.pkl')
if not os.path.exists(pk):
    j = json.load(open(os.path.join(d, 'ntff.json')))
    pe = {}
    for x in j['instruction']:
        if x['label'] in ('TensorMatrix', 'Tensor'):
            c = x['fully_qualified_subgraph'][-1]
            pe.setdefault((c, x['label']), []).append((x['timestamp'], x['duration'], x.get('nki_source_location', '').split(':')[-1], x.get('operands', '')[:60]))
    ham = {}
    for x in j['ham']:
        ham.setdefault(x['subgraph'][-1], []).append((x['timestamp'], x['duration'], x['type'], x['k']))
    pickle.dump((pe, ham), open(pk, 'wb'))
pe, ham = pickle.load(open(pk, 'rb'))
mm = sorted(pe[(core, 'TensorMatrix')])
ld = sorted(pe[(core, 'Tensor')])
allpe = sorted(mm + ld)
hm = sorted(ham[core])
t0 = allpe[0][0]; t1 = allpe[-1][0] + allpe[-1][1]
thr = [(t, t + du) for t, du, ty, k in hm if ty == 1]
tot_thr = sum(b - a for a, b in thr)
print('throttle intervals (us):', sorted([round((b-a)/1e3,1) for a,b in thr])[-5:])
print(f'core {core}: span {(t1 - t0)/1e3:.1f} us, throttled {tot_thr/1e3:.1f} us in {len(thr)} intervals')
# matrix busy time inside throttled intervals
mg = []
for x0, du, *_ in mm:
    x1 = x0 + du
    if mg and x0 <= mg[-1][1]:
        mg[-1][1] = max(mg[-1][1], x1)
    else:
        mg.append([x0, x1])
starts = [x[0] for x in mg]
def busy(a, b, lst=mg):
    s = 0
    i = bisect.bisect_left(starts, a - 5000)
    while i < len(lst) and lst[i][0] < b:
        x0, x1 = lst[i][0], lst[i][1]
        s += max(0, min(b, x1) - max(a, x0))
        i += 1
    return s
bt = sum(busy(a, b) for a, b in thr)
print(f'matrix busy while throttled {bt/1e3:.1f} us; matrix busy total {busy(t0, t1)/1e3:.1f} us')
# attribute: largest PE gap (all PE instrs) in the 3.4us before each throttle start -> line of the instr after gap
ast = [x[0] for x in allpe]
cause = collections.Counter(); cause_n = collections.Counter()
for a, b in thr:
    i0 = bisect.bisect_left(ast, a - 6827); i1 = bisect.bisect_left(ast, a + 200)
    best = (0, '?')
    prev_end = None
    for i in range(max(i0, 1), min(i1 + 1, len(allpe))):
        pe_end = allpe[i - 1][0] + allpe[i - 1][1]
        g = allpe[i][0] - pe_end
        if g > best[0]:
            best = (g, allpe[i][2])
    cause[best[1]] += busy(a, b); cause_n[best[1]] += 1
print('throttled matrix-busy time by line after the largest PE gap in the preceding 6.8us:')
for k, v in cause.most_common(25):
    print(f'  line {k}: {v/1e3:.1f} us ({cause_n[k]} intervals)')
# gaps > 1.5us overall by next line
gl = collections.Counter(); gn = collections.Counter()
for i in range(1, len(allpe)):
    g = allpe[i][0] - (allpe[i - 1][0] + allpe[i - 1][1])
    if g > 1500:
        gl[allpe[i][2]] += g; gn[allpe[i][2]] += 1
print('PE gaps > 1.5us by next line:', sum(gl.values())/1e3, 'us')
for k, v in gl.most_common(25):
    print(f'  line {k}: {v/1e3:.1f} us ({gn[k]})')
