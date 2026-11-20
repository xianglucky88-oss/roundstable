from . import accumulator,intervals,polynomials,sketch,fourier,groupstats

def sum_call(d):
    a=accumulator.Accumulator()
    for value in d['values']:a.add(value)
    return {'value':a.finalize(),'count':a.count,'state':a.dump()}
def interval_call(d):
    variables={k:intervals.Interval(*v) for k,v in d['variables'].items()};r=intervals.evaluate(d['expression'],variables)
    return {'lower':r.lower,'upper':r.upper}
def sketch_call(d):
    s=sketch.Sketch(d['dimension'],d.get('rows',2));s.append(d['values']);return s.dump()
def fourier_call(d):
    values=[complex(*x) if isinstance(x,list) else complex(x) for x in d['values']]
    return {'values':[[x.real,x.imag] for x in fourier.transform(values,d.get('inverse',False))]}
def groupstats_call(d):
    if 'state' in d:
        if any(k in d for k in ('on_invalid','variance','capacity')):raise ValueError('conventions are fixed by the restored state')
        stats=groupstats.GroupStats.load(d['state'])
    else:
        stats=groupstats.GroupStats(d.get('on_invalid','reject'),d.get('variance','sample'),d.get('capacity',groupstats.DEFAULT_CAPACITY))
    records=d.get('records',[])
    if not isinstance(records,list):raise ValueError('records must be a list')
    for record in records:
        if not isinstance(record,dict):raise ValueError('record objects required')
        stats.add(record.get('key'),record.get('value'),record.get('weight'))
    out=stats.finalize();out['state']=stats.dump();return out
OPERATIONS={'accumulator':sum_call,'intervals':interval_call,'polynomials':lambda d:{'roots':polynomials.isolate(d['coefficients'])},'sketch':sketch_call,'fourier':fourier_call,'groupstats':groupstats_call}
EXAMPLES={
 'accumulator':{'values':[1e16,1,-1e16]},
 'intervals':{'variables':{'x':[1,2],'y':[3,4]},'expression':['mul',['var','x'],['var','y']]},
 'polynomials':{'coefficients':[-1,0,1]},'sketch':{'dimension':3,'rows':2,'values':[[1,0,0],[0,1,0],[1,1,0],[2,0,0],[0,2,0]]},
 'fourier':{'values':[1,2,3,4]},
 'groupstats':{'on_invalid':'skip','variance':'sample','records':[{'key':'control','value':1.5},{'key':'control','value':2.5,'weight':2},{'key':'control','value':None},{'key':'treated','value':3.25},{'key':'treated','value':3.75,'weight':0}]},
}
