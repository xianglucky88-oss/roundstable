import math
from . import accumulator,intervals,polynomials,sketch,fourier,periodicity

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
def periodicity_call(d):
    return periodicity.analyze(
        d['values'],interval=d.get('interval',1.0),mask=d.get('mask'),times=d.get('times'),
        trend=d.get('trend','linear'),noise=d.get('noise','gaussian'),alpha=d.get('alpha',0.05),
        budget=d.get('budget'),seed=d.get('seed',0),min_window=d.get('min_window',8),
        max_periods=d.get('max_periods',5),min_cycles=d.get('min_cycles',2))
OPERATIONS={'accumulator':sum_call,'intervals':interval_call,'polynomials':lambda d:{'roots':polynomials.isolate(d['coefficients'])},'sketch':sketch_call,'fourier':fourier_call,'periodicity':periodicity_call}
EXAMPLES={
 'accumulator':{'values':[1e16,1,-1e16]},
 'intervals':{'variables':{'x':[1,2],'y':[3,4]},'expression':['mul',['var','x'],['var','y']]},
 'polynomials':{'coefficients':[-1,0,1]},'sketch':{'dimension':3,'rows':2,'values':[[1,0,0],[0,1,0],[1,1,0],[2,0,0],[0,2,0]]},
 'fourier':{'values':[1,2,3,4]},
 'periodicity':{'values':[round(3.0*math.sin(2*math.pi*t/8)+0.25*((t*37)%7-3),6) for t in range(64)],'interval':1.0,'trend':'linear','noise':'gaussian','alpha':0.05,'budget':50000,'seed':1234},
}
