from . import accumulator,intervals,polynomials,sketch,fourier

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
OPERATIONS={'accumulator':sum_call,'intervals':interval_call,'polynomials':lambda d:{'roots':polynomials.isolate(d['coefficients'])},'sketch':sketch_call,'fourier':fourier_call}
EXAMPLES={
 'accumulator':{'values':[1e16,1,-1e16]},
 'intervals':{'variables':{'x':[1,2],'y':[3,4]},'expression':['mul',['var','x'],['var','y']]},
 'polynomials':{'coefficients':[-1,0,1]},'sketch':{'dimension':3,'rows':2,'values':[[1,0,0],[0,1,0],[1,1,0],[2,0,0],[0,2,0]]},
 'fourier':{'values':[1,2,3,4]},
}
