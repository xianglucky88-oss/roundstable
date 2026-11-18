from fractions import Fraction as F
import itertools,math,random
import pytest,numpy as np
from roundstable import accumulator,intervals,polynomials,sketch,fourier

def test_accumulation_partitions_and_state():
    values=[1e16,1.,-1e16,3.25,-.125]
    expected=float(sum(map(F.from_float,values),F(0)))
    for order in itertools.permutations(values):
        a,b=accumulator.Accumulator(),accumulator.Accumulator()
        for x in order[:2]:a.add(x)
        for x in order[2:]:b.add(x)
        restored=accumulator.Accumulator.load(a.dump());assert restored.merge(b).finalize()==expected
    assert math.isnan(accumulator.Accumulator().add(float('inf')).add(float('-inf')).finalize())
    assert math.copysign(1,accumulator.Accumulator().add(-0.).finalize())==1

def test_interval_outward_bounds_with_exact_rationals():
    rng=random.Random(9)
    for _ in range(60):
        a,b,c,d=[rng.random()*10 for _ in range(4)];a,b=sorted([a,b]);c,d=sorted([c,d])
        x,y=intervals.Interval(a,b),intervals.Interval(c,d)
        z=x*y+x
        for xx in [a,b]:
            for yy in [c,d]:
                exact=F.from_float(xx)*F.from_float(yy)+F.from_float(xx)
                assert F.from_float(z.lower)<=exact<=F.from_float(z.upper)
    with pytest.raises(ValueError):intervals.Interval(1,2)/intervals.Interval(-1,1)
@pytest.mark.parametrize('coefficients,roots',[
 ([-1,0,1],[-1,1]),([0,0,1],[0]),([2,-3,1],[1,2]),([1,-2,1],[1]),([1,0,1],[]),([0,-1,0,1],[-1,0,1])])
def test_exact_root_certificates(coefficients,roots):
    out=polynomials.isolate(coefficients)
    assert len(out)==len(roots)
    for certificate,root in zip(out,roots):
        assert polynomials.verify(coefficients,certificate)
        assert F(certificate['lower'])<=root<=F(certificate['upper'])
def test_root_resource_limit_and_zero_polynomial():
    with pytest.raises(ValueError):polynomials.isolate([0,0])
    with pytest.raises(ValueError):polynomials.isolate([-2,0,1],budget=1)
def test_matrix_sketch_covariance_error_and_state():
    matrix=np.array([[1.,2.,3.],[2.,0.,1.],[0.,1.,2.],[3.,1.,1.],[2.,2.,2.],[0.,0.,1.]])
    s=sketch.Sketch(3,2).append(matrix);b=s.matrix();error=matrix.T@matrix-b.T@b
    assert np.linalg.eigvalsh(error).min()>=-1e-9
    assert np.linalg.norm(error,2)<=np.linalg.norm(matrix,'fro')**2/2+1e-9
    restored=sketch.Sketch.load(s.dump());assert np.allclose(restored.matrix(),b)
    a=sketch.Sketch(3,2).append(matrix[:3]);c=sketch.Sketch(3,2).append(matrix[3:]);a.merge(c);assert a.count==6
    with pytest.raises(ValueError):s.append([[float('nan'),0,0]])
@pytest.mark.parametrize('n',[0,1,2,3,4,7,8,16])
def test_fourier_against_numpy(n):
    values=[complex(i%3-1,.2*i) for i in range(n)];out=fourier.transform(values)
    if not n:assert out==[];return
    assert np.allclose(out,np.fft.fft(values),atol=1e-10)
    assert np.allclose(fourier.transform(out,True),values,atol=1e-10)
