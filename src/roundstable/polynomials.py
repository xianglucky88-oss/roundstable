from fractions import Fraction as F

def trim(p):
    p=list(map(F,p))
    while len(p)>1 and p[-1]==0:p.pop()
    return p

def value(p,x):
    out=F(0)
    for c in reversed(p):out=out*x+c
    return out

def derivative(p):return trim([i*p[i] for i in range(1,len(p))] or [0])
def divmod_poly(a,b):
    a,b=trim(a),trim(b)
    if b==[0]:raise ValueError("zero polynomial divisor")
    quotient=[F(0)]*max(1,len(a)-len(b)+1)
    while a!=[0] and len(a)>=len(b):
        n=len(a)-len(b);c=a[-1]/b[-1];quotient[n]+=c
        for i,x in enumerate(b):a[i+n]-=c*x
        a=trim(a)
    return trim(quotient),a

def sturm(p):
    a=trim(p)
    if len(a)<2:raise ValueError("nonconstant polynomial required")
    b=derivative(a);seq=[a,b]
    while b!=[0]:
        _,r=divmod_poly(a,b)
        if r==[0]:break
        r=[-x for x in r];seq.append(r);a,b=b,r
    return seq

def variations(seq,x,side):
    signs=[]
    for p in seq:
        q=p;order=0;v=value(q,x)
        while v==0 and len(q)>1:q=derivative(q);order+=1;v=value(q,x)
        if v:signs.append((1 if v>0 else -1)*(-1 if side<0 and order%2 else 1))
    return sum(a!=b for a,b in zip(signs,signs[1:]))

def isolate(coefficients,width=F(1,1024),budget=10000):
    p=trim(coefficients)
    if len(p)<2 or len(p)>21 or width<=0:raise ValueError("unsupported polynomial or width")
    # Sturm counts distinct roots even with a nontrivial derivative gcd.
    seq=sturm(p);bound=F(2)+max(abs(c/p[-1]) for c in p[:-1]);stack=[(-bound,bound)];out=[];work=0
    while stack:
        a,b=stack.pop();work+=1
        if work>budget:raise ValueError("isolation budget exceeded")
        n=variations(seq,a,1)-variations(seq,b,-1)
        if not n:continue
        if n==1 and b-a<=width:out.append({"lower":str(a),"upper":str(b),"exact":False});continue
        m=(a+b)/2
        if value(p,m)==0:out.append({"lower":str(m),"upper":str(m),"exact":True})
        stack.extend([(m,b),(a,m)])
    return sorted(out,key=lambda r:F(r['lower']))

def verify(coefficients,certificate):
    p=trim(coefficients);seq=sturm(p)
    a,b=F(certificate['lower']),F(certificate['upper'])
    if a==b:return bool(certificate['exact']) and value(p,a)==0
    return a<b and not certificate['exact'] and variations(seq,a,1)-variations(seq,b,-1)==1
