import cmath,math

def transform(values,inverse=False):
    n=len(values)
    if n>4096:raise ValueError("transform size limit")
    a=[complex(x) for x in values]
    if any(not math.isfinite(z.real) or not math.isfinite(z.imag) for z in a):raise ValueError("finite samples required")
    if not n:return []
    sign=1 if inverse else -1
    if n&(n-1):
        if n>64:raise ValueError("non-power-of-two length limit")
        out=[sum(v*cmath.exp(sign*2j*math.pi*k*t/n) for t,v in enumerate(a)) for k in range(n)]
    else:
        j=0
        for i in range(1,n):
            bit=n>>1
            while j&bit:j^=bit;bit>>=1
            j^=bit
            if i<j:a[i],a[j]=a[j],a[i]
        size=2
        while size<=n:
            root=cmath.exp(sign*2j*math.pi/size)
            for base in range(0,n,size):
                w=1
                for k in range(size//2):
                    u=a[base+k];v=a[base+k+size//2]*w
                    a[base+k],a[base+k+size//2]=u+v,u-v;w*=root
            size*=2
        out=a
    return [z/n for z in out] if inverse else out
