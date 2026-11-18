from dataclasses import dataclass
import math

def down(v):return math.nextafter(v,-math.inf)
def up(v):return math.nextafter(v,math.inf)
@dataclass(frozen=True)
class Interval:
    lower:float
    upper:float
    def __post_init__(self):
        if math.isnan(self.lower) or math.isnan(self.upper) or self.lower>self.upper:raise ValueError("invalid interval")
    def __add__(self,o):
        a,b=self.lower+o.lower,self.upper+o.upper
        if math.isnan(a) or math.isnan(b):return Interval(-math.inf,math.inf)
        return Interval(down(a),up(b))
    def __neg__(self):return Interval(-self.upper,-self.lower)
    def __sub__(self,o):return self+-o
    def __mul__(self,o):
        values=[x*y for x in (self.lower,self.upper) for y in (o.lower,o.upper)]
        if any(math.isnan(x) for x in values):return Interval(-math.inf,math.inf)
        return Interval(down(min(values)),up(max(values)))
    def __truediv__(self,o):
        if o.lower<=0<=o.upper:raise ValueError("division interval includes zero")
        return self*Interval(down(1/o.upper),up(1/o.lower))
    def contains(self,x):return self.lower<=x<=self.upper

def evaluate(expr,variables,depth=0):
    if depth>32:raise ValueError("expression depth limit")
    op=expr[0]
    if op=='var':return variables[expr[1]]
    if op=='lit':return Interval(float(expr[1]),float(expr[1]))
    if op not in {'add','sub','mul','div'}:raise ValueError("unsupported interval operator")
    a,b=evaluate(expr[1],variables,depth+1),evaluate(expr[2],variables,depth+1)
    return {'add':Interval.__add__,'sub':Interval.__sub__,'mul':Interval.__mul__,'div':Interval.__truediv__}[op](a,b)
