from fractions import Fraction
import math,json
class Accumulator:
    def __init__(self):self.total=Fraction(0);self.count=0;self.nan=False;self.positive=False;self.negative=False
    def add(self,value):
        value=float(value);self.count+=1
        if math.isnan(value):self.nan=True
        elif math.isinf(value):
            if value>0:self.positive=True
            else:self.negative=True
        else:self.total+=Fraction.from_float(value)
        return self
    def merge(self,other):
        self.total+=other.total;self.count+=other.count;self.nan|=other.nan;self.positive|=other.positive;self.negative|=other.negative;return self
    def finalize(self):
        if self.nan or (self.positive and self.negative):return float('nan')
        if self.positive:return float('inf')
        if self.negative:return float('-inf')
        try:return float(self.total)
        except OverflowError:return float('inf') if self.total.numerator>0 else float('-inf')
    def dump(self):return json.dumps({"version":1,"n":str(self.total.numerator),"d":str(self.total.denominator),"count":self.count,"nan":self.nan,"positive":self.positive,"negative":self.negative},sort_keys=True)
    @classmethod
    def load(cls,text):
        d=json.loads(text)
        if d.get('version')!=1 or not isinstance(d['count'],int) or d['count']<0 or int(d['d'])<=0 or any(type(d[k]) is not bool for k in ['nan','positive','negative']):raise ValueError("invalid accumulator state")
        a=cls();a.total=Fraction(int(d['n']),int(d['d']));a.count=d['count'];a.nan=d['nan'];a.positive=d['positive'];a.negative=d['negative'];return a
