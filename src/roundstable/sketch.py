import numpy as np
class Sketch:
    def __init__(self,dimension,rows=4):
        if not isinstance(dimension,int) or not 1<=rows<=dimension<=512:raise ValueError("invalid sketch shape")
        self.d=dimension;self.ell=rows;self.buffer=np.empty((0,dimension),dtype=np.float64);self.count=0
    def append(self,values):
        array=np.asarray(values,dtype=np.float64)
        if array.ndim!=2 or array.shape[1]!=self.d or not np.isfinite(array).all():raise ValueError("finite row matrix required")
        for row in array:
            self.buffer=np.vstack([self.buffer,row]);self.count+=1
            if len(self.buffer)>=2*self.ell:self._compress()
        return self
    def _compress(self):
        if len(self.buffer)<=self.ell:return
        _,s,v=np.linalg.svd(self.buffer,full_matrices=False)
        delta=s[self.ell-1]**2
        shrunk=np.sqrt(np.maximum(s[:self.ell]**2-delta,0))
        self.buffer=shrunk[:,None]*v[:self.ell]
    def matrix(self):self._compress();return self.buffer.copy()
    def merge(self,other):
        if self.d!=other.d or self.ell!=other.ell:raise ValueError("incompatible sketches")
        count=self.count+other.count
        self.append(other.matrix());self.count=count;return self
    def dump(self):return {"version":1,"dimension":self.d,"rows":self.ell,"count":self.count,"matrix":self.matrix().tolist()}
    @classmethod
    def load(cls,state):
        if state.get('version')!=1:raise ValueError("unsupported sketch version")
        obj=cls(state['dimension'],state['rows']);a=np.asarray(state['matrix'],dtype=np.float64)
        if a.ndim!=2 or a.shape[1]!=obj.d or len(a)>obj.ell or not np.isfinite(a).all() or not isinstance(state['count'],int) or state['count']<len(a):raise ValueError("invalid sketch state")
        obj.buffer=a;obj.count=state['count'];return obj
