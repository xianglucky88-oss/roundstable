"""Exact mergeable grouped descriptive statistics.

Each record carries a nonempty string group key, an observation value and an
optional non-negative finite weight (default one). Per group the state keeps a
record count and the exact rational sums of ``w``, ``w*x``, ``w*x*x`` and
``w*w``, so any partitioning, merge tree or dump/load round trip yields
identical results; no observation is ever dropped to keep the arithmetic
stable. Request-level conventions are echoed back in every result:

* ``on_invalid``: ``'reject'`` raises on missing or non-finite observations,
  ``'skip'`` drops them and counts the drops in ``skipped``. Weight
  violations (negative, non-finite or non-numeric) always raise.
* ``variance``: ``'population'`` divides the exact centered second moment by
  ``sum(w)``; ``'sample'`` divides by ``sum(w) - sum(w*w)/sum(w)``, the
  reliability-weight analogue of ``n - 1``.

Zero-weight records raise neither the weight sum nor the effective sample
size ``sum(w)**2 / sum(w*w)``, and a single effective observation leaves the
sample variance undefined. A statistic that is not definable is reported as
``None`` with the reason kept under ``undefined``. Final float conversion may
saturate to ``+-inf``; the exact rational state is unaffected.
"""

from fractions import Fraction
import math

POLICIES = ('reject', 'skip')
CONVENTIONS = ('population', 'sample')
DEFAULT_CAPACITY = 4096
_SUM_FIELDS = ('sw', 'swx', 'swxx', 'sww')


def _exact_float(value):
    try:
        return float(value)
    except OverflowError:
        return math.inf if value > 0 else -math.inf


def _rational(text):
    if isinstance(text, str):
        parts = text.split('/')
        try:
            if len(parts) == 1:
                return Fraction(int(parts[0]))
            if len(parts) == 2 and int(parts[1]) > 0:
                return Fraction(int(parts[0]), int(parts[1]))
        except ValueError:
            pass
    raise ValueError("invalid groupstats state")


class GroupStats:
    VERSION = 1

    def __init__(self, on_invalid='reject', variance='sample', capacity=DEFAULT_CAPACITY):
        if on_invalid not in POLICIES:
            raise ValueError("unsupported invalid-value policy")
        if variance not in CONVENTIONS:
            raise ValueError("unsupported variance convention")
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError("invalid group capacity")
        self.on_invalid = on_invalid
        self.variance = variance
        self.capacity = capacity
        self.groups = {}
        self.skipped = 0

    @staticmethod
    def _weight(weight):
        if weight is None:
            return Fraction(1)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("weight must be a non-negative finite number")
        if weight < 0 or (isinstance(weight, float) and not math.isfinite(weight)):
            raise ValueError("weight must be non-negative and finite")
        return Fraction(weight) if isinstance(weight, int) else Fraction.from_float(weight)

    @staticmethod
    def _observation(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if isinstance(value, int):
            return Fraction(value)
        return Fraction.from_float(value) if math.isfinite(value) else None

    def add(self, key, value, weight=None):
        if not isinstance(key, str) or not key:
            raise ValueError("group key must be a nonempty string")
        w = self._weight(weight)
        x = self._observation(value)
        if x is None:
            if self.on_invalid == 'reject':
                raise ValueError("missing or non-finite observation")
            self.skipped += 1
            return self
        group = self.groups.get(key)
        if group is None:
            if len(self.groups) >= self.capacity:
                raise ValueError("group capacity reached")
            group = self.groups[key] = [0, Fraction(0), Fraction(0), Fraction(0), Fraction(0)]
        group[0] += 1
        group[1] += w
        group[2] += w * x
        group[3] += w * x * x
        group[4] += w * w
        return self

    def merge(self, other):
        if not isinstance(other, GroupStats):
            raise ValueError("incompatible groupstats state")
        if other.variance != self.variance or other.on_invalid != self.on_invalid:
            raise ValueError("incompatible groupstats conventions")
        if len(self.groups) + sum(1 for key in other.groups if key not in self.groups) > self.capacity:
            raise ValueError("group capacity reached")
        for key, sums in other.groups.items():
            mine = self.groups.get(key)
            if mine is None:
                self.groups[key] = list(sums)
            else:
                for i in range(5):
                    mine[i] += sums[i]
        self.skipped += other.skipped
        return self

    def finalize(self):
        groups = {}
        for key in sorted(self.groups):
            count, sw, swx, swxx, sww = self.groups[key]
            entry = {'count': count, 'weight': _exact_float(sw),
                     'effective': _exact_float(sw * sw / sww) if sww else 0.0}
            reasons = {}
            if sw > 0:
                entry['mean'] = _exact_float(swx / sw)
                # Exact arithmetic keeps the centered second moment non-negative.
                moment = swxx - swx * swx / sw
                denominator = sw if self.variance == 'population' else sw - sww / sw
                if denominator > 0:
                    entry['variance'] = _exact_float(moment / denominator)
                else:
                    entry['variance'] = None
                    reasons['variance'] = ('undefined: sample denominator is zero; '
                                           'more than one effective observation required')
            else:
                entry['mean'] = None
                entry['variance'] = None
                reasons['mean'] = 'undefined: total weight is zero'
                reasons['variance'] = 'undefined: total weight is zero'
            if reasons:
                entry['undefined'] = reasons
            groups[key] = entry
        return {'convention': {'on_invalid': self.on_invalid, 'variance': self.variance},
                'skipped': self.skipped, 'groups': groups}

    def dump(self):
        def rational(x):
            return f"{x.numerator}/{x.denominator}"
        groups = {}
        for key, sums in sorted(self.groups.items()):
            count, sw, swx, swxx, sww = sums
            groups[key] = {'count': count, 'sw': rational(sw), 'swx': rational(swx),
                           'swxx': rational(swxx), 'sww': rational(sww)}
        return {'version': self.VERSION, 'on_invalid': self.on_invalid, 'variance': self.variance,
                'capacity': self.capacity, 'skipped': self.skipped, 'groups': groups}

    @classmethod
    def load(cls, state):
        if not isinstance(state, dict) or state.get('version') != cls.VERSION:
            raise ValueError("unsupported groupstats state version")
        obj = cls(state.get('on_invalid'), state.get('variance'), state.get('capacity'))
        skipped = state.get('skipped')
        groups = state.get('groups')
        if (not isinstance(skipped, int) or isinstance(skipped, bool) or skipped < 0
                or not isinstance(groups, dict) or len(groups) > obj.capacity):
            raise ValueError("invalid groupstats state")
        for key, entry in groups.items():
            if not isinstance(key, str) or not key or not isinstance(entry, dict):
                raise ValueError("invalid groupstats state")
            count = entry.get('count')
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("invalid groupstats state")
            sums = [_rational(entry.get(field)) for field in _SUM_FIELDS]
            if sums[0] < 0 or sums[2] < 0 or sums[3] < 0:
                raise ValueError("invalid groupstats state")
            obj.groups[key] = [count] + sums
        obj.skipped = skipped
        return obj
