from fractions import Fraction as F
import json,math,random
import pytest
from roundstable import api
from roundstable.groupstats import GroupStats
from roundstable.__main__ import demo


def reference(records,variance):
    """Independent exact grouped statistics straight from the definitions."""
    groups={}
    for key,value,weight in records:
        w=F(1) if weight is None else (F(weight) if isinstance(weight,int) else F.from_float(weight))
        v=F(value) if isinstance(value,int) else F.from_float(value)
        count,sw,swx,swxx,sww=groups.get(key,(0,F(0),F(0),F(0),F(0)))
        groups[key]=(count+1,sw+w,swx+w*v,swxx+w*v*v,sww+w*w)
    out={}
    for key,(count,sw,swx,swxx,sww) in groups.items():
        mean=swx/sw if sw else None
        moment=swxx-swx*swx/sw if sw else None
        if variance=='population':var=moment/sw if sw else None
        else:
            den=sw-sww/sw if sw else F(0)
            var=moment/den if den>0 else None
        out[key]={'count':count,'weight':sw,'effective':sw*sw/sww if sww else F(0),'mean':mean,'variance':var}
    return out


def as_float(x):
    try:return float(x)
    except OverflowError:return math.inf if x>0 else -math.inf


def assert_result_matches(result,want,on_invalid,variance,skipped):
    assert result['convention']=={'on_invalid':on_invalid,'variance':variance}
    assert result['skipped']==skipped
    assert set(result['groups'])==set(want)
    for key,entry in result['groups'].items():
        r=want[key]
        assert entry['count']==r['count']
        assert entry['weight']==as_float(r['weight'])
        assert entry['effective']==as_float(r['effective'])
        for stat in ('mean','variance'):
            if r[stat] is None:
                assert entry[stat] is None
                assert stat in entry['undefined']
            else:
                assert entry[stat]==as_float(r[stat])


def nasty_records(rng,keys,n):
    base=[1e16+2*i for i in range(-4,5)]+[1e16,1.0,-1e16,3.25,-0.125,1e-300,-1e-300,5e150,-5e150,2.5,-7.75]
    weights=[None,1,2,0.5,0.0,1e-12,1e12,3.75]
    return [(rng.choice(keys),rng.choice(base) if rng.random()<0.7 else rng.uniform(-1e6,1e6),
             rng.choice(weights)) for _ in range(n)]


def split_into(items,cuts):
    out,start=[],0
    for cut in list(cuts)+[len(items)]:
        out.append(items[start:cut]);start=cut
    return [part for part in out if part]


def build(records,rng,on_invalid='reject',variance='sample'):
    stats=GroupStats(on_invalid,variance)
    for key,value,weight in records:stats.add(key,value,weight)
    return GroupStats.load(json.loads(json.dumps(stats.dump()))) if rng.random()<0.6 else stats


def merge_tree(shards,rng):
    if len(shards)==1:return shards[0]
    cut=rng.randrange(1,len(shards))
    left,right=merge_tree(shards[:cut],rng),merge_tree(shards[cut:],rng)
    if rng.random()<0.5:left=GroupStats.load(left.dump())
    if rng.random()<0.5:right=GroupStats.load(right.dump())
    merged=left.merge(right)
    return GroupStats.load(merged.dump()) if rng.random()<0.5 else merged


@pytest.mark.parametrize('variance',['population','sample'])
def test_groupstats_partition_merge_recovery_invariance(variance):
    rng=random.Random(20261120)
    records=nasty_records(rng,['alpha','beta','gamma','delta'],120)
    want=reference(records,variance)
    results=[]
    for _ in range(8):
        shuffled=records[:];rng.shuffle(shuffled)
        shards=split_into(shuffled,sorted(rng.sample(range(1,len(shuffled)),rng.randrange(1,6))))
        merged=merge_tree([build(shard,rng,variance=variance) for shard in shards],rng)
        result=merged.finalize()
        assert_result_matches(result,want,'reject',variance,0)
        results.append(result)
    for other in results[1:]:assert other==results[0]


def test_groupstats_huge_mean_tiny_fluctuation():
    values=[1e16-4,1e16-2,1e16,1e16+2,1e16+4]
    stats=GroupStats(variance='sample')
    for v in values:stats.add('g',v)
    entry=stats.finalize()['groups']['g']
    exact=[F.from_float(v) for v in values]
    mean=sum(exact)/len(exact)
    assert entry['mean']==1e16==float(mean)
    assert entry['variance']==10.0==float(sum((x-mean)**2 for x in exact)/4)


def test_groupstats_strong_cancellation_keeps_small_terms():
    stats=GroupStats()
    for v in [1e16,1.0,-1e16]:stats.add('c',v)
    entry=stats.finalize()['groups']['c']
    exact=[F.from_float(1e16),F(1),F.from_float(-1e16)]
    mean=sum(exact)/3
    assert entry['mean']==float(F(1,3))!=0.0
    assert entry['variance']==float(sum((x-mean)**2 for x in exact)/2)


def test_groupstats_zero_weight_and_single_effective_observation():
    stats=GroupStats(variance='sample')
    stats.add('solo',5.0).add('solo',999.0,0.0).add('solo',-999.0,0.0)
    entry=stats.finalize()['groups']['solo']
    assert (entry['count'],entry['weight'],entry['effective'],entry['mean'])==(3,1.0,1.0,5.0)
    assert entry['variance'] is None and 'variance' in entry['undefined']
    pop=GroupStats(variance='population')
    pop.add('solo',5.0)
    assert pop.finalize()['groups']['solo']['variance']==0.0
    zero=GroupStats()
    zero.add('z',1.0,0.0)
    entry=zero.finalize()['groups']['z']
    assert (entry['count'],entry['weight'],entry['effective'])==(1,0.0,0.0)
    assert entry['mean'] is None and entry['variance'] is None
    assert set(entry['undefined'])=={'mean','variance'}


def test_groupstats_missing_and_nonfinite_policy():
    bad=[None,float('nan'),float('inf'),float('-inf'),'x']
    for value in bad:
        with pytest.raises(ValueError):GroupStats(on_invalid='reject').add('k',value)
    stats=GroupStats(on_invalid='skip')
    for value in bad:stats.add('k',value)
    stats.add('k',2.0)
    result=stats.finalize()
    assert result['skipped']==len(bad)
    assert result['groups']['k']['count']==1
    for weight in (-1.0,-0.5,float('nan'),float('inf'),'w'):
        with pytest.raises(ValueError):GroupStats(on_invalid='skip').add('k',1.0,weight)
    assert GroupStats(on_invalid='skip').add('k',1.0,0.0).groups['k'][0]==1


def test_groupstats_capacity_and_state_version_rejection():
    stats=GroupStats(capacity=2)
    stats.add('a',1.0).add('b',2.0)
    with pytest.raises(ValueError):stats.add('c',3.0)
    other=GroupStats(capacity=3)
    other.add('c',3.0)
    with pytest.raises(ValueError):stats.merge(other)
    with pytest.raises(ValueError):stats.merge(GroupStats(variance='population'))
    with pytest.raises(ValueError):GroupStats(on_invalid='skip').merge(GroupStats(on_invalid='reject'))
    with pytest.raises(ValueError):stats.merge(object())
    good=stats.dump()
    zero={'count':1,'sw':'0/1','swx':'0/1','swxx':'0/1','sww':'0/1'}
    for broken in ({**good,'version':2},{**good,'capacity':1},{**good,'skipped':-1},
                   {**good,'groups':{'a':{**zero,'count':-1}}},
                   {**good,'groups':{'a':{**zero,'sw':'1/0'}}},
                   {**good,'groups':{'a':{**zero,'sw':'1.5'}}},
                   {**good,'groups':{'a':{**zero,'swxx':'-1/1'}}},
                   {**good,'groups':{'a':{k:v for k,v in zero.items() if k!='sww'}}},
                   'not a state',{'version':1}):
        with pytest.raises(ValueError):GroupStats.load(broken)
    assert GroupStats.load(good).finalize()==stats.finalize()


def test_groupstats_extreme_weights_stay_exact():
    records=[('w',1.0,1e300),('w',3.0,1e-300),('w',2.0,1.0)]
    stats=GroupStats(variance='population')
    for key,value,weight in records:stats.add(key,value,weight)
    entry=stats.finalize()['groups']['w']
    want=reference(records,'population')['w']
    assert want['weight']>F.from_float(1e300)
    assert entry['weight']==float(want['weight'])
    assert entry['effective']==float(want['effective'])
    assert entry['mean']==float(want['mean'])
    assert entry['variance']==float(want['variance'])


def test_groupstats_overflow_saturates_but_state_stays_exact():
    stats=GroupStats(variance='population')
    stats.add('big',1e308,1e308).add('big',-1e308,1e308)
    entry=stats.finalize()['groups']['big']
    assert entry['weight']==math.inf
    assert entry['mean']==0.0
    assert entry['variance']==math.inf
    restored=GroupStats.load(stats.dump())
    assert restored.finalize()==stats.finalize()
    want=reference([('big',1e308,1e308),('big',-1e308,1e308)],'population')['big']
    assert restored.groups['big'][1]==want['weight']


def test_groupstats_api_request_restore_and_demo():
    records=[{'key':'a','value':1.0},{'key':'a','value':2.0,'weight':2},
             {'key':'b','value':None},{'key':'b','value':4.0,'weight':0}]
    first=api.OPERATIONS['groupstats']({'on_invalid':'skip','variance':'sample','records':records})
    assert first['skipped']==1
    assert first['groups']['a']['mean']==5/3
    assert first['groups']['b']['mean'] is None
    more=[{'key':'a','value':3.0},{'key':'c','value':1.0}]
    continued=api.OPERATIONS['groupstats']({'state':first['state'],'records':more})
    fresh=api.OPERATIONS['groupstats']({'on_invalid':'skip','variance':'sample','records':records+more})
    assert continued==fresh
    with pytest.raises(ValueError):api.OPERATIONS['groupstats']({'state':first['state'],'variance':'population'})
    with pytest.raises(ValueError):api.OPERATIONS['groupstats']({'records':{'key':'a'}})
    with pytest.raises(ValueError):api.OPERATIONS['groupstats']({'records':[['a',1.0]]})
    out=demo()
    assert 'groupstats' in out
    json.dumps(out,allow_nan=False)
