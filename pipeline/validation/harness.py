"""Offline C5 experiment skeleton; no network, imagery, or calibrated probabilities."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

SPLITS=('tuning','calibration','evaluation')
METRICS=('entrance_location_ft','entrance_rank','entrance_abstention','storey_error',
         'material_correct','roof_median_abs_ft','roof_p90_abs_ft','topology_pass')

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def digest(value): return hashlib.sha256(canonical(value)).hexdigest()

def distance(a,b):
    lat1,lat2=map(math.radians,(a['lat'],b['lat']))
    dl=math.radians(b['lon']-a['lon']); dt=lat2-lat1
    v=math.sin(dt/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dl/2)**2
    return 6371008.8*2*math.asin(math.sqrt(min(1,max(0,v))))

def validate(records):
    seen=set()
    for r in records:
        if not isinstance(r.get('id'),str) or not r['id'] or r['id'] in seen:
            raise ValueError('Candidate IDs must be unique nonempty strings')
        seen.add(r['id'])
        for field in ('site_id','group_id','group_basis','source'):
            if not isinstance(r.get(field),str) or not r[field]: raise ValueError('Missing '+field)
        for key,limit in (('lat',90),('lon',180)):
            v=r.get(key)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or abs(v)>limit:
                raise ValueError('Invalid '+key)
        strata=r.get('strata',{})
        for key in ('region','settlement','use'):
            label=strata.get(key)
            if label is not None and not isinstance(label,str): raise ValueError('Invalid stratum')
            if label and not r.get('strata_sources',{}).get(key): raise ValueError('Stratum requires independent source: '+key)

def manifest(records,development,seed='c5-v1',target_groups=60,separation_m=1000):
    """Freeze grouped/stratified selection. O(n²): bounded offline feasibility sample."""
    if target_groups<1 or not math.isfinite(separation_m) or separation_m<=0: raise ValueError('Invalid sampling limits')
    if len(records)>10000: raise ValueError('Pre-filter the NAD candidate frame to <=10000 records')
    validate(records);validate(development)
    rows=sorted(records,key=lambda r:r['id']); development=sorted(development,key=lambda r:r['id']); allrows=rows+development
    parent=list(range(len(allrows)))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]];i=parent[i]
        return i
    def merge(i,j): parent[find(j)]=find(i)
    for i,a in enumerate(allrows):
        for j in range(i):
            b=allrows[j]
            if a['id']==b['id'] or a['site_id']==b['site_id'] or a['group_id']==b['group_id'] or distance(a,b)<=separation_m: merge(i,j)
    dev_roots={find(i) for i in range(len(rows),len(allrows))}
    dev_roots.update(find(i) for i,r in enumerate(allrows) if r.get('development_only'))
    groups=defaultdict(list);excluded=[]
    for i,r in enumerate(rows):
        if find(i) in dev_roots or r.get('development_only'):
            excluded.append(r['id'])
        else: groups[find(i)].append(r)
    # Unknown labels remain explicit. Mixed groups preserve all constituent strata.
    buckets=defaultdict(list)
    for group in groups.values():
        stratum=tuple(sorted({tuple((r.get('strata') or {}).get(k) or 'missing' for k in ('region','settlement','use')) for r in group}))
        gid=digest(sorted(r['id'] for r in group))
        buckets[stratum].append((digest([seed,gid]),gid,group))
    for b in buckets.values(): b.sort()
    selected=[];bucket_keys=sorted(buckets)
    while len(selected)<target_groups and any(buckets.values()):
        for key in bucket_keys:
            if buckets[key] and len(selected)<target_groups: selected.append((key,buckets[key].pop(0)))
    assigned=[];stratum_counts=Counter();global_counts=Counter()
    for stratum,(_,gid,group) in selected:
        # Minimise group counts per stratum, then overall; deterministic tie order.
        split=min(SPLITS,key=lambda s:(stratum_counts[stratum,s],global_counts[s],SPLITS.index(s)))
        stratum_counts[stratum,split]+=1;global_counts[split]+=1
        assigned.append({'group_id':gid,'split':split,'strata':[list(t) for t in stratum],'records':group})
    result={'schema_version':1,'purpose':'initial offline feasibility experiment, not universal calibration',
       'seed':seed,'candidate_sha256':digest(rows),'development_sha256':digest(development),
       'minimum_separation_m':separation_m,'requested_groups':target_groups,'selected_groups':len(assigned),
       'excluded_development_ids':excluded,'groups':assigned,
       'group_counts':dict(global_counts),'address_counts':dict(Counter(g['split'] for g in assigned for _ in g['records'])),
       'limits':['Selection balances available group strata, not every crossed cell or address count.',
                 'Provided group/site identifiers and independent strata joins require source audit.',
                 'Development manifest must include all six project cases; no acquisition performed.',
                 'Group and distance checks do not establish representative geographic coverage.']}
    result['manifest_sha256']=digest(result)
    return result

def verify(frozen):
    body={k:v for k,v in frozen.items() if k!='manifest_sha256'}
    if digest(body)!=frozen.get('manifest_sha256'): raise ValueError('Frozen manifest modified')

def summarize(frozen,results,split):
    verify(frozen)
    if split not in SPLITS: raise ValueError('Unknown split')
    selected={r['id']:g for g in frozen['groups'] if g['split']==split for r in g['records']}
    if len({r['id'] for r in results})!=len(results): raise ValueError('Duplicate results')
    if any(r['id'] not in selected for r in results): raise ValueError('Results outside requested frozen split')
    by_id={r['id']:r for r in results};summary={}
    for metric in METRICS:
        vals=[];missing=0;strata=defaultdict(lambda:{'available':0,'missing':0})
        for rid,g in selected.items():
            value=by_id.get(rid,{}).get('metrics',{}).get(metric)
            ok=isinstance(value,(int,float)) and math.isfinite(value)
            key=json.dumps(g['strata'],sort_keys=True)
            strata[key]['available' if ok else 'missing']+=1
            if ok: vals.append(value)
            else: missing+=1
        summary[metric]={'available':len(vals),'missing':missing,'mean':sum(vals)/len(vals) if vals else None,
                         'by_stratum':dict(strata),'status':'complete' if vals and not missing else 'incomplete'}
    return {'manifest_sha256':frozen['manifest_sha256'],'split':split,'expected_addresses':len(selected),
            'result_addresses':len(results),'metrics':summary,
            'note':'Metric availability skeleton only. No calibration/accuracy claim, uncertainty intervals or automatic scoring of pixels/geometry.'}

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    sample=sub.add_parser('freeze');sample.add_argument('candidates',type=Path);sample.add_argument('development',type=Path)
    sample.add_argument('output',type=Path);sample.add_argument('--seed',default='c5-v1');sample.add_argument('--groups',type=int,default=60)
    summary=sub.add_parser('summarize');summary.add_argument('manifest',type=Path);summary.add_argument('results',type=Path)
    summary.add_argument('split',choices=SPLITS);summary.add_argument('output',type=Path)
    args=ap.parse_args()
    if args.command=='freeze':
        value=manifest(json.loads(args.candidates.read_text()),json.loads(args.development.read_text()),args.seed,args.groups)
    else: value=summarize(json.loads(args.manifest.read_text()),json.loads(args.results.read_text()),args.split)
    # Exclusive creation prevents accidentally replacing a frozen evaluation manifest.
    with args.output.open('x',encoding='utf-8') as f: json.dump(value,f,indent=2,allow_nan=False)

if __name__=='__main__': main()
