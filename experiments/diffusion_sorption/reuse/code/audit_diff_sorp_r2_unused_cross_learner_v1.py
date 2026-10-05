from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

PROTOCOL_ID = "DIFF_SORP_R2_UNUSED_CROSS_LEARNER_PORTABILITY_V1"
PROTOCOL_SHA256 = "52275a1302092801b8bd16d5818cf24ce5212f3bee4df3428e30bee1d2e66dba"
DATA_SHA256 = "8e48ab3efd39ab63524d92e85a4e0db46347b72e7c5b05edf81e1c9637a3ab2d"
DATA_MD5 = "9d466d1213065619d087319e16d9a938"
DATA_BYTES = 4217044280
BASIS_SHA256 = "3958eceb871c779cda4c3f5545b47cdff99f4e7ee32f2a7a5b4c830d58275f50"
MEAN_SHA256 = "b6c164c35b1506596a1a94bc7f7c788475498fa356eba8539cc6d77c0545fce3"
INDICES_SHA256 = "dc940ab47c70520db150fcd27e3297f03f689c3d4baf251fea07d4d8ec725894"
INDICES = [0,1,2,3,4,5,6,8]
SEEDS = (10,11,12,13,14)
THRESHOLD = 1.05
FNO_PARAMS = 74209
SPLIT_DIGESTS = {
    "train": "bdd5fe2abc0dc9706ddec253f702b9ebdd6751f69bbd5d9f5005d663cc60fcd2",
    "validation": "823b6f6a0ce08224ef14153babd0a52c54c1d2212492629fd6551d7c923e3ee6",
    "unused": "060451f6180354ffe574c8095da4bf0349a5032ea5cd8003102d77c9ac0242da",
    "selection": "d4ab1c03f1d2eb2a96baaeb1fea61ca42e73272a9f4ca5816df5198901290f44",
}
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 13859889106505471460


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b=f.read(8<<20)
            if not b: break
            h.update(b)
    return h.hexdigest()


def write_create_only(path: Path, obj: Mapping) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(obj,f,indent=2,sort_keys=True); f.write('\n')


def preholdout(package_dir: Path, out: Path) -> dict:
    failures=[]
    protocol=package_dir/'DIFF_SORP_R2_UNUSED_CROSS_LEARNER_PORTABILITY_V1.md'
    if sha256_file(protocol)!=PROTOCOL_SHA256: failures.append('protocol sha mismatch')
    package_seal_path=package_dir/'PACKAGE_SEAL.json'
    if not package_seal_path.is_file(): failures.append('package seal missing')
    else:
        ps=json.loads(package_seal_path.read_text())
        if ps.get('protocol_id')!=PROTOCOL_ID or ps.get('protocol_sha256')!=PROTOCOL_SHA256: failures.append('package identity mismatch')
        for rel,rec in ps.get('bound_files',{}).items():
            p=package_dir/rel
            if not p.is_file() or sha256_file(p)!=rec.get('sha256'): failures.append(f'package bound file mismatch: {rel}')

    seal_path=out/'PREHOLDOUT_SEAL.json'
    ledger_path=out/'ACCESS_LEDGER_PREHOLDOUT.json'
    if not seal_path.is_file(): failures.append('preholdout seal missing')
    if not ledger_path.is_file(): failures.append('preholdout ledger missing')
    if (out/'UNUSED_ACCESS_ONCE.json').exists(): failures.append('UNUSED marker exists before audit')
    if failures:
        result={'status':'PREHOLDOUT_VERIFY_FAIL','failures':failures,'verified_at_utc':now()}
        write_create_only(out/'PREHOLDOUT_VERIFY.json',result); return result

    seal=json.loads(seal_path.read_text())
    ledger=json.loads(ledger_path.read_text())
    if seal.get('status')!='PREHOLDOUT_READY': failures.append(f"seal status {seal.get('status')}")
    if seal.get('protocol_id')!=PROTOCOL_ID or seal.get('protocol_sha256')!=PROTOCOL_SHA256: failures.append('preholdout protocol mismatch')
    data=seal.get('data',{})
    if (data.get('bytes'),data.get('sha256'),data.get('md5'))!=(DATA_BYTES,DATA_SHA256,DATA_MD5): failures.append('data identity mismatch')
    inh=seal.get('inherited',{})
    if inh.get('basis_sha256')!=BASIS_SHA256 or inh.get('mean_sha256')!=MEAN_SHA256 or inh.get('indices_sha256')!=INDICES_SHA256: failures.append('inherited target hash mismatch')
    if inh.get('indices')!=INDICES or inh.get('K')!=8: failures.append('inherited target indices/K mismatch')
    roles=seal.get('roles',{})
    for role,digest in SPLIT_DIGESTS.items():
        if roles.get(role,{}).get('sha256')!=digest: failures.append(f'{role} split digest mismatch')
    if not (ledger.get('train')==480 and ledger.get('validation')==160 and ledger.get('selection')==0 and ledger.get('original_test')==0 and ledger.get('unused')==0): failures.append('preholdout role ledger mismatch')
    if seal.get('access_ledger_sha256')!=sha256_file(ledger_path): failures.append('ledger hash binding mismatch')
    if seal.get('new_seeds')!=list(SEEDS): failures.append('seed set mismatch')
    persist=float(seal.get('persistence_validation_mse',float('nan')))
    if not np.isfinite(persist) or persist<=0: failures.append('invalid persistence validation mse')

    for family in ('mlp','fno'):
        fam=seal.get(family,{})
        for arm in ('dense','compact'):
            for seed in SEEDS:
                cell=fam.get(arm,{}).get(str(seed))
                if not isinstance(cell,dict): failures.append(f'missing {family} {arm} seed {seed}'); continue
                if cell.get('stop_reason')!='PATIENCE': failures.append(f'{family} {arm} seed {seed}: stop not patience')
                if cell.get('nonfinite_count')!=0: failures.append(f'{family} {arm} seed {seed}: nonfinite')
                rel=cell.get('checkpoint_relative_path')
                if not rel: failures.append(f'{family} {arm} seed {seed}: checkpoint path missing'); continue
                p=out/rel
                if not p.is_file() or sha256_file(p)!=cell.get('checkpoint_file_sha256'): failures.append(f'{family} {arm} seed {seed}: checkpoint hash mismatch')
                if arm=='dense':
                    val=float(cell.get('selected_validation_mse',float('nan')))
                    if not (np.isfinite(val) and val<persist): failures.append(f'{family} dense seed {seed}: not better than persistence')
        if family=='fno':
            for seed in SEEDS:
                d=fam.get('dense',{}).get(str(seed),{})
                c=fam.get('compact',{}).get(str(seed),{})
                if d.get('parameter_count')!=FNO_PARAMS or c.get('parameter_count')!=FNO_PARAMS: failures.append(f'fno seed {seed}: param mismatch')
                if d.get('state_shapes')!=c.get('state_shapes'): failures.append(f'fno seed {seed}: state-shape mismatch')
                if d.get('initial_state_sha256')!=c.get('initial_state_sha256'): failures.append(f'fno seed {seed}: init mismatch')

    result={
        'status':'PREHOLDOUT_VERIFY_PASS' if not failures else 'PREHOLDOUT_VERIFY_FAIL',
        'protocol_id':PROTOCOL_ID,
        'protocol_sha256':PROTOCOL_SHA256,
        'preholdout_seal_sha256':sha256_file(seal_path),
        'package_seal_sha256':sha256_file(package_seal_path),
        'failures':failures,
        'unused_marker_absent':not (out/'UNUSED_ACCESS_ONCE.json').exists(),
        'verified_at_utc':now(),
    }
    write_create_only(out/'PREHOLDOUT_VERIFY.json',result)
    return result


def aggregate(per: Mapping[str,np.ndarray]) -> dict:
    ratios={'mlp':{},'fno':{}}
    for family in ('mlp','fno'):
        for seed in SEEDS:
            d=np.asarray(per[f'{family}_dense_seed{seed}'],dtype=np.float64)
            c=np.asarray(per[f'{family}_compact_seed{seed}'],dtype=np.float64)
            if d.shape!=(160,) or c.shape!=(160,) or not np.all(np.isfinite(d)) or not np.all(np.isfinite(c)) or np.any(d<=0) or np.any(c<0): raise ValueError(f'invalid {family} seed {seed} arrays')
            ratios[family][str(seed)]=float(c.mean()/d.mean())
    q_mlp=float(np.median(list(ratios['mlp'].values())))
    q_fno=float(np.median(list(ratios['fno'].values())))
    q_port=max(q_mlp,q_fno)
    allr=[*ratios['mlp'].values(),*ratios['fno'].values()]
    rng=np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    bm=np.empty(BOOTSTRAP_REPLICATES); bf=np.empty(BOOTSTRAP_REPLICATES); bp=np.empty(BOOTSTRAP_REPLICATES)
    for i in range(BOOTSTRAP_REPLICATES):
        draw=rng.integers(0,160,size=160)
        qs={}
        for family in ('mlp','fno'):
            rr=[]
            for seed in SEEDS:
                d=np.asarray(per[f'{family}_dense_seed{seed}'],dtype=np.float64)
                c=np.asarray(per[f'{family}_compact_seed{seed}'],dtype=np.float64)
                rr.append(float(c[draw].mean()/d[draw].mean()))
            qs[family]=float(np.median(rr))
        bm[i]=qs['mlp']; bf[i]=qs['fno']; bp[i]=max(qs['mlp'],qs['fno'])
    return {
        'Q_MLP':q_mlp,'Q_FNO':q_fno,'Q_port':float(q_port),'threshold':THRESHOLD,
        'verdict':'PORTABILITY_PASS' if q_port<=THRESHOLD else 'PORTABILITY_FAIL',
        'seed_ratios':ratios,
        'seed_uniform_pass':bool(all(x<=THRESHOLD for x in allr)),
        'max_seed_ratio':float(max(allr)),
        'bootstrap':{'replicates':BOOTSTRAP_REPLICATES,'seed':BOOTSTRAP_SEED,'interval95':{
            'Q_MLP':np.quantile(bm,[.025,.975]).tolist(),
            'Q_FNO':np.quantile(bf,[.025,.975]).tolist(),
            'Q_port':np.quantile(bp,[.025,.975]).tolist(),
        }}
    }


def final(out: Path) -> dict:
    failures=[]
    result_path=out/'FINAL_RESULT.json'; per_path=out/'UNUSED_PER_TRAJECTORY_ERRORS.npz'; ledger_path=out/'ACCESS_LEDGER_FINAL.json'; marker=out/'UNUSED_ACCESS_ONCE.json'
    for p in (result_path,per_path,ledger_path,marker,out/'PREHOLDOUT_VERIFY.json',out/'PREHOLDOUT_SEAL.json'):
        if not p.is_file(): failures.append(f'missing {p.name}')
    if failures:
        r={'status':'FINAL_VERIFY_FAIL','failures':failures,'verified_at_utc':now()}; write_create_only(out/'FINAL_VERIFY.json',r); return r
    with np.load(per_path,allow_pickle=False) as z: per={k:np.asarray(z[k],dtype=np.float64) for k in z.files}
    expected_names={f'{family}_{arm}_seed{seed}' for family in ('mlp','fno') for arm in ('dense','compact') for seed in SEEDS}
    if set(per)!=expected_names: failures.append('per-trajectory key set mismatch')
    agg=aggregate(per)
    result=json.loads(result_path.read_text())
    for key in ('Q_MLP','Q_FNO','Q_port','threshold','verdict','seed_ratios','seed_uniform_pass','max_seed_ratio','bootstrap'):
        if result.get(key)!=agg.get(key): failures.append(f'aggregate mismatch: {key}')
    if result.get('per_trajectory_sha256')!=sha256_file(per_path): failures.append('per-trajectory hash mismatch')
    ledger=json.loads(ledger_path.read_text())
    if not (ledger.get('train')==480 and ledger.get('validation')==160 and ledger.get('selection')==0 and ledger.get('original_test')==0 and ledger.get('unused')==160 and ledger.get('unused_rows')==3200): failures.append('final role ledger mismatch')
    if ledger.get('unused_split_sha256')!=SPLIT_DIGESTS['unused']: failures.append('final UNUSED split digest mismatch')
    if result.get('unused_split_sha256')!=SPLIT_DIGESTS['unused']: failures.append('result UNUSED split digest mismatch')
    r={
        'status':'FINAL_VERIFY_PASS' if not failures else 'FINAL_VERIFY_FAIL',
        'protocol_id':PROTOCOL_ID,
        'recomputed':agg,
        'final_result_sha256':sha256_file(result_path),
        'per_trajectory_sha256':sha256_file(per_path),
        'final_ledger_sha256':sha256_file(ledger_path),
        'unused_marker_sha256':sha256_file(marker),
        'failures':failures,
        'verified_at_utc':now(),
    }
    write_create_only(out/'FINAL_VERIFY.json',r)
    return r


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('preholdout'); p.add_argument('--package-dir',required=True); p.add_argument('--out',required=True)
    p=sub.add_parser('final'); p.add_argument('--out',required=True)
    a=ap.parse_args()
    r=preholdout(Path(a.package_dir),Path(a.out)) if a.cmd=='preholdout' else final(Path(a.out))
    print(json.dumps(r,indent=2,sort_keys=True))
    if r['status'].endswith('FAIL'): raise SystemExit(2)

if __name__=='__main__': main()
