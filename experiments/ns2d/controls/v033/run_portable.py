from __future__ import annotations
import argparse, copy, hashlib, json, pathlib, platform, sys, time
from datetime import datetime, timezone
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
CODE = HERE.parents[1] / 'code'
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from run_quality_convergence_confirm_v1 import FAMILIES, MAX_EPOCHS, MIN_DELTA, PAIRS_PER_TRAJECTORY, PATIENCE, SPLIT_SEED, TRAJECTORIES_PER_FILE, index_digest, replay_stopping, sha256_file, split_family
from e2e_cost import MLP

PID = 'V033_NS2D_REDUCEDVAL_CONTROL_20260826'
K = 32
K_DIGEST = '7bdcc1e20de8efd6ac5ed54633e9f3a2a8bf99724aafd05956afee0663a5f763'
SEEDS = (0,1,2)
ARM_A = 'K32_REDUCEDVAL_COMPACT_HEAD'
ARM_B = 'K32_REDUCEDVAL_DENSE_HEAD'
ARMS = (ARM_A, ARM_B)
LR, BS, EBS = 1e-3, 64, 512
ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT/'data/ns2d'
A_DIR = ROOT/'results/quality_convergence_confirm_v1/stage_a_r1'
B_DIR = ROOT/'results/quality_convergence_confirm_v1/stage_b_r1'
C_DIR = ROOT/'results/quality_convergence_confirm_v1/stage_c_r1'
PROTOCOL = ROOT/'protocols/V033_NS2D_REDUCEDVAL_CONTROL_20260826.md'
OUT = ROOT/'results/v033_ns2d_reducedval_control_20260826_r1'

def now(): return datetime.now(timezone.utc).isoformat()
def readj(p): return json.loads(pathlib.Path(p).read_text())
def jhash(o): return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def newj(p,o):
    p=pathlib.Path(p)
    if p.exists(): raise FileExistsError(f'create-only refusal: {p}')
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n'); q.replace(p)
def state_hash(m):
    h=hashlib.sha256()
    for n,t in sorted(m.state_dict().items()):
        h.update(n.encode()); h.update(np.ascontiguousarray(t.detach().cpu().numpy(),dtype=np.float32).tobytes())
    return h.hexdigest()

def upstream(check_ckpt=True):
    ap,bp,cp=A_DIR/'STAGE_A.json',B_DIR/'STAGE_B.json',C_DIR/'STAGE_C.json'
    a,b,c=readj(ap),readj(bp),readj(cp)
    if (a.get('status'),b.get('status'),c.get('status')) != ('STAGE_A_COMPLETE','STAGE_B_COMPLETE','STAGE_C_COMPLETE'):
        raise RuntimeError('historical A/B/C incomplete')
    idx=[int(x) for x in c['index_sets']['K_prop']]
    if int(c['K_prop'])!=K or index_digest(idx)!=K_DIGEST or c['index_digests']['K_prop']!=K_DIGEST:
        raise RuntimeError('K32 identity mismatch')
    basis=A_DIR/'parent_basis_float32.npy'; mean=A_DIR/'train_mean_float32.npy'
    if sha256_file(basis)!=a['arrays']['parent_basis_float32.npy'] or sha256_file(mean)!=a['arrays']['train_mean_float32.npy']:
        raise RuntimeError('basis/mean digest mismatch')
    dense={}
    for s in SEEDS:
        e=dict(b['reference_checkpoints'][str(s)]); p=B_DIR/e['path']
        if check_ckpt and sha256_file(p)!=e['file_sha256']: raise RuntimeError(f'dense seed{s} digest mismatch')
        dense[str(s)]=e
    files=[]
    for fam in FAMILIES:
        m=[n for n in a['train_file_digests'] if f'_train_{fam}_' in n]
        if len(m)!=1: raise RuntimeError(f'family {fam} train shard mismatch')
        p=TRAIN_DIR/m[0]
        if not p.is_file(): raise RuntimeError(f'missing {p}')
        files.append((fam,p,a['train_file_digests'][m[0]]))
    return dict(a=a,b=b,c=c,ap=ap,bp=bp,cp=cp,idx=idx,basis=basis,mean=mean,dense=dense,files=files)

def load_tv(files,role):
    if role not in ('train','validation'): raise PermissionError(f'train runner forbids {role}')
    import h5py
    xs=[]; ys=[]; idsrec={}; reads=0
    for fi,(fam,p,_) in enumerate(files):
        tr,va,un=split_family(SPLIT_SEED,fi); wanted=tr if role=='train' else va
        if set(map(int,wanted)) & set(map(int,un)): raise PermissionError('UNUSED overlap')
        ids=sorted(map(int,wanted))
        with h5py.File(p,'r') as h:
            dset=h[list(h.keys())[0]]['u']
            if int(dset.shape[0])!=TRAJECTORIES_PER_FILE: raise RuntimeError('trajectory count mismatch')
            z=dset[ids]
        reads += len(ids); idsrec[str(fam)]=ids
        z=z[np.argsort(np.argsort(np.asarray(wanted,dtype=np.int64)))]
        if int(z.shape[1]-1)!=PAIRS_PER_TRAJECTORY: raise RuntimeError('pair count mismatch')
        d=int(z.shape[-2]*z.shape[-1]); xs.append(z[:,:-1].reshape(-1,d).astype(np.float32)); ys.append(z[:,1:].reshape(-1,d).astype(np.float32))
    return np.concatenate(xs),np.concatenate(ys),reads,idsrec

def coeff_val_mse(model,arm,xv,target,bg,mg,device):
    import torch
    tot=0.0; cnt=0; model.eval()
    with torch.no_grad():
        for i in range(0,len(xv),EBS):
            out=model(xv[i:i+EBS]); c=out if arm==ARM_A else (out-mg)@bg.T
            t=torch.from_numpy(np.ascontiguousarray(target[i:i+EBS],dtype=np.float32)).to(device); q=(c-t).double(); tot += float((q*q).sum().item()); cnt += q.numel()
    return tot/cnt

def train_one(arm,seed,x,y,xv,yv,basis,mean,device):
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed); t0=time.perf_counter()
    d=x.shape[1]; ctr=np.ascontiguousarray((y-mean)@basis.T,dtype=np.float32); cva=np.ascontiguousarray((yv-mean)@basis.T,dtype=np.float32)
    xg=torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32)).to(device); xvg=torch.from_numpy(np.ascontiguousarray(xv,dtype=np.float32)).to(device); tg=torch.from_numpy(ctr)
    bg=torch.from_numpy(basis).to(device); mg=torch.from_numpy(mean).to(device)
    m=MLP(d,K if arm==ARM_A else d).to(device); opt=torch.optim.Adam(m.parameters(),LR); mse=nn.MSELoss()
    trc=[]; vc=[]; perms=[]; best=None; sel=None; selstate=None; selhash=None; raw=None; rawep=None; counter=0; nonfinite=0; stop=MAX_EPOCHS; reason='CEILING_REACHED'
    for ep in range(1,MAX_EPOCHS+1):
        g=torch.Generator().manual_seed(seed*1000+ep-1); perm=torch.randperm(len(x),generator=g); perms.append(perm.numpy().astype(np.int32)); m.train(); total=torch.zeros((),dtype=torch.float64,device=device)
        for i in range(0,len(x),BS):
            ic=perm[i:i+BS]; ig=ic.to(device); ct=tg[ic].to(device); opt.zero_grad(); out=m(xg[ig]); cp=out if arm==ARM_A else (out-mg)@bg.T
            loss=mse(cp@bg+mg,ct@bg+mg); loss.backward(); opt.step(); total += loss.detach().double()*len(ic)
        tl=float((total/len(x)).item()); vl=float(coeff_val_mse(m,arm,xvg,cva,bg,mg,device)); trc.append(tl); vc.append(vl)
        if not(np.isfinite(tl) and np.isfinite(vl)): nonfinite += 1
        dh=state_hash(m)
        if raw is None or vl<raw: raw,rawep=vl,ep
        if best is None: best,sel,selstate,selhash=vl,ep,copy.deepcopy(m.state_dict()),dh
        elif (best-vl)/best >= MIN_DELTA: best,sel,selstate,selhash,counter=vl,ep,copy.deepcopy(m.state_dict()),dh,0
        else:
            counter += 1
            if counter>=PATIENCE: stop,reason=ep,'PATIENCE'; break
        if ep%25==0: print(f'[{arm} s{seed}] ep={ep} train={tl:.8e} valK32={vl:.8e} sel={sel}',flush=True)
    rp=replay_stopping(vc,patience=PATIENCE,min_delta=MIN_DELTA)
    obs=dict(selected_epoch=sel,raw_best_epoch=rawep,stop_epoch=stop,stop_reason=reason)
    for k,v in obs.items():
        if rp[k]!=v: raise RuntimeError(f'stopping replay mismatch {arm} s{seed} {k} {v} {rp[k]}')
    if nonfinite: raise RuntimeError(f'nonfinite {arm} s{seed}: {nonfinite}')
    ck=OUT/f'{arm}_seed{seed}.pt'
    if ck.exists(): raise FileExistsError(f'create-only refusal: {ck}')
    torch.save(selstate,ck)
    return dict(arm=arm,seed=seed,selected_epoch=sel,selected_validation_k32_coefficient_mse=best,raw_best_epoch=rawep,raw_best_validation_k32_coefficient_mse=raw,stop_epoch=stop,stop_reason=reason,nonfinite_count=nonfinite,checkpoint_file=ck.name,checkpoint_file_sha256=sha256_file(ck),checkpoint_state_sha256=selhash,batch_order_sha256=hashlib.sha256(np.concatenate(perms).astype(np.int32).tobytes()).hexdigest(),runtime_seconds=time.perf_counter()-t0,train_loss_projected_field_mse_per_epoch=trc,validation_k32_coefficient_mse_per_epoch=vc)

def selftest():
    import torch
    rng=np.random.default_rng(7); d,k=16,4; q,_=np.linalg.qr(rng.normal(size=(d,k))); b=np.ascontiguousarray(q.T,dtype=np.float32); mn=np.ascontiguousarray(rng.normal(size=d),dtype=np.float32); x=np.ascontiguousarray(rng.normal(size=(5,d)),dtype=np.float32); y=np.ascontiguousarray(rng.normal(size=(5,d)),dtype=np.float32); c=np.ascontiguousarray((y-mn)@b.T,dtype=np.float32); bg=torch.from_numpy(b); mg=torch.from_numpy(mn)
    for arm in ARMS:
        m=MLP(d,k if arm==ARM_A else d); o=m(torch.from_numpy(x)); cp=o if arm==ARM_A else (o-mg)@bg.T; loss=((cp@bg+mg)-(torch.from_numpy(c)@bg+mg)).square().mean(); loss.backward(); assert np.isfinite(float(loss))
    try: load_tv([], 'unused'); raise RuntimeError('UNUSED firewall failed')
    except PermissionError: pass
    print('SELFTEST_PASS')

def preflight(device):
    u=upstream(True); OUT.mkdir(parents=True,exist_ok=True)
    import torch
    rec=dict(protocol_id=PID,status='PREFLIGHT_COMPLETE',runner_sha256=sha256_file(pathlib.Path(__file__)),protocol_sha256=sha256_file(PROTOCOL),stage_a_json_sha256=sha256_file(u['ap']),stage_b_json_sha256=sha256_file(u['bp']),stage_c_json_sha256=sha256_file(u['cp']),parent_basis_sha256=u['a']['arrays']['parent_basis_float32.npy'],train_mean_sha256=u['a']['arrays']['train_mean_float32.npy'],k=K,k32_index_digest=K_DIGEST,k32_indices=u['idx'],seeds=list(SEEDS),arms=list(ARMS),dense_reference_checkpoints=u['dense'],inherited_train_file_digests={p.name:h for _,p,h in u['files']},training_policy=dict(optimizer='Adam',learning_rate=LR,batch_size=BS,evaluation_batch_size=EBS,max_epochs=MAX_EPOCHS,patience=PATIENCE,min_delta=MIN_DELTA,permutation_seed='seed*1000+epoch-1',fit_metric='K32 projected-field MSE',checkpoint_metric='K32 validation coefficient MSE only'),role_policy=dict(train_allowed=['TRAIN','VALIDATION'],train_forbidden=['UNUSED','SELECTION','TEST'],unused_rule='separate final evaluator only after six-checkpoint TRAIN_SEAL'),environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,device=device),written_at_utc=now())
    rec['preflight_payload_sha256']=jhash(rec); newj(OUT/'PREFLIGHT.json',rec); print(json.dumps({k:rec[k] for k in ('status','runner_sha256','protocol_sha256','k32_index_digest')},indent=2))

def train(device):
    pf=OUT/'PREFLIGHT.json'
    if not pf.is_file(): raise RuntimeError('PREFLIGHT missing')
    p=readj(pf)
    if p['status']!='PREFLIGHT_COMPLETE' or p['runner_sha256']!=sha256_file(pathlib.Path(__file__)) or p['protocol_sha256']!=sha256_file(PROTOCOL): raise RuntimeError('preflight identity mismatch')
    if (OUT/'TRAIN_SEAL.json').exists(): raise FileExistsError('TRAIN_SEAL exists')
    u=upstream(True); basis=np.ascontiguousarray(np.load(u['basis'])[np.asarray(u['idx'],dtype=np.int64)],dtype=np.float32); mean=np.ascontiguousarray(np.load(u['mean']),dtype=np.float32)
    x,y,ntr,idtr=load_tv(u['files'],'train'); xv,yv,nva,idva=load_tv(u['files'],'validation'); access=dict(train=ntr,validation=nva,unused=0,selection=0,test=0)
    traces=[]
    for arm in ARMS:
        for s in SEEDS:
            print(f'BEGIN {arm} seed={s}',flush=True); z=train_one(arm,s,x,y,xv,yv,basis,mean,device); traces.append(z); print(f'END {arm} seed={s} selected={z["selected_epoch"]} stop={z["stop_epoch"]}',flush=True)
            if str(device).startswith('cuda'):
                import torch; torch.cuda.empty_cache()
    ck={f'{z["arm"]}_seed{z["seed"]}':dict(file=z['checkpoint_file'],file_sha256=z['checkpoint_file_sha256'],state_sha256=z['checkpoint_state_sha256'],selected_epoch=z['selected_epoch'],stop_epoch=z['stop_epoch'],stop_reason=z['stop_reason'],selected_validation_k32_coefficient_mse=z['selected_validation_k32_coefficient_mse']) for z in traces}
    if len(ck)!=6: raise RuntimeError('six checkpoints not produced')
    seal=dict(protocol_id=PID,status='TRAIN_SEALED_6_OF_6',runner_sha256=sha256_file(pathlib.Path(__file__)),protocol_sha256=sha256_file(PROTOCOL),preflight_sha256=sha256_file(pf),stage_a_json_sha256=sha256_file(u['ap']),stage_b_json_sha256=sha256_file(u['bp']),stage_c_json_sha256=sha256_file(u['cp']),parent_basis_sha256=u['a']['arrays']['parent_basis_float32.npy'],train_mean_sha256=u['a']['arrays']['train_mean_float32.npy'],k=K,k32_index_digest=K_DIGEST,dense_reference_checkpoints=u['dense'],new_checkpoints=ck,traces=traces,access_counters=access,trajectory_ids=dict(train=idtr,validation=idva),explicitly_unopened_roles=['UNUSED','SELECTION','TEST'],written_at_utc=now()); seal['seal_payload_sha256']=jhash(seal); newj(OUT/'TRAIN_SEAL.json',seal); print(json.dumps(dict(status=seal['status'],access=access,selected_epochs={k:v['selected_epoch'] for k,v in ck.items()}),indent=2))

def load_state(p,device):
    import torch
    try: return torch.load(p,map_location=device,weights_only=True)
    except TypeError: return torch.load(p,map_location=device)
def raw_mse(m,kind,x,y,basis,mean,device):
    import torch
    xg=torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32)).to(device); bg=torch.from_numpy(basis).to(device); mg=torch.from_numpy(mean).to(device); total=0.0; cnt=0; m.eval()
    with torch.no_grad():
        for i in range(0,len(x),EBS):
            o=m(xg[i:i+EBS]); f=o if kind=='dense' else (o@bg+mg if kind==ARM_A else (((o-mg)@bg.T)@bg+mg)); t=torch.from_numpy(np.ascontiguousarray(y[i:i+EBS],dtype=np.float32)).to(device); q=(f-t).double(); total += float((q*q).sum().item()); cnt += q.numel()
    return total/cnt

def eval_unused(device):
    sealp=OUT/'TRAIN_SEAL.json'
    if not sealp.is_file(): raise RuntimeError('UNUSED refused: TRAIN_SEAL missing')
    if (OUT/'UNUSED_OPEN_RECEIPT.json').exists() or (OUT/'UNUSED_EVAL.json').exists(): raise FileExistsError('UNUSED one-shot already attempted')
    s=readj(sealp)
    if s['status']!='TRAIN_SEALED_6_OF_6' or s['runner_sha256']!=sha256_file(pathlib.Path(__file__)) or s['protocol_sha256']!=sha256_file(PROTOCOL) or s['k32_index_digest']!=K_DIGEST or s['access_counters']['unused']!=0 or len(s['new_checkpoints'])!=6: raise RuntimeError('UNUSED authorization failed')
    u=upstream(True)
    if (sha256_file(u['ap']),sha256_file(u['bp']),sha256_file(u['cp']))!=(s['stage_a_json_sha256'],s['stage_b_json_sha256'],s['stage_c_json_sha256']): raise RuntimeError('upstream changed after training')
    for e in s['new_checkpoints'].values():
        if sha256_file(OUT/e['file'])!=e['file_sha256']: raise RuntimeError('new checkpoint changed after seal')
    receipt=dict(protocol_id=PID,status='UNUSED_OPEN_AUTHORIZED_ONCE',runner_sha256=s['runner_sha256'],train_seal_sha256=sha256_file(sealp),authorized_trajectory_count=160,original_test_authorized=False,written_at_utc=now()); newj(OUT/'UNUSED_OPEN_RECEIPT.json',receipt)
    import h5py
    xs=[]; ys=[]; idrec={}; opens=0; reads=0
    for fi,(fam,p,_) in enumerate(u['files']):
        tr,va,un=split_family(SPLIT_SEED,fi); ids=sorted(map(int,un))
        if len(ids)!=20 or set(ids)&set(map(int,tr)) or set(ids)&set(map(int,va)): raise RuntimeError('UNUSED split invalid')
        with h5py.File(p,'r') as h:
            opens += 1; dset=h[list(h.keys())[0]]['u']; z=dset[ids]; reads += len(ids)
        z=z[np.argsort(np.argsort(np.asarray(un,dtype=np.int64)))]; d=int(z.shape[-2]*z.shape[-1]); xs.append(z[:,:-1].reshape(-1,d).astype(np.float32)); ys.append(z[:,1:].reshape(-1,d).astype(np.float32)); idrec[str(fam)]=list(map(int,un))
    x=np.concatenate(xs); y=np.concatenate(ys); basis=np.ascontiguousarray(np.load(u['basis'])[np.asarray(u['idx'],dtype=np.int64)],dtype=np.float32); mean=np.ascontiguousarray(np.load(u['mean']),dtype=np.float32); d=x.shape[1]
    rows=[]
    for seed in SEEDS:
        de=u['b']['reference_checkpoints'][str(seed)]; dm=MLP(d,d).to(device); dm.load_state_dict(load_state(B_DIR/de['path'],device)); ae=s['new_checkpoints'][f'{ARM_A}_seed{seed}']; be=s['new_checkpoints'][f'{ARM_B}_seed{seed}']; am=MLP(d,K).to(device); bm=MLP(d,d).to(device); am.load_state_dict(load_state(OUT/ae['file'],device)); bm.load_state_dict(load_state(OUT/be['file'],device)); md=raw_mse(dm,'dense',x,y,basis,mean,device); ma=raw_mse(am,ARM_A,x,y,basis,mean,device); mb=raw_mse(bm,ARM_B,x,y,basis,mean,device); r=dict(seed=seed,dense_reference_mse=md,arm_a_compact_reducedval_mse=ma,arm_b_densehead_reducedval_mse=mb,arm_a_over_dense=ma/md,arm_b_over_dense=mb/md,arm_b_over_arm_a=mb/ma); rows.append(r); print(json.dumps(r,sort_keys=True),flush=True)
        if str(device).startswith('cuda'):
            import torch; torch.cuda.empty_cache()
    ra=np.asarray([r['arm_a_over_dense'] for r in rows]); rb=np.asarray([r['arm_b_over_dense'] for r in rows]); rba=np.asarray([r['arm_b_over_arm_a'] for r in rows]); summary=dict(arm_a_median_over_dense=float(np.median(ra)),arm_b_median_over_dense=float(np.median(rb)),arm_b_median_over_arm_a=float(np.median(rba)),arm_a_adverse_seed_count_ratio_gt_1=int((ra>1).sum()),arm_b_adverse_seed_count_ratio_gt_1=int((rb>1).sum()),arm_a_fail_seed_count_ratio_gt_1p05=int((ra>1.05).sum()),arm_b_fail_seed_count_ratio_gt_1p05=int((rb>1.05).sum()),arm_a_all_three_le_1p05=bool(np.all(ra<=1.05)),arm_b_all_three_le_1p05=bool(np.all(rb<=1.05)))
    rec=dict(protocol_id=PID,status='UNUSED_EVAL_COMPLETE',runner_sha256=s['runner_sha256'],train_seal_sha256=sha256_file(sealp),unused_open_receipt_sha256=sha256_file(OUT/'UNUSED_OPEN_RECEIPT.json'),unused_access=dict(hdf5_opens=opens,unused_trajectory_reads=reads),unused_ids_per_family=idrec,original_test_access_count=0,selection_access_count=0,rows_evaluated=int(len(x)),field_dimension=int(d),results=rows,summary=summary,interpretation_guard=['Arm A tests reduced-only checkpoint control for the compact K32 learner.','Arm B tests dense-output parameterization under the same reduced-only fitting/checkpoint interface.','Arm B versus historical dense is not a pure parameter-count causal ablation because checkpoint metric also differs.','Original TEST is never read.'],written_at_utc=now()); newj(OUT/'UNUSED_EVAL.json',rec); print(json.dumps(dict(status=rec['status'],unused_access=rec['unused_access'],summary=summary),indent=2))

def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['selftest','preflight','train','eval']); p.add_argument('--device',default='cuda'); a=p.parse_args()
    if a.mode=='selftest': selftest()
    elif a.mode=='preflight': preflight(a.device)
    elif a.mode=='train': train(a.device)
    else: eval_unused(a.device)
if __name__=='__main__': main()
