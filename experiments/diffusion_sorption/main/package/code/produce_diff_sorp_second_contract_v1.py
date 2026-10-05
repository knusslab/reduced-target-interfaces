"""Production primitives for DIFF_SORP_SECOND_CONTRACT_V1(+V1.1).

All routines are deterministic and create-only at artifact boundaries. This module can be unit-tested
without scientific data. The actual TEST entry point validates sealed source identities before any
basis, checkpoint, or HDF5 access and consumes its marker before TEST values are read.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import diff_sorp_second_contract_v1 as core
from run_diff_sorp_second_contract_v1 import load_role_pairs


def configure_cpu_determinism():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    return {"device":"cpu","dtype":"float32","torch_num_threads":1,
            "deterministic_algorithms":True}


def _state_digest_from_state(state) -> str:
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        h.update(name.encode('utf-8'))
        h.update(np.ascontiguousarray(tensor.detach().cpu().numpy(),dtype=np.float32).tobytes())
    return h.hexdigest()


def _physical_prediction(model,x,basis_rows=None,mean=None,batch_size=512):
    model.eval(); xt=torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32)); chunks=[]
    B=(None if basis_rows is None else torch.from_numpy(np.ascontiguousarray(basis_rows,dtype=np.float32)))
    mu=(None if mean is None else torch.from_numpy(np.ascontiguousarray(mean,dtype=np.float32).reshape(-1)))
    with torch.no_grad():
        for start in range(0,len(xt),int(batch_size)):
            raw=model(xt[start:start+int(batch_size)])
            if B is not None:
                raw=raw@B+mu
            chunks.append(raw.cpu().numpy())
    return np.concatenate(chunks,0).astype(np.float64)


def _validation_mse(model,xv,yv,basis_rows=None,mean=None):
    pred=_physical_prediction(model,xv,basis_rows,mean)
    diff=pred-np.asarray(yv,dtype=np.float64)
    return float(np.mean(diff*diff))


def train_cell(arm,seed,x_train,target_train,x_validation,y_validation,output_dir,*,hidden,dout,
               basis_rows,mean,max_epochs=core.MAX_EPOCHS,patience=core.PATIENCE,
               min_delta=core.MIN_DELTA,batch_size=core.BATCH_SIZE,lr=core.LEARNING_RATE):
    configure_cpu_determinism(); torch.manual_seed(int(seed)); np.random.seed(int(seed))
    started=time.perf_counter()
    x=torch.from_numpy(np.ascontiguousarray(x_train,dtype=np.float32))
    target=torch.from_numpy(np.ascontiguousarray(target_train,dtype=np.float32))
    B=(None if basis_rows is None else torch.from_numpy(np.ascontiguousarray(basis_rows,dtype=np.float32)))
    model=core.MLP(int(x.shape[1]),int(dout),hidden=int(hidden))
    opt=torch.optim.Adam(model.parameters(),float(lr))
    train_curve=[]; val_curve=[]; perms=[]
    selected_state=None; selected_digest=None; selected_epoch=None; running_best=None
    raw_best=None; raw_best_epoch=None; raw_best_digest=None
    counter=0; nonfinite=0; stop_epoch=None; stop_reason='CEILING_REACHED'
    n=len(x)
    for epoch in range(1,int(max_epochs)+1):
        perm_np=core.batch_permutation(n,seed=int(seed),epoch=epoch).astype(np.int64)
        perms.append(perm_np.astype(np.int32)); perm=torch.from_numpy(perm_np)
        model.train(); total=torch.zeros((),dtype=torch.float64)
        for start in range(0,n,int(batch_size)):
            idx=perm[start:start+int(batch_size)]; opt.zero_grad(); pred=model(x[idx])
            if arm=='dense':
                loss=torch.mean((pred-target[idx])**2)
            elif arm=='compressed':
                if B is None or mean is None: raise ValueError('compressed arm requires basis and mean')
                loss=core.projected_field_mse_from_coefficients(pred,target[idx],B)
            else:
                raise ValueError(f'unknown arm {arm}')
            loss.backward(); opt.step(); total += loss.detach().double()*idx.shape[0]
        train_loss=float((total/n).item())
        val=_validation_mse(model,x_validation,y_validation,basis_rows if arm=='compressed' else None,
                            mean if arm=='compressed' else None)
        train_curve.append(train_loss); val_curve.append(val)
        if not (np.isfinite(train_loss) and np.isfinite(val)):
            nonfinite+=1; stop_epoch=epoch; stop_reason='NONFINITE'; break
        digest=_state_digest_from_state(model.state_dict())
        if raw_best is None or val<raw_best:
            raw_best=val; raw_best_epoch=epoch; raw_best_digest=digest
        if running_best is None:
            running_best=val; selected_epoch=epoch
            selected_state=copy.deepcopy(model.state_dict()); selected_digest=digest
        elif (running_best-val)/running_best >= float(min_delta):
            running_best=val; selected_epoch=epoch; counter=0
            selected_state=copy.deepcopy(model.state_dict()); selected_digest=digest
        else:
            counter+=1
            if counter>=int(patience): stop_epoch=epoch; stop_reason='PATIENCE'; break
    if stop_epoch is None: stop_epoch=len(val_curve)
    if stop_reason!='NONFINITE':
        replay=core.replay_stopping(val_curve,patience=patience,min_delta=min_delta)
        for key,got in {'selected_epoch':selected_epoch,'stop_epoch':stop_epoch,
                        'stop_reason':stop_reason,'raw_best_epoch':raw_best_epoch}.items():
            if replay[key]!=got: raise core.TrainingGateFail(f'{arm} seed {seed}: stopping replay mismatch {key}')
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); ckpt=out/f'{arm}_seed{int(seed)}.pt'
    file_sha=None
    if selected_state is not None:
        if ckpt.exists(): raise FileExistsError(ckpt)
        torch.save(selected_state,ckpt); file_sha=core.sha256_file(ckpt)
    return {'arm':str(arm),'seed':int(seed),'hidden':int(hidden),'input_dim':int(x.shape[1]),
            'output_dim':int(dout),'target_space':('raw_field' if arm=='dense' else 'selected_coefficients'),
            'parameter_count':int(sum(q.numel() for q in model.parameters())),
            'train_loss_per_epoch':train_curve,'validation_mse_per_epoch':val_curve,
            'raw_best_epoch':(None if raw_best_epoch is None else int(raw_best_epoch)),
            'raw_best_validation_mse':(None if raw_best is None else float(raw_best)),
            'raw_best_checkpoint_sha256':raw_best_digest,
            'selected_epoch':(None if selected_epoch is None else int(selected_epoch)),
            'selected_validation_mse':(None if running_best is None else float(running_best)),
            'checkpoint_state_sha256':selected_digest,'checkpoint_file_sha256':file_sha,
            'checkpoint_path':(None if selected_state is None else ckpt.name),
            'stop_epoch':int(stop_epoch),'stop_reason':stop_reason,
            'converged':stop_reason=='PATIENCE','nonfinite_count':int(nonfinite),
            'batch_order_sha256':hashlib.sha256(np.concatenate(perms).astype(np.int32).tobytes()).hexdigest(),
            'runtime_seconds':float(time.perf_counter()-started)}


def selector_from_predictions(y,pred,basis,mean,*,ladder=core.LADDER,tau=core.PRIMARY_TAU):
    y=np.asarray(y,dtype=np.float64); pred=np.asarray(pred,dtype=np.float64)
    B=np.asarray(basis,dtype=np.float64); mu=np.asarray(mean,dtype=np.float64).reshape(-1)
    if y.shape!=pred.shape or y.ndim!=2 or y.shape[1]!=B.shape[1] or mu.shape[0]!=y.shape[1]:
        raise ValueError('selection shapes incompatible')
    c=(y-mu)@B.T; chat=(pred-mu)@B.T
    q=np.mean(c*c,axis=0); e=np.mean((c-chat)**2,axis=0); gain=q-e
    reference_mse=float(np.mean((pred-y)**2)); threshold=float(tau)*y.shape[1]*reference_mse
    chosen=core.select_budget(gain,ladder=ladder,threshold=threshold)
    return {'K_prop':int(chosen['selected_k']),'indices':[int(i) for i in chosen['indices']],
            'indices_sha256':chosen['indices_sha256'],'order_sha256':chosen['order_sha256'],'gain_sha256':chosen['gain_sha256'],
            'reference_mse':reference_mse,'threshold':threshold,'tau':float(tau),
            'q_sha256':core.float64_array_digest(q),'e_sha256':core.float64_array_digest(e),
            'positive_gain_directions':int((gain>0).sum())}


def build_selection_record(y,predictions_by_seed,basis,mean,*,ladder=core.LADDER,tau=core.PRIMARY_TAU):
    if sorted(predictions_by_seed)!=list(core.SEEDS): raise core.ProvenanceFail('dense seed set changed')
    primary=selector_from_predictions(y,predictions_by_seed[0],basis,mean,ladder=ladder,tau=tau)
    diagnostics={str(s):selector_from_predictions(y,predictions_by_seed[s],basis,mean,ladder=ladder,tau=tau)
                 for s in (1,2)}
    return {'protocol_id':core.PROTOCOL_ID,'protocol_sha256':core.PROTOCOL_SHA256,
            'addendum_sha256':core.ADDENDUM_SHA256,'addendum_v1_2_sha256':core.ADDENDUM_V1_2_SHA256,'addendum_v1_3_sha256':core.ADDENDUM_V1_3_SHA256,'addendum_v1_4_sha256':core.ADDENDUM_V1_4_SHA256,
            'official_source_blobs':dict(core.OFFICIAL_SOURCE_BLOBS_V1_4),'status':'SELECTION_SEALED','proposal_seed':0,
            'K_prop':primary['K_prop'],'indices':primary['indices'],
            'indices_sha256':primary['indices_sha256'],'order_sha256':primary['order_sha256'],'gain_sha256':primary['gain_sha256'],
            'selection_split_sha256':core.SPLIT_DIGESTS['selection'],'primary':primary,
            'stability_diagnostics':diagnostics,'ladder':[int(k) for k in ladder],'tau':float(tau)}


def _load_npz_per_trajectory(path):
    with np.load(path) as z:
        return {name:np.asarray(z[name],dtype=np.float64) for name in z.files}


def result_from_per_trajectory(raw_path,*,replicates=core.BOOTSTRAP_REPLICATES,seed=core.BOOTSTRAP_SEED):
    raw=_load_npz_per_trajectory(raw_path)
    nested={'dense':{},'compressed':{}}
    for arm in nested:
        for s in core.SEEDS:
            key=f'{arm}_seed{s}'
            if key not in raw: raise core.ProvenanceFail(f'missing {key}')
            nested[arm][s]=raw[key]
    dense={s:float(nested['dense'][s].mean()) for s in core.SEEDS}
    compressed={s:float(nested['compressed'][s].mean()) for s in core.SEEDS}
    ratios={str(s):compressed[s]/dense[s] for s in core.SEEDS}
    q=float(np.median(list(ratios.values())))
    boot=core.paired_trajectory_bootstrap(nested,dense_arm='dense',replicates=replicates,seed=seed)
    return {'protocol_id':core.PROTOCOL_ID,'status':'TEST_EVALUATION_COMPLETE','Q_TEST':q,
            'verdict':('PASS' if q<=core.QUALITY_TOLERANCE else 'FAIL'),
            'tolerance':core.QUALITY_TOLERANCE,'per_seed_ratios':ratios,
            'dense_per_seed_mse':{str(s):dense[s] for s in core.SEEDS},
            'compressed_per_seed_mse':{str(s):compressed[s] for s in core.SEEDS},
            'bootstrap':boot['arms']['compressed'],'bootstrap_replicates':int(replicates),
            'bootstrap_seed':int(seed),'raw_per_trajectory_sha256':core.sha256_file(raw_path)}


def verify_test_result(result,raw_path,*,replicates=core.BOOTSTRAP_REPLICATES,seed=core.BOOTSTRAP_SEED):
    expected=result_from_per_trajectory(raw_path,replicates=replicates,seed=seed)
    if result!=expected:
        keys=sorted(set(result)|set(expected)); mismatch=next((k for k in keys if result.get(k)!=expected.get(k)),'unknown')
        raise core.ProvenanceFail(f'TEST result mismatch: {mismatch}')
    return True


def _verify_static_pretest_bindings(pre):
    if pre.get('status')!='PRETEST_SEAL_READY': raise core.ProvenanceFail('pretest seal not ready')
    if pre.get('protocol_sha256')!=core.PROTOCOL_SHA256: raise core.ProvenanceFail('protocol drift')
    if pre.get('addendum_sha256')!=core.ADDENDUM_SHA256: raise core.ProvenanceFail('addendum drift')
    if pre.get('addendum_v1_2_sha256')!=core.ADDENDUM_V1_2_SHA256: raise core.ProvenanceFail('V1.2 addendum drift')
    if pre.get('addendum_v1_3_sha256')!=core.ADDENDUM_V1_3_SHA256: raise core.ProvenanceFail('V1.3 addendum drift')
    if pre.get('addendum_v1_4_sha256')!=core.ADDENDUM_V1_4_SHA256: raise core.ProvenanceFail('V1.4 addendum drift')
    if pre.get('official_source_blobs')!=core.OFFICIAL_SOURCE_BLOBS_V1_4: raise core.ProvenanceFail('official source registry drift')
    current=core.sha256_file(Path(__file__))
    if pre.get('evaluator_source_sha256')!=current: raise core.ProvenanceFail('evaluator source drift')
    bootstrap=core.sha256_file(Path(core.__file__))
    if pre.get('bootstrap_source_sha256')!=bootstrap: raise core.ProvenanceFail('bootstrap source drift')
    counters=pre.get('access_counters',{})
    if int(counters.get('test',-1))!=0 or int(counters.get('unused',-1))!=0:
        raise core.ProvenanceFail('pretest forbidden access counter changed')
    return True


def _load_model(path,din,dout,hidden=core.DENSE_HIDDEN):
    model=core.MLP(din,dout,hidden=hidden)
    state=torch.load(path,map_location='cpu',weights_only=True); model.load_state_dict(state); model.eval(); return model


def load_test_transaction(data_file,dense_dir,compressed_dir,basis_dir,pretest,selection_dir=None):
    """Load all fixed objects and TEST rows; caller must consume marker before calling this."""
    basis_dir=Path(basis_dir)
    B=np.load(basis_dir/'BASIS_FLOAT32.npy'); mean=np.load(basis_dir/'TRAIN_MEAN_FLOAT32.npy')
    if selection_dir is None: raise core.ProvenanceFail('selection_dir required after source precheck')
    idx=np.load(Path(selection_dir)/'SELECTED_INDICES_INT64.npy').astype(np.int64)
    rows=np.ascontiguousarray(B[idx],dtype=np.float32)
    x,y,groups=load_role_pairs(data_file,'test',stage='TEST_EVAL')
    dense={s:_load_model(Path(dense_dir)/f'dense_seed{s}.pt',core.FIELD_DIM,core.FIELD_DIM) for s in core.SEEDS}
    comp={s:_load_model(Path(compressed_dir)/f'compressed_seed{s}.pt',core.FIELD_DIM,len(idx)) for s in core.SEEDS}
    return x,y,groups,dense,comp,rows,mean


def test_evaluation(pretest_path,eval_dir,*,data_file,dense_dir,compressed_dir,basis_dir,selection_dir=None,
                    replicates=core.BOOTSTRAP_REPLICATES,seed=core.BOOTSTRAP_SEED):
    pre=json.loads(Path(pretest_path).read_text(encoding='utf-8'))
    # Immutable source/protocol bindings first: no basis, model or HDF5 access before this passes.
    _verify_static_pretest_bindings(pre)
    eval_dir=Path(eval_dir); eval_dir.mkdir(parents=True,exist_ok=True)
    marker=core.OneShotMarker(eval_dir/'TEST_ACCESS_CONSUMED.json',
                              {'pretest_seal_sha256':core.sha256_file(pretest_path),
                               'evaluator_source_sha256':core.sha256_file(Path(__file__)),
                               'bootstrap_source_sha256':core.sha256_file(Path(core.__file__))})
    marker.open()
    x,y,groups,dense,comp,rows,mean=load_test_transaction(
        data_file,dense_dir,compressed_dir,basis_dir,pre,selection_dir=selection_dir)
    per={}
    for arm,models in (('dense',dense),('compressed',comp)):
        for s in core.SEEDS:
            pred=_physical_prediction(models[s],x,rows if arm=='compressed' else None,
                                      mean if arm=='compressed' else None)
            row=((pred-np.asarray(y,dtype=np.float64))**2).mean(axis=1)
            per[f'{arm}_seed{s}']=np.asarray([row[groups==g].mean() for g in np.unique(groups)],dtype=np.float64)
    raw=eval_dir/'TEST_PER_TRAJECTORY.npz'
    if raw.exists(): raise FileExistsError(raw)
    np.savez(raw,**per)
    result=result_from_per_trajectory(raw,replicates=replicates,seed=seed)
    path=eval_dir/'TEST_RESULT.json'
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return result
