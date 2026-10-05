from pathlib import Path
import copy
import json
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import diff_sorp_second_contract_v1 as core
import produce_diff_sorp_second_contract_v1 as p


def small_basis(k=4,d=9,seed=3):
    rng=np.random.default_rng(seed)
    q,_=np.linalg.qr(rng.normal(size=(d,k)))
    return q.T.astype(np.float32)


def test_dense_training_cell_is_deterministic_and_replays(tmp_path):
    rng=np.random.default_rng(1)
    x=rng.normal(size=(32,9)).astype(np.float32)
    y=rng.normal(size=(32,7)).astype(np.float32)
    xv=rng.normal(size=(12,9)).astype(np.float32)
    yv=rng.normal(size=(12,7)).astype(np.float32)
    t1=p.train_cell('dense',1,x,y,xv,yv,tmp_path/'a',hidden=7,dout=7,
                    basis_rows=None,mean=None,max_epochs=4,patience=10,batch_size=8)
    t2=p.train_cell('dense',1,x,y,xv,yv,tmp_path/'b',hidden=7,dout=7,
                    basis_rows=None,mean=None,max_epochs=4,patience=10,batch_size=8)
    assert t1['train_loss_per_epoch']==pytest.approx(t2['train_loss_per_epoch'])
    assert t1['validation_mse_per_epoch']==pytest.approx(t2['validation_mse_per_epoch'])
    assert t1['batch_order_sha256']==t2['batch_order_sha256']
    assert t1['output_dim']==7


def test_compressed_training_uses_coefficients_and_decodes(tmp_path):
    rng=np.random.default_rng(2)
    x=rng.normal(size=(24,9)).astype(np.float32)
    B=small_basis(k=3,d=7)
    mu=rng.normal(size=7).astype(np.float32)
    c=rng.normal(size=(24,3)).astype(np.float32)
    xv=rng.normal(size=(8,9)).astype(np.float32)
    yv=rng.normal(size=(8,7)).astype(np.float32)
    t=p.train_cell('compressed',0,x,c,xv,yv,tmp_path/'c',hidden=6,dout=3,
                   basis_rows=B,mean=mu,max_epochs=3,patience=10,batch_size=6)
    assert t['target_space']=='selected_coefficients'
    assert t['output_dim']==3
    assert t['parameter_count']==sum(q.numel() for q in core.MLP(9,3,hidden=6).parameters())


def test_ceiling_is_not_converged(tmp_path):
    rng=np.random.default_rng(3)
    x=rng.normal(size=(16,5)).astype(np.float32); y=rng.normal(size=(16,4)).astype(np.float32)
    xv=rng.normal(size=(8,5)).astype(np.float32); yv=rng.normal(size=(8,4)).astype(np.float32)
    t=p.train_cell('dense',0,x,y,xv,yv,tmp_path/'d',hidden=4,dout=4,basis_rows=None,mean=None,
                   max_epochs=2,patience=20,batch_size=8)
    assert t['stop_reason']=='CEILING_REACHED' and t['converged'] is False


def test_selection_is_seed0_only_and_stability_does_not_change_proposal():
    rng=np.random.default_rng(5)
    B=small_basis(k=6,d=8); mu=np.zeros(8,dtype=np.float32)
    y=rng.normal(size=(30,8)).astype(np.float32)
    pred0=y.copy(); pred0[:,3:]+=2.0
    pred1=y.copy(); pred1[:,:2]+=3.0
    pred2=y.copy(); pred2[:,4:]+=4.0
    rec=p.build_selection_record(y,{0:pred0,1:pred1,2:pred2},B,mu,
                                 ladder=(1,2,3,4,5,6),tau=0.05)
    seed0=p.selector_from_predictions(y,pred0,B,mu,ladder=(1,2,3,4,5,6),tau=0.05)
    assert rec['proposal_seed']==0
    assert rec['K_prop']==seed0['K_prop']
    assert rec['indices']==seed0['indices']
    assert set(rec['stability_diagnostics'])=={'1','2'}
    assert rec['addendum_v1_4_sha256']==core.ADDENDUM_V1_4_SHA256
    assert rec['official_source_blobs']==core.OFFICIAL_SOURCE_BLOBS_V1_4


def test_test_evaluator_refuses_source_drift_before_marker(tmp_path,monkeypatch):
    pre={'status':'PRETEST_SEAL_READY','protocol_sha256':core.PROTOCOL_SHA256,
         'addendum_sha256':core.ADDENDUM_SHA256,'addendum_v1_2_sha256':core.ADDENDUM_V1_2_SHA256,'addendum_v1_3_sha256':core.ADDENDUM_V1_3_SHA256,'addendum_v1_4_sha256':core.ADDENDUM_V1_4_SHA256,
         'official_source_blobs':core.OFFICIAL_SOURCE_BLOBS_V1_4,
         'evaluator_source_sha256':'x'*64,
         'bootstrap_source_sha256':'z'*64}
    pre_path=tmp_path/'PRETEST_SEAL.json'; pre_path.write_text(json.dumps(pre))
    called={'loader':0}
    def bomb(*a,**k): called['loader']+=1; raise AssertionError('TEST loader reached')
    monkeypatch.setattr(p,'load_test_transaction',bomb)
    with pytest.raises(core.ProvenanceFail,match='evaluator'):
        p.test_evaluation(pre_path,tmp_path/'eval',data_file='nope',dense_dir='nope',
                          compressed_dir='nope',basis_dir='nope')
    assert called['loader']==0
    assert not (tmp_path/'eval'/'TEST_ACCESS_CONSUMED.json').exists()


def test_result_verifier_recomputes_point_and_rejects_mutation(tmp_path):
    rng=np.random.default_rng(6)
    per={}
    for seed in core.SEEDS:
        base=rng.uniform(.8,1.2,size=10)
        per[f'dense_seed{seed}']=base
        per[f'compressed_seed{seed}']=base*0.98
    raw=tmp_path/'TEST_PER_TRAJECTORY.npz'; np.savez(raw,**per)
    result=p.result_from_per_trajectory(raw,replicates=200,seed=77)
    assert result['Q_TEST']==pytest.approx(0.98)
    assert p.verify_test_result(result,raw,replicates=200,seed=77)
    bad=copy.deepcopy(result); bad['Q_TEST']+=0.1
    with pytest.raises(core.ProvenanceFail): p.verify_test_result(bad,raw,replicates=200,seed=77)
