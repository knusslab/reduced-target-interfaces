from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

PKG = Path(__file__).resolve().parents[1]
RUNNER_PATH = PKG / 'code' / 'run_diff_sorp_r2_unused_cross_learner_v1.py'
AUDIT_PATH = PKG / 'code' / 'audit_diff_sorp_r2_unused_cross_learner_v1.py'


def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); assert spec and spec.loader
    sys.modules[name]=mod; spec.loader.exec_module(mod); return mod

r=load_module('portability_runner_v1',RUNNER_PATH)
a=load_module('portability_audit_v1',AUDIT_PATH)


def require_inherited_artifacts():
    required = [
        PKG / 'inherited' / 'BASIS_FLOAT32.npy',
        PKG / 'inherited' / 'TRAIN_MEAN_FLOAT32.npy',
        PKG / 'inherited' / 'SELECTED_INDICES_INT64.npy',
    ]
    if not all(path.is_file() for path in required):
        import pytest
        pytest.skip('derived inherited arrays are intentionally not bundled pending redistribution review')


def test_frozen_splits_match_all_digests_and_are_disjoint():
    s=r.frozen_splits()
    assert {k:len(v) for k,v in s.items()} == {'train':480,'validation':160,'unused':160,'selection':200}
    flat=sum([s[k] for k in ('train','validation','unused','selection')],[])
    assert len(flat)==len(set(flat))
    for role in s:
        assert r.key_digest(s[role])==r.SPLIT_DIGESTS[role]


def test_inherited_target_representation_exact():
    require_inherited_artifacts()
    rec=r.verify_inherited(PKG)
    assert rec['K']==8
    assert rec['indices']==[0,1,2,3,4,5,6,8]
    assert rec['basis_sha256']==r.BASIS_SHA256
    assert rec['mean_sha256']==r.MEAN_SHA256
    assert rec['indices_sha256']==r.INDICES_SHA256
    assert rec['selected_basis_gram_residual_max_abs'] < 2e-5


def test_fno_parameter_count_and_paired_init_are_stable():
    m=r.FNO1d()
    assert r.param_count(m)==74209
    s1,d1,sh1=r.fno_initial_state(10)
    s2,d2,sh2=r.fno_initial_state(10)
    assert d1==d2
    assert sh1==sh2
    for k in s1:
        assert torch.equal(s1[k],s2[k])


def test_projected_coefficient_loss_matches_decoded_field_mse():
    require_inherited_artifacts()
    basis=np.load(PKG/'inherited'/'BASIS_FLOAT32.npy',allow_pickle=False)
    idx=np.load(PKG/'inherited'/'SELECTED_INDICES_INT64.npy',allow_pickle=False)
    rows=np.ascontiguousarray(basis[idx],dtype=np.float32)
    rng=np.random.default_rng(42)
    p=torch.from_numpy(rng.normal(size=(17,8)).astype(np.float32))
    t=torch.from_numpy(rng.normal(size=(17,8)).astype(np.float32))
    B=torch.from_numpy(rows)
    got=float(r.projected_field_loss_coeff(p,t,B))
    decoded=(p-t)@B
    expected=float(torch.sum(decoded*decoded)/(17*1024))
    assert abs(got-expected) < 2e-9


def test_coeff_encode_decode_uses_exact_selected_rows():
    require_inherited_artifacts()
    basis=np.load(PKG/'inherited'/'BASIS_FLOAT32.npy',allow_pickle=False)
    mean=np.load(PKG/'inherited'/'TRAIN_MEAN_FLOAT32.npy',allow_pickle=False)
    idx=np.load(PKG/'inherited'/'SELECTED_INDICES_INT64.npy',allow_pickle=False)
    rows=np.ascontiguousarray(basis[idx],dtype=np.float32)
    rng=np.random.default_rng(1)
    y=rng.normal(size=(9,1024)).astype(np.float32)
    c=r.coeff_targets(y,mean,rows)
    dec=r.decode_coeff(c,mean,rows)
    expected=(y-mean[None,:])@rows.T@rows + mean[None,:]
    np.testing.assert_allclose(dec,expected,rtol=1e-5,atol=1e-5)


def synthetic_per(scale_mlp=0.8, scale_fno=0.9):
    out={}
    base=np.linspace(1.0,2.0,160,dtype=np.float64)
    for family,scale in [('mlp',scale_mlp),('fno',scale_fno)]:
        for seed in r.SEEDS:
            d=base*(1+0.001*seed)
            c=d*scale
            out[f'{family}_dense_seed{seed}']=d
            out[f'{family}_compact_seed{seed}']=c
    return out


def test_runner_and_independent_auditor_aggregate_match():
    per=synthetic_per()
    rr=r.aggregate_from_per_trajectory(per)
    aa=a.aggregate(per)
    assert rr==aa
    assert rr['verdict']=='PORTABILITY_PASS'
    assert rr['seed_uniform_pass'] is True
    assert abs(rr['Q_MLP']-0.8)<1e-12
    assert abs(rr['Q_FNO']-0.9)<1e-12
    assert abs(rr['Q_port']-0.9)<1e-12


def test_failure_verdict_is_not_rescued_by_other_family():
    per=synthetic_per(scale_mlp=0.8,scale_fno=1.2)
    rr=r.aggregate_from_per_trajectory(per)
    assert rr['Q_MLP']<1.05
    assert rr['Q_FNO']>1.05
    assert rr['Q_port']==rr['Q_FNO']
    assert rr['verdict']=='PORTABILITY_FAIL'


def test_unused_loader_rejects_without_marker_before_hdf5_open(tmp_path):
    fake=tmp_path/'fake.h5'
    fake.write_bytes(b'not hdf5')
    try:
        r.materialize_unused_after_marker(fake,tmp_path)
    except r.ProtocolError as e:
        assert 'marker missing' in str(e)
    else:
        raise AssertionError('UNUSED loader did not reject missing marker')
