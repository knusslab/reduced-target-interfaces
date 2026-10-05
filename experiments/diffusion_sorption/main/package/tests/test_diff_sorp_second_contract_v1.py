from pathlib import Path
import copy
import hashlib
import json
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import diff_sorp_second_contract_v1 as m


def test_protocol_hash_binding():
    p = ROOT/'protocols'/'DIFF_SORP_SECOND_CONTRACT_V1.md'
    assert m.sha256_file(p) == m.PROTOCOL_SHA256


def test_frozen_split_digests_and_disjointness():
    s = m.frozen_splits()
    assert {k: len(v) for k,v in s.items()} == {
        'train':480,'validation':160,'unused':160,'selection':200,'test':200}
    expected = {
        'train':'bdd5fe2abc0dc9706ddec253f702b9ebdd6751f69bbd5d9f5005d663cc60fcd2',
        'validation':'823b6f6a0ce08224ef14153babd0a52c54c1d2212492629fd6551d7c923e3ee6',
        'unused':'060451f6180354ffe574c8095da4bf0349a5032ea5cd8003102d77c9ac0242da',
        'selection':'d4ab1c03f1d2eb2a96baaeb1fea61ca42e73272a9f4ca5816df5198901290f44',
        'test':'38c307667103362555436ba5a02203df0dc6f60c69add785bb44e563d4db73a3',
    }
    for role, keys in s.items():
        assert m.key_digest(keys) == expected[role]
    roles=list(s)
    for i,a in enumerate(roles):
        for b in roles[i+1:]:
            assert set(s[a]).isdisjoint(s[b])
    assert all(int(k)<9000 for r in ('train','validation','unused','selection') for k in s[r])
    assert all(9000<=int(k)<10000 for k in s['test'])


def test_temporal_pair_contract():
    idx=m.pair_time_indices()
    assert idx.tolist() == list(range(0,100,5))
    assert len(idx)==20 and idx[-1]==95
    assert m.ROWS_BY_ROLE == {'train':9600,'validation':3200,'unused':3200,'selection':4000,'test':4000}


def test_role_firewall():
    assert m.assert_role_openable('train','DENSE_TRAIN')
    assert m.assert_role_openable('validation','DENSE_TRAIN')
    for role in ('selection','unused','test'):
        with pytest.raises(PermissionError): m.assert_role_openable(role,'DENSE_TRAIN')
    assert m.assert_role_openable('selection','SELECT')
    for role in ('train','validation','unused','test'):
        with pytest.raises(PermissionError): m.assert_role_openable(role,'SELECT')
    assert m.assert_role_openable('test','TEST_EVAL')
    for role in ('train','validation','selection','unused'):
        with pytest.raises(PermissionError): m.assert_role_openable(role,'TEST_EVAL')


def good_schema():
    return {
        'group_names':[f'{i:04d}' for i in range(10000)],
        'data_shape':[101,1024,1],
        'x_shape':[1024],
        't_shape':[101],
        'data_dtype':'float32','x_dtype':'float32','t_dtype':'float32'
    }


def test_schema_gate_fail_closed():
    assert m.validate_schema_record(good_schema())
    for key,val in [('data_shape',[501,1024,1]),('x_shape',[1023]),('t_shape',[100])]:
        bad=good_schema(); bad[key]=val
        with pytest.raises(m.SchemaNoGo): m.validate_schema_record(bad)
    bad=good_schema(); bad['group_names'][-1]='oops'
    with pytest.raises(m.SchemaNoGo): m.validate_schema_record(bad)


def test_basis_canonicalization_and_gram():
    rng=np.random.default_rng(3)
    y=rng.normal(size=(80,17))
    out1=m.fit_train_basis(y)
    out2=m.fit_train_basis(y)
    assert np.array_equal(out1['basis_float32'],out2['basis_float32'])
    B=out1['basis_float32']
    gram=(torch.from_numpy(B)@torch.from_numpy(B).T).numpy()
    assert m.float32_array_digest(gram)==out1['gram_sha256']
    assert m.float64_array_digest(out1['basis_float64'])==out1['basis64_sha256']
    assert m.float64_array_digest(out1['mean_float64'])==out1['mean64_sha256']
    # largest-absolute pivot is nonnegative by canonical sign rule
    for row in B:
        j=int(np.argmax(np.abs(row)))
        assert row[j]>=0


def test_realised_gram_loss_matches_lifted_field_mse():
    rng=np.random.default_rng(4)
    B=rng.normal(size=(5,19)).astype(np.float32)
    pred=torch.tensor(rng.normal(size=(7,5)),dtype=torch.float64)
    target=torch.tensor(rng.normal(size=(7,5)),dtype=torch.float64)
    Bt=torch.tensor(B,dtype=torch.float64)
    got=m.projected_field_mse_from_coefficients(pred,target,Bt)
    explicit=torch.mean((pred@Bt-target@Bt)**2)
    assert torch.allclose(got,explicit,atol=1e-12,rtol=1e-12)


def test_selector_known_case_and_seed0_only():
    gain=np.array([8.,4.,2.,1.,0.5,-100.])
    # threshold 1.6 -> keep first 3: discarded positive mass=1.5
    out=m.select_budget(gain, ladder=(1,2,3,4,6), threshold=1.6)
    assert out['selected_k']==3
    assert out['indices'].tolist()==[0,1,2]
    # negative gain is clipped, not rewarded
    assert out['tail_by_k'][4]==0.5


def test_stopping_replay_contract():
    curve=[1.0,0.9,0.8998,0.8997,0.8996]
    out=m.replay_stopping(curve,patience=3,min_delta=1e-3)
    assert out['selected_epoch']==2
    assert out['stop_epoch']==5
    assert out['stop_reason']=='PATIENCE'


def good_pretest_inputs():
    traces=[]
    for arm in ('dense','compressed'):
        for seed in m.SEEDS:
            traces.append({'arm':arm,'seed':seed,'converged':True,'stop_reason':'PATIENCE',
                'nonfinite_count':0,'selected_epoch':10,'checkpoint_file_sha256':f'f-{arm}-{seed}',
                'checkpoint_state_sha256':f's-{arm}-{seed}','batch_order_sha256':f'b-{arm}-{seed}'})
    return traces


def test_pretest_seal_rederives_and_rejects_mutation():
    selection={'proposal_seed':0,'K_prop':32,'indices_sha256':'i'*64,'gain_sha256':'g'*64,
               'selection_split_sha256':m.SPLIT_DIGESTS['selection']}
    seal=m.build_pretest_seal(good_pretest_inputs(),selection=selection,data_sha256='d'*64,
        basis_sha256='b'*64,mean_sha256='m'*64,basis64_sha256='a'*64,mean64_sha256='u'*64,gram_sha256='r'*64,
        evaluator_sha256='e'*64,bootstrap_sha256='z'*64)
    assert seal['access_counters']['test']==0 and seal['access_counters']['unused']==0
    assert m.verify_pretest_seal(seal,good_pretest_inputs(),selection=selection,data_sha256='d'*64,
        basis_sha256='b'*64,mean_sha256='m'*64,basis64_sha256='a'*64,mean64_sha256='u'*64,gram_sha256='r'*64,
        evaluator_sha256='e'*64,bootstrap_sha256='z'*64)
    bad=copy.deepcopy(seal); bad['selection']['K_prop']=64
    with pytest.raises(m.ProvenanceFail):
        m.verify_pretest_seal(bad,good_pretest_inputs(),selection=selection,data_sha256='d'*64,
            basis_sha256='b'*64,mean_sha256='m'*64,basis64_sha256='a'*64,mean64_sha256='u'*64,gram_sha256='r'*64,
            evaluator_sha256='e'*64,bootstrap_sha256='z'*64)


def test_one_shot_test_marker(tmp_path):
    p=tmp_path/'TEST_ACCESS_CONSUMED.json'
    m.OneShotMarker(p,{'protocol_sha256':m.PROTOCOL_SHA256}).open()
    assert p.is_file()
    with pytest.raises(RuntimeError): m.OneShotMarker(p,{}).open()


def test_paired_trajectory_bootstrap_deterministic():
    rng=np.random.default_rng(8)
    dense={s:rng.uniform(.8,1.2,size=20) for s in m.SEEDS}
    comp={s:dense[s]*rng.uniform(.9,1.05,size=20) for s in m.SEEDS}
    a=m.paired_trajectory_bootstrap({'dense':dense,'compressed':comp},replicates=200,seed=77)
    b=m.paired_trajectory_bootstrap({'dense':dense,'compressed':comp},replicates=200,seed=77)
    assert a==b and a['shared_resample'] is True


def test_addendum_hash_binding():
    p = ROOT/'protocols'/'DIFF_SORP_SECOND_CONTRACT_V1_1_ADDENDUM.md'
    assert m.sha256_file(p) == m.ADDENDUM_SHA256


def test_v12_addendum_hash_binding():
    p = ROOT/'protocols'/'DIFF_SORP_SECOND_CONTRACT_V1_2_ADDENDUM.md'
    assert m.sha256_file(p) == m.ADDENDUM_V1_2_SHA256


def test_schema_dtype_is_exact_float32():
    for field in ('data_dtype','x_dtype','t_dtype'):
        bad=good_schema(); bad[field]='float64'
        with pytest.raises(m.SchemaNoGo):
            m.validate_schema_record(bad)


def test_population_covariance_and_energy_rank_definition():
    y=np.array([[1.,0.],[3.,0.],[5.,2.],[7.,2.]],dtype=np.float64)
    out=m.fit_train_basis(y)
    centered=y-y.mean(axis=0)
    cov=centered.T@centered/len(y)
    evals=np.linalg.eigvalsh(cov)[::-1]
    assert np.allclose(out['eigenvalues_float64'],evals,rtol=0,atol=1e-12)
    clipped=np.clip(evals,0,None)
    expected=int(np.searchsorted(np.cumsum(clipped)/clipped.sum(),0.999)+1)
    assert out['k_energy']==expected


def test_batch_order_and_parameter_arithmetic():
    a=m.batch_permutation(100,seed=2,epoch=7)
    g=torch.Generator().manual_seed(2006)
    b=torch.randperm(100,generator=g).numpy()
    assert np.array_equal(a,b)
    dense=m.mlp_parameter_count(1024,256,1024)
    compressed=m.mlp_parameter_count(1024,256,32)
    assert dense == sum(p.numel() for p in m.MLP(1024,1024,hidden=256).parameters())
    assert compressed == sum(p.numel() for p in m.MLP(1024,32,hidden=256).parameters())
    assert compressed < dense


def test_v13_addendum_and_source_registry_binding():
    p=ROOT/'protocols'/'DIFF_SORP_SECOND_CONTRACT_V1_3_ADDENDUM.md'
    assert m.sha256_file(p)==m.ADDENDUM_V1_3_SHA256
    assert m.OFFICIAL_SOURCE_BLOBS == {
        'data_registry':'3332b679127a6db3059243983a237332273483e7',
        'generator':'2f1462b81996d71df10ec8570420e951331fbf07',
        'generator_config':'2ad7e1f3725250b75ce876c16290ffa450984af9',
        'model_config':'7ed4371bafc4922d89e3f9245d1c4e7f82a2bf92',
        'dataset_loader':'d5bc69db79639ab2dc5e2c7fa3ca286eade8ea55',
    }


def test_selector_seals_complete_order_digest():
    gain=np.array([3.,3.,1.,0.,-2.])
    out=m.select_budget(gain,ladder=(1,2,3,4,5),threshold=1.1)
    assert out['order'].tolist()==[0,1,2,3,4]
    expected=hashlib.sha256(np.asarray(out['order'],dtype=np.int64).tobytes()).hexdigest()
    assert out['order_sha256']==expected


def test_v14_released_view_schema_reconciliation_binding():
    p=ROOT/'protocols'/'DIFF_SORP_SECOND_CONTRACT_V1_4_ADDENDUM.md'
    assert m.sha256_file(p)==m.ADDENDUM_V1_4_SHA256
    assert m.OFFICIAL_SOURCE_BLOBS_V1_4['release_visualizer']=='1c75218ea977fc408434c770acfb7b218417fa25'
    assert m.OFFICIAL_SOURCE_BLOBS_V1_4['model_config']=='7ed4371bafc4922d89e3f9245d1c4e7f82a2bf92'
    rec=good_schema()
    assert m.validate_schema_record(rec)
    rec501=good_schema(); rec501['t_shape']=[501]
    assert m.validate_schema_record(rec501)
    bad=good_schema(); bad['t_shape']=[100]
    with pytest.raises(m.SchemaNoGo): m.validate_schema_record(bad)
    bad=good_schema(); bad['t_shape']=[101,1]
    with pytest.raises(m.SchemaNoGo): m.validate_schema_record(bad)
