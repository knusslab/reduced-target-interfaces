from pathlib import Path
import hashlib
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import diff_sorp_second_contract_v1 as core
import run_diff_sorp_second_contract_v1 as r


def test_raw_byte_verifier_matches_known_bytes(tmp_path):
    p=tmp_path/'x.bin'; p.write_bytes(b'abc123')
    md5=hashlib.md5(b'abc123').hexdigest()
    rec=r.raw_byte_record(p, expected_md5=md5)
    assert rec['bytes']==6
    assert rec['md5']==md5
    assert rec['sha256']==hashlib.sha256(b'abc123').hexdigest()
    assert rec['publisher_md5_matches'] is True
    with pytest.raises(r.DataByteNoGo):
        r.raw_byte_record(p, expected_md5='0'*32)


def test_raw_byte_seal_is_create_only(tmp_path):
    p=tmp_path/'x.bin'; p.write_bytes(b'xyz')
    md5=hashlib.md5(b'xyz').hexdigest()
    out=r.write_raw_byte_seal(p,tmp_path/'seal',expected_md5=md5)
    assert out['status']=='RAW_BYTE_SEALED'
    assert out['scientific_tensor_reads']==0
    with pytest.raises(FileExistsError):
        r.write_raw_byte_seal(p,tmp_path/'seal',expected_md5=md5)


class FakeDataset:
    def __init__(self,shape,dtype='float32'):
        self.shape=tuple(shape); self.dtype=np.dtype(dtype)
    def __array__(self,*args,**kwargs):
        raise AssertionError('scientific tensor value read during schema gate')
    def __getitem__(self,key):
        raise AssertionError('scientific tensor value read during schema gate')


class FakeGrid:
    def __init__(self):
        self.items={'x':FakeDataset((1024,)),'t':FakeDataset((101,))}
    def __getitem__(self,k): return self.items[k]
    def keys(self): return self.items.keys()


class FakeGroup:
    def __init__(self):
        self.items={'data':FakeDataset((101,1024,1)),'grid':FakeGrid()}
    def __getitem__(self,k): return self.items[k]
    def keys(self): return self.items.keys()


class FakeHandle:
    def __init__(self): self.groups={f'{i:04d}':FakeGroup() for i in range(10000)}
    def keys(self): return self.groups.keys()
    def __getitem__(self,k): return self.groups[k]


def test_schema_record_reads_metadata_only():
    rec=r.schema_record_from_handle(FakeHandle())
    assert core.validate_schema_record(rec)
    assert rec['scientific_tensor_reads']==0


def test_schema_seal_requires_byte_seal_and_is_create_only(tmp_path,monkeypatch):
    byte={'status':'RAW_BYTE_SEALED','data_sha256':'d'*64,'publisher_md5_matches':True,
          'protocol_sha256':core.PROTOCOL_SHA256,'addendum_sha256':core.ADDENDUM_SHA256,
          'addendum_v1_2_sha256':core.ADDENDUM_V1_2_SHA256,
          'addendum_v1_3_sha256':core.ADDENDUM_V1_3_SHA256,
          'addendum_v1_4_sha256':core.ADDENDUM_V1_4_SHA256}
    monkeypatch.setattr(r,'open_h5_schema',lambda _: core_schema())
    out=r.write_schema_seal(tmp_path/'fake.h5',byte,tmp_path/'schema')
    assert out['status']=='SCHEMA_ADMITTED'
    assert out['addendum_v1_4_sha256']==core.ADDENDUM_V1_4_SHA256
    assert out['official_source_blobs_v1_4']['release_visualizer']=='1c75218ea977fc408434c770acfb7b218417fa25'
    assert out['scientific_tensor_reads']==0
    with pytest.raises(FileExistsError):
        r.write_schema_seal(tmp_path/'fake.h5',byte,tmp_path/'schema')
    bad=dict(byte); bad['status']='BAD'
    with pytest.raises(r.ProvenanceFail):
        r.write_schema_seal(tmp_path/'fake.h5',bad,tmp_path/'other')


def core_schema():
    return {'group_names':[f'{i:04d}' for i in range(10000)],
            'data_shape':[101,1024,1],'x_shape':[1024],'t_shape':[101],
            'data_dtype':'float32','x_dtype':'float32','t_dtype':'float32',
            'scientific_tensor_reads':0}


def test_forbidden_role_refused_before_hdf5_open(monkeypatch):
    called={'n':0}
    def bomb(*a,**k):
        called['n']+=1
        raise AssertionError('HDF5 opened before role firewall')
    monkeypatch.setattr(r,'_open_h5_for_science',bomb)
    with pytest.raises(PermissionError):
        r.load_role_pairs('fake.h5','test',stage='DENSE_TRAIN')
    assert called['n']==0
    with pytest.raises(PermissionError):
        r.load_role_pairs('fake.h5','unused',stage='TEST_EVAL')
    assert called['n']==0


def test_pair_extraction_exact_stride_and_shapes():
    arr=np.arange(101*8,dtype=np.float32).reshape(101,8,1)
    x,y=r.pair_rows_from_trajectory(arr,field_dim=8)
    idx=core.pair_time_indices()
    assert x.shape==(20,8) and y.shape==(20,8)
    assert np.array_equal(x,arr[idx,:,0])
    assert np.array_equal(y,arr[idx+1,:,0])


def test_schema_mutation_fails_closed_without_repair():
    rec=core_schema(); rec['data_shape']=[101,1023,1]
    with pytest.raises(core.SchemaNoGo):
        r.validate_schema_before_science(rec)
