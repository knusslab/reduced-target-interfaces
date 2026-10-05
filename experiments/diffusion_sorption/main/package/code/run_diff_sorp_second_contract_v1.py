"""Byte/schema gates and role-limited loader for DIFF_SORP_SECOND_CONTRACT_V1.

This executor deliberately stops before basis fitting, training, selection, or TEST evaluation.
The first accepted transitions are raw-byte verification and schema-only admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from datetime import datetime, timezone

import numpy as np

from diff_sorp_second_contract_v1 import (
    ADDENDUM_SHA256, ADDENDUM_V1_2_SHA256, ADDENDUM_V1_3_SHA256, ADDENDUM_V1_4_SHA256, OFFICIAL_SOURCE_BLOBS_V1_4, DATA_FILENAME, FIELD_DIM, PINNED_PDEBENCH_COMMIT, PROTOCOL_ID,
    PROTOCOL_SHA256, PUBLISHER_MD5, SchemaNoGo, ProvenanceFail, assert_role_openable,
    frozen_splits, pair_time_indices, sha256_file, validate_schema_record,
)


class DataByteNoGo(RuntimeError):
    pass


def _md5_sha256(path, block=1<<24):
    md5=hashlib.md5(); sha=hashlib.sha256(); size=0
    with open(path,'rb') as f:
        while True:
            chunk=f.read(block)
            if not chunk: break
            size += len(chunk); md5.update(chunk); sha.update(chunk)
    return size,md5.hexdigest(),sha.hexdigest()


def raw_byte_record(path, *, expected_md5=PUBLISHER_MD5):
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(p)
    size,md5,sha=_md5_sha256(p)
    ok=(md5==str(expected_md5))
    if not ok:
        raise DataByteNoGo(f'publisher MD5 mismatch: {md5} != {expected_md5}')
    return {'bytes':int(size),'md5':md5,'sha256':sha,'publisher_md5_matches':True}


def _write_json_create_only(path,payload):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'x',encoding='utf-8') as f:
        json.dump(payload,f,indent=2,sort_keys=True); f.write('\n')
    return p


def write_raw_byte_seal(data_file, output_dir, *, expected_md5=PUBLISHER_MD5):
    fact=raw_byte_record(data_file,expected_md5=expected_md5)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    record={'protocol_id':PROTOCOL_ID,'protocol_sha256':PROTOCOL_SHA256,
            'addendum_sha256':ADDENDUM_SHA256,'addendum_v1_2_sha256':ADDENDUM_V1_2_SHA256,'addendum_v1_3_sha256':ADDENDUM_V1_3_SHA256,'addendum_v1_4_sha256':ADDENDUM_V1_4_SHA256,'status':'RAW_BYTE_SEALED',
            'data_filename':Path(data_file).name,'expected_filename':DATA_FILENAME,
            'publisher_md5':str(expected_md5),'publisher_md5_matches':True,
            'data_sha256':fact['sha256'],'bytes':fact['bytes'],'md5':fact['md5'],
            'pinned_pdebench_commit':PINNED_PDEBENCH_COMMIT,'scientific_tensor_reads':0,
            'environment':{'python':platform.python_version(),'numpy':np.__version__},
            'sealed_at_utc':datetime.now(timezone.utc).isoformat()}
    _write_json_create_only(out/'RAW_BYTE_SEAL.json',record)
    return record


def schema_record_from_handle(handle):
    names=sorted(str(k) for k in handle.keys())
    expected=[f'{i:04d}' for i in range(10000)]
    if names != expected:
        # Let the central validator produce the terminal classification.
        return {'group_names':names,'data_shape':[],'x_shape':[],'t_shape':[],
                'data_dtype':'','x_dtype':'','t_dtype':'','scientific_tensor_reads':0}
    first=None
    for name in names:
        group=handle[name]
        keys=set(str(k) for k in group.keys())
        if keys != {'data','grid'}:
            raise SchemaNoGo(f'{name}: top-level datasets differ: {sorted(keys)}')
        grid=group['grid']
        if set(str(k) for k in grid.keys()) != {'x','t'}:
            raise SchemaNoGo(f'{name}: grid datasets differ')
        current=(list(group['data'].shape),list(grid['x'].shape),list(grid['t'].shape),
                 str(group['data'].dtype),str(grid['x'].dtype),str(grid['t'].dtype))
        if first is None:
            first=current
        elif current != first:
            raise SchemaNoGo(f'{name}: schema differs across groups')
    data_shape,x_shape,t_shape,data_dtype,x_dtype,t_dtype=first
    return {'group_names':names,'data_shape':data_shape,'x_shape':x_shape,'t_shape':t_shape,
            'data_dtype':data_dtype,'x_dtype':x_dtype,'t_dtype':t_dtype,
            'scientific_tensor_reads':0}


def open_h5_schema(data_file):
    import h5py
    with h5py.File(data_file,'r') as handle:
        return schema_record_from_handle(handle)


def validate_schema_before_science(record):
    return validate_schema_record(record)


def _validate_raw_seal(raw_seal):
    if raw_seal.get('status')!='RAW_BYTE_SEALED': raise ProvenanceFail('raw byte seal not ready')
    if raw_seal.get('protocol_sha256')!=PROTOCOL_SHA256: raise ProvenanceFail('raw seal protocol mismatch')
    if raw_seal.get('addendum_sha256')!=ADDENDUM_SHA256: raise ProvenanceFail('raw seal addendum mismatch')
    if raw_seal.get('addendum_v1_2_sha256')!=ADDENDUM_V1_2_SHA256: raise ProvenanceFail('raw seal V1.2 addendum mismatch')
    if raw_seal.get('addendum_v1_3_sha256')!=ADDENDUM_V1_3_SHA256: raise ProvenanceFail('raw seal V1.3 addendum mismatch')
    if raw_seal.get('addendum_v1_4_sha256')!=ADDENDUM_V1_4_SHA256: raise ProvenanceFail('raw seal V1.4 addendum mismatch')
    if raw_seal.get('publisher_md5_matches') is not True: raise ProvenanceFail('publisher MD5 not verified')
    value=raw_seal.get('data_sha256')
    if not isinstance(value,str) or len(value)!=64: raise ProvenanceFail('raw seal data SHA missing')
    return True


def write_schema_seal(data_file, raw_seal, output_dir):
    _validate_raw_seal(raw_seal)  # before HDF5 is opened
    record=open_h5_schema(data_file)
    validate_schema_before_science(record)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    payload={'protocol_id':PROTOCOL_ID,'protocol_sha256':PROTOCOL_SHA256,
             'addendum_sha256':ADDENDUM_SHA256,'addendum_v1_2_sha256':ADDENDUM_V1_2_SHA256,'addendum_v1_3_sha256':ADDENDUM_V1_3_SHA256,'addendum_v1_4_sha256':ADDENDUM_V1_4_SHA256,'status':'SCHEMA_ADMITTED',
             'official_source_blobs_v1_4':dict(OFFICIAL_SOURCE_BLOBS_V1_4),
             'binds_to_data_sha256':raw_seal['data_sha256'],
             'schema':{k:v for k,v in record.items() if k!='scientific_tensor_reads'},
             'scientific_tensor_reads':0,
             'sealed_at_utc':datetime.now(timezone.utc).isoformat()}
    _write_json_create_only(out/'SCHEMA_SEAL.json',payload)
    return payload


def pair_rows_from_trajectory(array, *, field_dim=FIELD_DIM):
    arr=np.asarray(array,dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] < 101 or arr.shape[1] != int(field_dim) or arr.shape[2] != 1:
        raise ValueError(f'trajectory shape {arr.shape} incompatible with [>=101,{field_dim},1]')
    idx=pair_time_indices()
    x=np.ascontiguousarray(arr[idx,:,0],dtype=np.float32)
    y=np.ascontiguousarray(arr[idx+1,:,0],dtype=np.float32)
    return x,y


def _open_h5_for_science(path):
    import h5py
    return h5py.File(path,'r')


def load_role_pairs(data_file, role, *, stage, field_dim=FIELD_DIM):
    # This must happen before any HDF5 open.
    assert_role_openable(role,stage)
    keys=frozen_splits()[role]
    idx=pair_time_indices()
    needed=np.sort(np.unique(np.concatenate([idx,idx+1]))).astype(np.int64)
    xs=[]; ys=[]; groups=[]
    with _open_h5_for_science(data_file) as handle:
        for trajectory_id,key in enumerate(keys):
            ds=handle[key]['data']
            # h5py fancy time indexing is increasing; only the 40 required frames are read.
            block=np.asarray(ds[needed,:,:],dtype=np.float32)
            positions={int(t):i for i,t in enumerate(needed.tolist())}
            selected=np.empty((101,int(field_dim),1),dtype=np.float32)
            # Fill only times used by pair extraction. Other rows are never observed downstream.
            selected.fill(np.nan)
            for t in needed:
                selected[int(t)]=block[positions[int(t)]]
            x,y=pair_rows_from_trajectory(selected,field_dim=field_dim)
            xs.append(x); ys.append(y); groups.extend([trajectory_id]*len(x))
    return np.concatenate(xs,axis=0),np.concatenate(ys,axis=0),np.asarray(groups,dtype=np.int64)


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='stage',required=True)
    b=sub.add_parser('bytes'); b.add_argument('--data-file',required=True); b.add_argument('--output-dir',required=True)
    s=sub.add_parser('schema'); s.add_argument('--data-file',required=True); s.add_argument('--raw-seal',required=True); s.add_argument('--output-dir',required=True)
    args=p.parse_args()
    if args.stage=='bytes':
        out=write_raw_byte_seal(args.data_file,args.output_dir)
    else:
        raw=json.loads(Path(args.raw_seal).read_text(encoding='utf-8'))
        out=write_schema_seal(args.data_file,raw,args.output_dir)
    print(json.dumps(out,indent=2,sort_keys=True))


if __name__=='__main__': main()
