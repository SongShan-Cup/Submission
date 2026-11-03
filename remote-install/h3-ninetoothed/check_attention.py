"""Acceptance through the real InfiniCore MHA dispatcher, with FP32 oracles."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import time
import torch
import infinicore as ic

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=Path('audit/20261103-ninetoothed/attention-results.json'))
args=parser.parse_args()
libpath=Path(os.environ['INFINI_ROOT'])/'lib/libh3_ninetoothed.so'
lib=ctypes.CDLL(str(libpath))
count=lib.h3_nt_launch_count;count.argtypes=[ctypes.c_int]*3;count.restype=ctypes.c_uint64
report={'library':str(libpath),'cases':[]}

def save():
 args.output.write_text(json.dumps(report,indent=2))

def wrap(t):
 # from_torch currently assumes contiguous strides. Explicitly restore view metadata.
 return ic.from_torch(t).as_strided(list(t.shape),list(t.stride()))

def reference(q,k,v,scale,causal,indices=None):
 h=q.shape[2];hk=k.shape[2];sq=q.shape[1];sk=k.shape[1]
 query=q if indices is None else q[:,indices]
 mask=None
 if causal:
  positions=torch.arange(sq,device=q.device) if indices is None else indices
  mask=positions[:,None]+sk-sq>=torch.arange(sk,device=q.device)[None,:]
 return torch.nn.functional.scaled_dot_product_attention(query.float().transpose(1,2),k.float().repeat_interleave(h//hk,2).transpose(1,2),v.float().repeat_interleave(h//hk,2).transpose(1,2),attn_mask=mask,scale=scale).transpose(1,2)

def check(name,b,sq,sk,h,hk,d,causal,dtype,scale=None,strided=False,alias=False,device=0):
 torch.cuda.set_device(device);ic.set_device(ic.device('cuda',device));torch.manual_seed(42)
 scale=d**-.5 if scale is None else scale
 q=torch.randn(b,sq,h,d*(2 if strided else 1),device='cuda',dtype=dtype)
 k=torch.randn(b,sk,hk,d*(2 if strided else 1),device='cuda',dtype=dtype);v=torch.randn_like(k)
 if strided:q=q[...,::2];k=k[...,::2];v=v[...,::2]
 if alias:k=q;v=q
 dest_storage=torch.empty(b,sq,h,d*2,device='cuda',dtype=dtype);dest=dest_storage[...,::2]
 torch.cuda.synchronize()
 before=count(device,d,int(causal))
 ic.mha(wrap(q),wrap(k),wrap(v),scale=scale,is_causal=causal,out=wrap(dest))
 ic.sync_stream();torch.cuda.synchronize()
 assert count(device,d,int(causal))==before+1,'MHA did not execute NineToothed exactly once'
 ref=reference(q,k,v,scale,causal)
 atol,rtol=(.025,.025) if dtype==torch.bfloat16 else (.004,.01)
 error=(dest.float()-ref).abs().max().item()
 torch.testing.assert_close(dest.float(),ref,atol=atol,rtol=rtol)
 item={'name':name,'dtype':str(dtype),'shape':[b,sq,sk,h,hk,d],'causal':causal,'device':device,'max_abs_error':error,'atol':atol,'rtol':rtol,'pass':True}
 report['cases'].append(item);save();print(json.dumps(item),flush=True)

for dtype in (torch.bfloat16,torch.float16):
 for d in (72,96,128):
  for causal in (False,True):check('tail_and_strided_output',2,127,127,4,4,d,causal,dtype)
 check('gqa_rectangular',1,17,29,8,2,128,True,dtype)
 check('fully_masked_causal_rows',1,29,17,8,2,128,True,dtype)
 check('noncontiguous_input_custom_scale',1,63,97,8,2,72,False,dtype,scale=.17,strided=True)
 check('q_equals_k_equals_v',1,127,127,7,7,128,False,dtype,alias=True)
 check('q_equals_k_equals_v_vision',1,127,127,16,16,72,False,dtype,alias=True)
for device in range(1,8):check('device_context',1,31,43,2,2,128,False,torch.bfloat16,device=device)

torch.cuda.set_device(0);ic.set_device(ic.device('cuda',0))
# Fail-closed contract: ALiBi and unsupported heads never fall back to the vendor.
for name,d,alibi in [('unsupported_head',64,None),('unsupported_alibi',128,torch.ones(2,device='cuda'))]:
 q=torch.randn(1,16,2,d,device='cuda',dtype=torch.bfloat16)
 try:ic.mha(wrap(q),wrap(q),wrap(q),scale=d**-.5,alibi_slopes=None if alibi is None else ic.from_torch(alibi))
 except (RuntimeError,ValueError) as exc:
  report[name]={'pass':True,'message':str(exc).splitlines()[0]}
 else:raise AssertionError(f'{name} was not rejected')

# Same 32768 x 7 x 128 workload as the frozen baseline. Sample rows avoid an S^2 oracle.
torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
free_before,total=torch.cuda.mem_get_info()
q=torch.randn(1,32768,7,128,device='cuda',dtype=torch.bfloat16);k=torch.randn_like(q);v=torch.randn_like(q);out=torch.empty_like(q)
torch.cuda.synchronize();before=count(0,128,0);times=[]
for _ in range(3):
 start=time.perf_counter();ic.mha(wrap(q),wrap(k),wrap(v),scale=128**-.5,out=wrap(out));ic.sync_stream();torch.cuda.synchronize();times.append(time.perf_counter()-start)
assert count(0,128,0)==before+3
peak=torch.cuda.max_memory_allocated()/2**20
free_after,_=torch.cuda.mem_get_info()
indices=torch.tensor([0,1,127,8191,16384,32767],device='cuda')
ref=reference(q,k,v,128**-.5,False,indices)
torch.testing.assert_close(out[:,indices].float(),ref,atol=.025,rtol=.025)
assert torch.isfinite(out).all()
# Whole-device free-memory delta supplements (but is not a peak substitute for) the Torch allocator statistic.
report['long_sequence']={'sequence':32768,'heads':7,'pass':True,'sampled_max_abs_error':(out[:,indices].float()-ref).abs().max().item(),'torch_peak_MiB':peak,'device_free_delta_MiB':(free_before-free_after)/2**20,'seconds':times}
assert peak<2048,'Unexpected quadratic allocation'
report['loaded_libraries']=sorted({l.split()[-1] for l in Path('/proc/self/maps').read_text().splitlines() if any(x in l for x in ['libinfinicore_cpp_api','libh3_ninetoothed','libinfiniop.so'])})
assert any(str(libpath) in p for p in report['loaded_libraries'])
assert all('/h3-ninetoothed/native/' in p for p in report['loaded_libraries']), report['loaded_libraries']
report['pass']=True;save();print(json.dumps(report['long_sequence']),flush=True)
