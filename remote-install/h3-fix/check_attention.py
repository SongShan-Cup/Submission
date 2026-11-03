import json,math,torch,infinicore as ic
from pathlib import Path
from flash_attn import flash_attn_func
report={}
def to_torch(x):
 out=torch.empty(x.shape,device='cuda',dtype=ic.utils.to_torch_dtype(x.dtype))
 ic.from_torch(out).copy_(x)
 return out
torch.manual_seed(42)
for name,b,s,h,hk,causal,d in [('noncausal',1,127,7,7,False,128),('causal',2,64,8,8,True,128),('gqa',1,63,8,2,False,128),('supplemental_head96',1,127,7,7,False,96),('vision_head72',1,127,16,16,False,72),('self_attention128',1,127,7,7,False,128),('self_attention72',1,127,16,16,False,72)]:
 q=torch.randn(b,s,h,d,device='cuda',dtype=torch.bfloat16)
 k=torch.randn(b,s,hk,d,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
 if name.startswith('self_attention'): k=q;v=q
 out=ic.mha(ic.from_torch(q),ic.from_torch(k),ic.from_torch(v),scale=d**-.5,is_causal=causal)
 actual=to_torch(out)
 expected=torch.nn.functional.scaled_dot_product_attention(q.float().transpose(1,2),k.float().repeat_interleave(h//hk,2).transpose(1,2),v.float().repeat_interleave(h//hk,2).transpose(1,2),is_causal=causal).transpose(1,2)
 err=(actual.float()-expected).abs().max().item();torch.testing.assert_close(actual.float(),expected,atol=.025,rtol=.025)
 report[name]={'max_abs_error':err,'pass':True}
 print(name,report[name],flush=True)
 # Exercise a caller-supplied noncontiguous destination.
 storage=torch.empty(b,s,h,d*2,device='cuda',dtype=torch.bfloat16)
 dest=storage[...,::2]
 dest_ic=ic.from_torch(storage).as_strided(list(dest.shape),list(dest.stride()))
 ic.mha(ic.from_torch(q),ic.from_torch(k),ic.from_torch(v),scale=d**-.5,is_causal=causal,out=dest_ic)
 torch.testing.assert_close(dest,actual,atol=.01,rtol=.01)
torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
q=torch.randn(1,32768,7,128,device='cuda',dtype=torch.bfloat16)
k=torch.randn_like(q);v=torch.randn_like(q)
y=to_torch(ic.mha(ic.from_torch(q),ic.from_torch(k),ic.from_torch(v),scale=128**-.5));torch.cuda.synchronize()
assert torch.isfinite(y).all()
peak=torch.cuda.max_memory_allocated()/2**20
idx=torch.tensor([0,1,127,8191,16384,32767],device='cuda')
ref=torch.nn.functional.scaled_dot_product_attention(q[:,idx].float().transpose(1,2),k.float().transpose(1,2),v.float().transpose(1,2)).transpose(1,2)
torch.testing.assert_close(y[:,idx].float(),ref,atol=.025,rtol=.025)
report['long_sequence']={'sequence':32768,'heads':7,'peak_torch_memory_MiB':peak,'finite':True,'sampled_max_abs_error':(y[:,idx].float()-ref).abs().max().item(),'pass':True}
print(report['long_sequence'],flush=True)
Path('audit/20261102-fix/attention-results.json').write_text(json.dumps(report,indent=2))
