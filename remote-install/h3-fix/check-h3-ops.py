import torch,json,infinicore as ic
from pathlib import Path
from diffusers.models.embeddings import get_timestep_embedding
from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
from infinilm.multimodal.minimax_h3 import minimax_h3_rotary_cache
report={}
def tt(x):
 out=torch.empty(x.shape,device='cuda',dtype=ic.utils.to_torch_dtype(x.dtype));ic.from_torch(out).copy_(x);return out
def compare(name,y,ref,atol=.04,rtol=.04):
 err=(y.float()-ref.float()).abs();report[name]={'max_abs':err.max().item(),'mean_abs':err.mean().item()};torch.testing.assert_close(y,ref,atol=atol,rtol=rtol);print(name,report[name],flush=True)
t=torch.tensor([0.,.1,.5,1.],device='cuda')
compare('timestep',tt(ic.timestep_embedding(ic.from_torch(t))),get_timestep_embedding(t,256,flip_sin_to_cos=True,downscale_freq_shift=0),1e-6,1e-6)
torch.manual_seed(42)
q=torch.randn(127,7,128,device='cuda',dtype=torch.bfloat16);k=torch.randn_like(q)
pos=torch.randn(127,3,device='cuda')*100
cache=minimax_h3_rotary_cache(pos);cos,sin=cache.chunk(2,-1)
actual,_=ic.mrope(ic.from_torch(q.reshape(127,-1)),ic.from_torch(k.reshape(127,-1)),ic.from_torch(cos.contiguous()),ic.from_torch(sin.contiguous()),ic.from_torch(torch.arange(127,device='cuda')),128,96,16,16,16,False)
ref=_apply_rotary_emb(q.unsqueeze(0),cos.repeat(1,2),sin.repeat(1,2)).squeeze(0)
compare('rotary',tt(actual).reshape(q.shape),ref)
Path('audit/20261102-fix/h3-ops-results.json').write_text(json.dumps(report,indent=2))
