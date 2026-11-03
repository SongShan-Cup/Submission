"""Compare native first-step tensors against the same installed reference model."""
import sys,json
from pathlib import Path
import torch
sys.path.insert(0,str(Path('InfiniLM/examples').resolve()))
import minimax_h3 as entry
root=Path('audit/20261102-fix')
def move(x,device):
 if isinstance(x,torch.Tensor):return x.to(device)
 if isinstance(x,dict):return {k:move(v,device) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return type(x)(move(v,device) for v in x)
 return x
def metric(actual,reference):
 a=actual.cpu().double().flatten();b=reference.cpu().double().flatten();e=a-b
 return {'shape':list(actual.shape),'native_finite':bool(torch.isfinite(a).all()),'reference_finite':bool(torch.isfinite(b).all()),'max_abs_error':e.abs().max().item(),'rmse':e.square().mean().sqrt().item(),'relative_rmse':(e.square().mean()/b.square().mean()).sqrt().item(),'cosine':torch.nn.functional.cosine_similarity(a,b,dim=0).item()}
paths=entry.resolve_model_paths('/public/home/aclsylqidf/shared/MiniMax-H3/diffusers','fl2va')
pipe,text,transformer=entry.load_pipeline(paths,'fl2va',list(range(8)),entry.TimingReport(list(range(8))),legacy=True,legacy_text_devices=[0,1,2,3],legacy_transformer_devices=[4,5,6,7],legacy_memory_fraction=.4)
text.eval();transformer.eval()
report={}
with torch.inference_mode():
 data=torch.load(root/'native-first-input.pt',weights_only=False)
 out=transformer(*move(data['args'],'cuda:4'),**move(data['kwargs'],'cuda:4'))
 values=out if isinstance(out,tuple) else (out.sample,out.audio_sample)
 native=torch.load(root/'native-first-output.pt',weights_only=False)
 for name,a,b in zip(('video_same_input','audio_same_input'),native,values):report[name]=metric(a,b)
 print(json.dumps(report,indent=2),flush=True)
 inputs=torch.load(root/'native-condition-input.pt',weights_only=False)
 ref=text.model(**move(inputs,'cuda:0')).hidden_states[50]
 native_cond=torch.load(root/'native-condition-output.pt',weights_only=False)
 torch.save(ref.cpu(),root/'reference-condition-output.pt')
 report['condition_encoder']=metric(native_cond,ref)
 print(json.dumps(report,indent=2),flush=True)
(root/'first-step-comparison.json').write_text(json.dumps(report,indent=2))
