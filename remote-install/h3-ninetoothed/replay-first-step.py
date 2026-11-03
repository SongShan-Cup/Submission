"""Replay frozen, identical H3 first-step inputs through the new native backend."""
import ctypes
import json
import os
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path('InfiniLM/examples').resolve()))
import minimax_h3 as entry
root=Path('audit/20261103-ninetoothed');baseline=Path('audit/20261102-fix')

def move(x):
 if isinstance(x,torch.Tensor):return x.to('cuda:0')
 if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return type(x)(move(v) for v in x)
 return x

def metric(a,b):
 a=a.detach().cpu().double().flatten();b=b.detach().cpu().double().flatten();e=a-b
 return {'finite':bool(torch.isfinite(a).all()),'max_abs_error':e.abs().max().item(),'relative_rmse':(e.square().mean()/b.square().mean()).sqrt().item(),'cosine':torch.nn.functional.cosine_similarity(a,b,dim=0).item()}

paths=entry.resolve_model_paths('/public/home/aclsylqidf/shared/MiniMax-H3/diffusers','fl2va')
pipe,text,transformer=entry.load_pipeline(paths,'fl2va',list(range(8)),entry.TimingReport(list(range(8))),legacy=False,legacy_text_devices=[0,1,2,3],legacy_transformer_devices=[4,5,6,7],legacy_memory_fraction=.4)
lib=ctypes.CDLL(os.environ['INFINI_ROOT']+'/lib/libh3_ninetoothed.so')
count=lib.h3_nt_launch_count;count.argtypes=[ctypes.c_int]*3;count.restype=ctypes.c_uint64
with torch.inference_mode():
 data=torch.load(baseline/'native-first-input.pt',weights_only=False)
 a=move(data['args']);kw=move(data['kwargs']);torch.cuda.synchronize()
 out=transformer(*a,**kw)
 values=out if isinstance(out,tuple) else (out.sample,out.audio_sample)
 values=tuple(x.detach().cpu() for x in values)
 torch.save(values,root/'ninetoothed-first-output.pt')
 old=torch.load(baseline/'native-first-output.pt',weights_only=False)
 report={name:metric(a,b) for name,a,b in zip(('video','audio'),values,old)}
 report['calls_by_device']={str(d):count(d,128,0) for d in range(8)}
 report['loaded_libraries']=sorted({l.split()[-1] for l in Path('/proc/self/maps').read_text().splitlines() if 'libinfinicore_cpp_api' in l or 'libh3_ninetoothed' in l})
 report['pass']=all(report[k]['finite'] and report[k]['relative_rmse']<.05 and report[k]['cosine']>.998 for k in ('video','audio')) and all(report['calls_by_device'].values())
 (root/'first-step-comparison.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2),flush=True)
 assert report['pass'],'Same-input integration acceptance failed'
