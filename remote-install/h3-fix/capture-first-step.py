"""Run the example unchanged, recording first-step tensors for numerical QA."""
import sys,functools
from pathlib import Path
import torch
sys.path.insert(0,str(Path('InfiniLM/examples').resolve()))
import minimax_h3 as entry
root=Path('audit/20261102-fix')
def cpu(x):
 if isinstance(x,torch.Tensor): return x.detach().cpu()
 if isinstance(x,dict): return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)): return type(x)(cpu(v) for v in x)
 return x
load=entry.load_pipeline
def instrument(*args,**kwargs):
 pipe,text,transformer=load(*args,**kwargs)
 forward=transformer.forward
 seen=False
 @functools.wraps(forward)
 def capture(*a,**kw):
  nonlocal seen
  if seen:return forward(*a,**kw)
  seen=True
  torch.save(cpu({'args':a,'kwargs':kw}),root/'native-first-input.pt')
  out=forward(*a,**kw)
  values=out if isinstance(out,tuple) else (out.sample,out.audio_sample)
  torch.save(cpu(values),root/'native-first-output.pt')
  print('[audit] First transformer input/output saved',flush=True)
  return out
 transformer.forward=capture
 model_forward=text.model.forward
 @functools.wraps(model_forward)
 def capture_condition(**kw):
  torch.save(cpu(kw),root/'native-condition-input.pt')
  out=model_forward(**kw)
  torch.save(cpu(out.hidden_states[50]),root/'native-condition-output.pt')
  return out
 text.model.forward=capture_condition
 return pipe,text,transformer
entry.load_pipeline=instrument
entry.main()
