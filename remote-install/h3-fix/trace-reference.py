import sys,torch
from pathlib import Path
sys.path.insert(0,str(Path('InfiniLM/examples').resolve()))
import minimax_h3 as entry
root=Path('audit/20261102-fix')
pipe,text,model=entry.load_pipeline(entry.resolve_model_paths('/public/home/aclsylqidf/shared/MiniMax-H3/diffusers','fl2va'),'fl2va',list(range(8)),entry.TimingReport(list(range(8))),legacy=True,legacy_text_devices=[0,1,2,3],legacy_transformer_devices=[4,5,6,7],legacy_memory_fraction=.4)
print('native library loaded?',any('libinfinicore' in x for x in open('/proc/self/maps')),flush=True)
stages={}
def pre(m,a):
 stages['embeddings']=a[0].detach().cpu();stages['temb']=a[1].detach().cpu()
 print('captured embeddings',flush=True)
model.transformer_blocks[0].register_forward_pre_hook(pre)
for i in [0,1,2,9,49]:
 def hook(m,a,out,i=i):
  stages[f'block:{i}']=out.detach().cpu();print('captured',i,flush=True)
 model.transformer_blocks[i].register_forward_hook(hook)
def move(x):
 if isinstance(x,torch.Tensor):return x.to('cuda:4')
 if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return type(x)(move(v) for v in x)
 return x
data=torch.load(root/'native-first-input.pt',weights_only=False)
with torch.inference_mode():model(*move(data['args']),**move(data['kwargs']))
torch.save(stages,root/'reference-stages.pt')
print('reference stages saved',flush=True)
