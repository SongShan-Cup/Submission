import sys,json,torch
from pathlib import Path
sys.path.insert(0,str(Path('InfiniLM/examples').resolve()))
import minimax_h3 as entry
from infinilm.multimodal.minimax_h3 import MiniMaxH3ConditionEncoder
root=Path('audit/20261102-fix');path='/public/home/aclsylqidf/shared/MiniMax-H3/diffusers/text_encoder'
ref=entry.Qwen3VLForConditionalGeneration.from_pretrained(path,dtype=torch.bfloat16,device_map='balanced_low_0',max_memory=entry.make_max_memory([0,1,2,3],.4),local_files_only=True).eval()
native=MiniMaxH3ConditionEncoder(path,tp_device_ids=list(range(8))).eval()
inputs=torch.load(root/'native-condition-input.pt',weights_only=False)
def move(x):
 if isinstance(x,torch.Tensor):return x.to('cuda:0')
 return x
inputs={k:move(v) for k,v in inputs.items()}
mask=inputs.get('mm_token_type_ids',torch.zeros_like(inputs['input_ids']))[0]==0
ids=inputs['input_ids'][:,mask]
text={'input_ids':ids,'attention_mask':torch.ones_like(ids),'output_hidden_states':True,'use_cache':False,'mm_token_type_ids':torch.zeros_like(ids)}
report={}
with torch.inference_mode():
 for name,data in [('image',inputs),('text_only',text)]:
  a=native.model(**data).hidden_states[50].cpu().double()
  b=ref.model(**data).hidden_states[50].cpu().double()
  e=a-b;r=(e.square().mean(-1)/b.square().mean(-1)).sqrt()
  report[name]={'tokens':a.shape[1],'relative_rmse':(e.square().mean()/b.square().mean()).sqrt().item(),'cosine':torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0).item(),'token_relative_rmse_median':r.median().item(),'token_relative_rmse_max':r.max().item(),'native_max':a.abs().max().item(),'reference_max':b.abs().max().item()}
  print(name,json.dumps(report[name]),flush=True)
  (root/'condition-comparison.json').write_text(json.dumps(report,indent=2))
(root/'condition-comparison.json').write_text(json.dumps(report,indent=2))
