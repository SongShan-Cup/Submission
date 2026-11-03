import os,json,torch
from pathlib import Path
from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner,MiniMaxH3TransformerInputs
root=Path('audit/20261102-fix');data=torch.load(root/'native-first-input.pt',weights_only=False)['kwargs'];ref=torch.load(root/'reference-stages.pt',weights_only=False)
data['video_hidden_states']=data.pop('hidden_states');data={k:v.to('cuda:0') for k,v in data.items() if k in MiniMaxH3TransformerInputs.__dataclass_fields__}
inputs=MiniMaxH3TransformerInputs(**data)
runner=MiniMaxH3TransformerRunner('/public/home/aclsylqidf/shared/MiniMax-H3/diffusers/transformer',tp_device_ids=list(range(8)))
def metrics(a,b):
 a=a.cpu().double();b=b.cpu().double();e=a-b
 return {'max_abs':e.abs().max().item(),'rel_rmse':(e.square().mean()/b.square().mean()).sqrt().item(),'cosine':torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0).item()}
report={}
for stage in ['embeddings','block:0','block:1','block:2','block:9','block:49']:
 os.environ['MINIMAX_H3_TRACE_STAGE']=stage
 for d in range(8):torch.cuda.synchronize(d)
 with torch.inference_mode():out=runner(inputs)
 report[stage]=metrics(out.sample,ref[stage]);report[stage]['temb']=metrics(out.audio_sample,ref['temb'])
 for name in ['video','audio','text']:
  idx=data[name+'_indices'].cpu();report[stage][name]=metrics(out.sample.cpu().index_select(1,idx),ref[stage].index_select(1,idx))
 print(stage,json.dumps(report[stage]),flush=True)
 (root/'layer-comparison.json').write_text(json.dumps(report,indent=2))
