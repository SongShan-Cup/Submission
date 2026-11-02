"""One causal Transformer split across eight devices, with real backprop.

This is a model-parallel acceptance workload, not DDP or eight unrelated jobs.
Use only in an exclusive whole-node allocation. No download or installation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, device, width=64, heads=4):
        super().__init__()
        self.heads=heads;self.width=width
        self.norm1=nn.LayerNorm(width,device=device)
        self.qkv=nn.Linear(width,3*width,device=device)
        self.proj=nn.Linear(width,width,device=device)
        self.norm2=nn.LayerNorm(width,device=device)
        self.ff=nn.Sequential(nn.Linear(width,2*width,device=device),nn.GELU(),nn.Linear(2*width,width,device=device))

    def forward(self,x):
        b,t,d=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(b,t,3,self.heads,d//self.heads).unbind(2)
        q,k,v=(a.transpose(1,2) for a in (q,k,v))
        scores=(q@k.transpose(-1,-2))/(d//self.heads)**0.5
        causal=torch.ones(t,t,device=x.device,dtype=torch.bool).triu(1)
        scores=scores.masked_fill(causal,float('-inf'))
        h=(scores.softmax(-1)@v).transpose(1,2).contiguous().reshape(b,t,d)
        x=x+self.proj(h)
        return x+self.ff(self.norm2(x))


class Model(nn.Module):
    def __init__(self,devices,vocab=32,width=64):
        super().__init__();self.devices=devices
        self.embedding=nn.Embedding(vocab,width,device=devices[0])
        self.blocks=nn.ModuleList(Block(d,width) for d in devices)
        self.head=nn.Linear(width,vocab,device=devices[-1])

    def forward(self,tokens):
        x=self.embedding(tokens.to(self.devices[0]))
        for device,block in zip(self.devices,self.blocks):x=block(x.to(device))
        return self.head(x)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=12)
    p.add_argument('--checkpoint',default='transformer8.pt')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--infer',action='store_true')
    p.add_argument('--cpu-test',action='store_true',help='CPU correctness only; never accelerator acceptance')
    a=p.parse_args()
    if not 1<=a.steps<=1000:p.error('steps must be 1..1000')
    torch.set_num_threads(2);torch.manual_seed(20261023)
    if a.cpu_test:devices=['cpu']*8
    else:
        if not torch.cuda.is_available() or torch.cuda.device_count()!=8:
            raise RuntimeError('EXCLUSIVE_EIGHT_DEVICE_ALLOCATION_REQUIRED')
        devices=['cuda:'+str(i) for i in range(8)]
    started=time.monotonic();model=Model(devices)
    checkpoint=Path(a.checkpoint);previous_steps=0
    if a.resume or a.infer:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(saved['model']);previous_steps=int(saved['steps'])
    x=(torch.arange(64).reshape(4,16)%32).long()
    y=((x+1)%32).to(devices[-1])
    rows=[];changed=[]
    if not a.infer:
        initial=[next(b.parameters()).detach().cpu().clone() for b in model.blocks]
        optimizer=torch.optim.SGD(model.parameters(),lr=0.03,foreach=False)
        model.train()
        for step in range(a.steps):
            optimizer.zero_grad(set_to_none=True)
            logits=model(x);loss=F.cross_entropy(logits.reshape(-1,32),y.reshape(-1))
            loss.backward()
            norms=[]
            for block in model.blocks:
                norm=sum(float(p.grad.detach().abs().sum().item()) for p in block.parameters() if p.grad is not None)
                if not norm>0:raise RuntimeError('A_MODEL_STAGE_HAS_NO_GRADIENT')
                norms.append(norm)
            optimizer.step()
            row=dict(step=previous_steps+step+1,loss=float(loss.item()),stage_gradient_l1=norms)
            rows.append(row);print(json.dumps(row),flush=True)
        changed=[not torch.equal(before,next(block.parameters()).detach().cpu()) for before,block in zip(initial,model.blocks)]
        if not all(changed):raise RuntimeError('A_MODEL_STAGE_DID_NOT_TRAIN')
        if len(rows)>1 and not rows[-1]['loss']<rows[0]['loss']:raise RuntimeError('LOSS_DID_NOT_DECREASE')
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},steps=previous_steps+a.steps),checkpoint)
    model.eval()
    with torch.no_grad():prediction=model(x).cpu()
    reference=Model(['cpu']*8);reference.load_state_dict({k:v.detach().cpu() for k,v in model.state_dict().items()});reference.eval()
    with torch.no_grad():expected=reference(x)
    torch.testing.assert_close(prediction,expected,rtol=1e-3,atol=1e-4)
    result=dict(scope='CPU correctness only' if a.cpu_test else 'Single-process eight-device model parallel; not DDP',
        devices=devices,stages=8,mode='inference' if a.infer else 'resume' if a.resume else 'train',
        trained_stages=changed,steps=previous_steps+(0 if a.infer else a.steps),rows=rows,
        max_cpu_reference_error=float((prediction-expected).abs().max().item()),elapsed_seconds=time.monotonic()-started,
        checkpoint_bytes=checkpoint.stat().st_size,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        predictions=prediction.argmax(-1).tolist())
    Path('transformer8-result.json').write_text(json.dumps(result,indent=2))
    Path('transformer8-'+result['mode']+'-result.json').write_text(json.dumps(result,indent=2))
    print('EIGHT_DEVICE_MODEL_OK',json.dumps({k:v for k,v in result.items() if k not in ('rows','predictions')}),flush=True)


if __name__=='__main__':main()
