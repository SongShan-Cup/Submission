"""Eight ranks, DDP, two DataLoader workers per rank, checkpoint + inference.

Run with: torchrun --standalone --nnodes=1 --nproc_per_node=8 torch_ddp_8card.py
Requires a separately certified whole-node communication profile.
"""
import json
import os
from pathlib import Path
import time
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader,TensorDataset,DistributedSampler


def model():return nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,8))


def main():
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=8 or torch.cuda.device_count()!=8:raise RuntimeError('EIGHT_RANK_WHOLE_NODE_REQUIRED')
    torch.set_num_threads(1);torch.cuda.set_device(local);torch.manual_seed(20261023)
    dist.init_process_group('nccl',timeout=timedelta(seconds=90));started=time.monotonic()
    try:
        device='cuda:'+str(local);net=model().to(device);before=next(net.parameters()).detach().clone()
        ddp=DistributedDataParallel(net,device_ids=[local],output_device=local)
        generator=torch.Generator().manual_seed(37)
        x=torch.randn(512,64,generator=generator);teacher=torch.randn(64,8,generator=generator)*0.1;y=x@teacher
        dataset=TensorDataset(x,y);sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,shuffle=False)
        loader=DataLoader(dataset,batch_size=8,sampler=sampler,num_workers=2,multiprocessing_context='spawn',persistent_workers=False)
        optimizer=torch.optim.SGD(ddp.parameters(),lr=0.08);losses=[]
        for epoch in range(3):
            sampler.set_epoch(epoch);total=torch.zeros(2,device=device)
            for inputs,targets in loader:
                optimizer.zero_grad(set_to_none=True)
                output=ddp(inputs.to(device));loss=(output-targets.to(device)).square().mean()
                loss.backward();optimizer.step();total[0]+=loss.detach();total[1]+=1
            dist.all_reduce(total);losses.append(float((total[0]/total[1]).item()))
            if rank==0:print(json.dumps(dict(epoch=epoch+1,loss=losses[-1],ranks=world,loader_workers_per_rank=2)),flush=True)
        if not losses[-1]<losses[0]:raise RuntimeError('DDP_LOSS_NOT_DECREASING')
        if torch.equal(before,next(net.parameters()).detach()):raise RuntimeError('DDP_WEIGHTS_UNCHANGED')
        checksum=torch.stack([p.detach().float().sum() for p in net.parameters()]).sum().reshape(1)
        checksums=[torch.empty_like(checksum) for _ in range(world)];dist.all_gather(checksums,checksum)
        if max(abs(float(c.item()-checksums[0].item())) for c in checksums)>1e-3:raise RuntimeError('DDP_RANK_WEIGHTS_DIVERGED')
        if rank==0:torch.save({'model':{k:v.detach().cpu() for k,v in net.state_dict().items()},'epochs':3},'ddp8.pt')
        dist.barrier()
        saved=torch.load('ddp8.pt',map_location='cpu',weights_only=True);restored=model();restored.load_state_dict(saved['model']);restored.eval()
        net.eval()
        with torch.no_grad():
            expected=restored(x[:4]);actual=net(x[:4].to(device)).cpu()
        torch.testing.assert_close(actual,expected,rtol=1e-3,atol=1e-4)
        error=torch.tensor([(actual-expected).abs().max().item()],device=device);dist.all_reduce(error,op=dist.ReduceOp.MAX)
        if rank==0:
            result=dict(world_size=world,devices=list(range(8)),loader_workers_per_rank=2,losses=losses,
                max_cpu_reference_error=float(error.item()),elapsed_seconds=time.monotonic()-started,
                checkpoint_bytes=Path('ddp8.pt').stat().st_size,rank_weight_checksums=[float(t.item()) for t in checksums])
            Path('ddp8-result.json').write_text(json.dumps(result,indent=2));print('DDP_EIGHT_CARD_OK',json.dumps(result),flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
