"""Small ordinary PyTorch train/infer project; no downloads or platform API.

python3 gpu_quickstart.py --steps 12
python3 gpu_quickstart.py --steps 16 --resume checkpoint.pt
python3 gpu_quickstart.py --infer --resume checkpoint.pt

An optional settings.json containing {"lr": 0.02} is reread between steps.
Changing Python source takes effect when the program is restarted.
"""
import argparse
import json
from pathlib import Path
import time


def main():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=12)
    p.add_argument('--resume')
    p.add_argument('--infer', action='store_true')
    p.add_argument('--delay', type=float, default=0)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if not 1 <= a.steps <= 10000 or not 0 <= a.delay <= 10:
        p.error('steps must be 1..10000 and delay 0..10 seconds')
    torch.set_num_threads(2)
    torch.manual_seed(17)
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('No GPU: open a certified GPU development instance first')
    model = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2)).to(a.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    first = 0
    if a.resume:
        saved = torch.load(a.resume, map_location=a.device, weights_only=True)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        first = saved['step']
    x = torch.linspace(-1, 1, 128).reshape(32, 4)
    y = torch.stack((x.sum(1), x[:, 0] - x[:, 3]), dim=1)
    if a.infer:
        model.eval()
        with torch.no_grad():
            result = model(x[:2].to(a.device)).cpu().tolist()
        Path('predictions.json').write_text(json.dumps(result), encoding='utf-8')
        print(json.dumps(dict(mode='inference', restored_step=first, output='predictions.json', values=result)), flush=True)
        return
    # num_workers=0 is deliberate: this sample works within the current small
    # process budget and does not imply multi-process GPU/DDP certification.
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False, num_workers=0)
    for step in range(first, a.steps):
        settings = Path('settings.json')
        if settings.exists():
            lr = float(json.loads(settings.read_text(encoding='utf-8'))['lr'])
            if not 0 < lr <= 1:
                raise ValueError('settings.json lr must be in (0, 1]')
            for group in optimizer.param_groups:
                group['lr'] = lr
        losses = []
        for inputs, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = (model(inputs.to(a.device)) - target.to(a.device)).square().mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        torch.save(dict(step=step + 1, model=model.state_dict(), optimizer=optimizer.state_dict()), 'checkpoint.pt')
        print(json.dumps(dict(step=step + 1, loss=sum(losses) / len(losses), lr=optimizer.param_groups[0]['lr'], checkpoint='checkpoint.pt')), flush=True)
        time.sleep(a.delay)
    print(json.dumps(dict(mode='training', device=a.device, checkpoint_bytes=Path('checkpoint.pt').stat().st_size)), flush=True)


if __name__ == '__main__':
    main()