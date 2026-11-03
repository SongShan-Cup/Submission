"""Run the unmodified example, recording loaded libraries and actual NT launches."""
import ctypes
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time
import torch

root=Path('audit/20261103-ninetoothed')
lib=ctypes.CDLL(os.environ['INFINI_ROOT']+'/lib/libh3_ninetoothed.so')
count=lib.h3_nt_launch_count;count.argtypes=[ctypes.c_int]*3;count.restype=ctypes.c_uint64
stop=threading.Event();peaks=[0]*torch.cuda.device_count()
def monitor():
 while not stop.wait(.5):
  for d in range(len(peaks)):
   free,total=torch.cuda.mem_get_info(d);peaks[d]=max(peaks[d],total-free)
thread=threading.Thread(target=monitor,daemon=True);thread.start()
start=time.perf_counter();success=False
try:
 sys.path.insert(0,str(Path('InfiniLM/examples').resolve()))
 runpy.run_path('InfiniLM/examples/minimax_h3.py',run_name='__main__')
 success=True
finally:
 stop.set();thread.join(timeout=5)
 calls={str(d):{f'head{h}_causal{c}':count(d,h,c) for h in (72,96,128) for c in (0,1)} for d in range(len(peaks))}
 report={'completed':success,'wrapper_seconds':time.perf_counter()-start,'calls':calls,
 'device_memory_sampled_peak_MiB':[x/2**20 for x in peaks],
 'memory_note':'Whole-device used memory sampled every 0.5s, includes runtime and other processes; not an exact instantaneous peak.',
 'libraries':sorted({l.split()[-1] for l in Path('/proc/self/maps').read_text().splitlines() if any(n in l for n in ['libinfinicore_cpp_api','libh3_ninetoothed','libinfiniop.so'])})}
 (root/'inference-provenance.json').write_text(json.dumps(report,indent=2))
 print('[NineToothed acceptance]',json.dumps(report),flush=True)
