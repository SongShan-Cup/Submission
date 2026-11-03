"""Reuse the repository's comparison method unchanged; also make a contact sheet."""
import json
import sys
from pathlib import Path
import av
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path('InfiniLM/examples').resolve()))
from minimax_h3_compare import compare_video_streams
root=Path('outputs/20261103-ninetoothed');new=root/'fl2va_50step.mp4'
baseline=Path('outputs/20261102-fix/fl2va_50step.mp4')
reference=Path('outputs/20261102-fix/fl2va_legacy_50step.mp4')
report={'against_native_baseline':compare_video_streams(new,baseline),
        'against_pytorch_reference':compare_video_streams(new,reference)}
sheet=Image.new('RGB',(1280,1260),'white');draw=ImageDraw.Draw(sheet)
for row,(label,path) in enumerate([('PyTorch reference',reference),('Native baseline',baseline),('NineToothed',new)]):
 with av.open(str(path)) as c:
  slot=0
  for i,f in enumerate(c.decode(video=0)):
   if i in (0,24,60,100):
    im=f.to_image();im.thumbnail((320,387));sheet.paste(im,(320*slot,420*row+28))
    draw.text((320*slot+4,420*row+8),f'{label}: frame {i}',fill='black');slot+=1
 with av.open(str(path)) as c:
  audio=c.streams.audio[0];samples=[]
  for f in c.decode(audio=0):samples.append(f.to_ndarray())
  data=np.concatenate(samples,axis=-1)
  report[label+'_audio']={'rate':audio.rate,'channels':audio.channels,'samples':data.shape[-1],'finite':bool(np.isfinite(data).all()),'peak':float(np.max(np.abs(data))),'rms':float(np.sqrt(np.mean(data.astype(np.float64)**2)))}
sheet.save(root/'acceptance-comparison.jpg')
report['specification_pass']=all(r['metadata_size_match'] and r['decoded_length_match'] and r['left_frames']==124 and r['left_metadata']['average_fps']==24 for k,r in report.items() if k.startswith('against_'))
Path('audit/20261103-ninetoothed/video-comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
assert report['specification_pass']
