import av,json
from pathlib import Path
from PIL import Image,ImageDraw
root=Path('outputs/20261102-fix')
files=['fl2va_legacy_50step.mp4','fl2va_50step.mp4']
report={};sheet=Image.new('RGB',(1536,1000),'white');draw=ImageDraw.Draw(sheet)
for row,name in enumerate(files):
 p=root/name
 with av.open(str(p)) as c:
  v=c.streams.video[0];report[name]={'width':v.width,'height':v.height,'frames':v.frames,'fps':str(v.average_rate)}
  slot=0
  for i,f in enumerate(c.decode(video=0)):
   if i in (0,24,60,100):
    im=f.to_image();im.thumbnail((384,464));sheet.paste(im,(384*slot,500*row+28));draw.text((384*slot+5,500*row+8),f'{"PyTorch" if row==0 else "InfiniLM"}: frame {i}',fill='black');slot+=1
  report[name]['decoded_frames']=i+1
sheet.save(root/'backend-comparison.jpg')
Path('audit/20261102-fix/comparison-metadata.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
