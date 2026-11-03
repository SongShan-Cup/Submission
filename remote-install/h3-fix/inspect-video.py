import av,json
from pathlib import Path
from PIL import Image,ImageDraw
p=Path('outputs/20261102-fix/fl2va_50step.mp4')
with av.open(str(p)) as c:
 v=c.streams.video[0]
 info={'file':str(p),'width':v.width,'height':v.height,'frames':v.frames,'fps':str(v.average_rate),'codec':v.codec_context.name}
 frames=[]
 for i,f in enumerate(c.decode(video=0)):
  if i in [0,24,60,100]:
   im=f.to_image();im.thumbnail((384,464));frames.append((i,im.copy()))
info['decoded_frames']=i+1
assert info['width']==768 and info['height']==928 and info['frames']==124
sheet=Image.new('RGB',(384*len(frames),500),'white');d=ImageDraw.Draw(sheet)
for j,(i,im) in enumerate(frames):sheet.paste(im,(j*384,28));d.text((j*384+8,8),f'Frame {i}',fill='black')
sheet.save('outputs/20261102-fix/fl2va-contact-sheet.jpg')
Path('audit/20261102-fix/video-metadata.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
