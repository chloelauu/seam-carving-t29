from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
out=Path(__file__).resolve().parents[1]/'assets'/'example.png'
w,h=64,48
img=Image.new('RGB',(w,h))
p=np.array(img)
for y in range(h):
  if y<30:
    p[y,:,:]=(np.array([83,172,220])*(1-y/140)+np.array([230,224,181])*(y/140)).astype(np.uint8)
  else:
    p[y,:,:]=np.array([48+((y-30)*3)%20,130-((y-30)//2),100])
img=Image.fromarray(p)
d=ImageDraw.Draw(img)
d.ellipse((7,5,17,15),fill=(251,229,146))
d.polygon([(0,31),(15,21),(27,32)],fill=(79,135,120))
d.polygon([(19,30),(34,18),(53,31)],fill=(63,117,119))
d.rectangle((25,11,44,33),fill=(211,195,160),outline=(64,54,70),width=1)
d.rectangle((26,4,30,32),fill=(178,156,127),outline=(72,62,73))
d.rectangle((39,5,43,32),fill=(187,170,139),outline=(72,62,73))
d.rectangle((33,23,38,33),fill=(66,83,87),outline=(65,58,69))
d.rectangle((33,13,37,18),fill=(86,114,133),outline=(45,56,71))
d.line((0,34,63,35),fill=(37,107,89),width=2)
d.line((8,37,9,45),fill=(30,48,53),width=1)
d.ellipse((7,35,10,38),fill=(224,171,121))
img.save(out)
print(out)
