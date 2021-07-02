import os
from PIL import Image

k = os.listdir('./Cropped_2/')
for f in k:
    #print(f)
    a = Image.open('./Cropped_2/'+f)
    output = Image.new('RGB', (a.size[0] * 2, a.size[1] * 2))

    w = a.size[0]
    h = a.size[1]

    output.paste(a, (0,0,w,h))
    output.paste(a, (w,0,w+w,h))
    output.paste(a, (0,h,w,h+h))
    output.paste(a, (w,h,w+w,h+h))

    output.resize(a.size)
    output.save('./Cropped_2/'+f)