from PIL import Image
import os
import os.path
import string
imgdirs = ["./img/1/","./img/4/"]
#"./img/0/","./img/1/","./img/2/","./img/3/","./img/4/"
# 对图片进行翻转处理
j=0
for dir in imgdirs:
    i=0
    for filenames in os.listdir(dir):  
        currentPath = os.path.join(dir,filenames)
        im = Image.open(currentPath)
        out = im.transpose(Image.FLIP_TOP_BOTTOM)
        # out = im.transpose(Image.FLIP_LEFT_RIGHT)   #水平翻转
        newname = dir + "+"+str(i) + ".bmp"
        out.save(newname)
        i=i+1
    j=j+1

