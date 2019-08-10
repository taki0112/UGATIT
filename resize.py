from PIL import Image
import os
import sys

paths = [
    "dataset/selfie2anime/testA/"
    "dataset/selfie2anime/testB/"
    "dataset/selfie2anime/trainA/"
    "dataset/selfie2anime/trainB/"
]


def resize():
    for path in paths:
        dirs = os.listdir(path)
        for item in dirs:
            if os.path.isfile(path+item):
                im = Image.open(path+item)
                f, e = os.path.splitext(path+item)
                imResize = im.resize((256, 256), Image.ANTIALIAS)
                imResize.save(f + '.jpg', 'JPEG', quality=90)


resize()
