import cv2
import numpy as np
import os
import rasterio
from rasterio.plot import show


imgmat = cv2.imread('./test3.tif',255)
# gamma factor brightness
imgmat = (((imgmat/65535).astype(np.float16)**0.3)*65535).astype(np.uint16)
cv2.imshow('original',imgmat)


# rotating matrix
rows,cols = imgmat.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),44.25,1)
dst = cv2.warpAffine(src=imgmat,M=M,dsize=imgmat.shape,flags=cv2.INTER_NEAREST)
cv2.imshow('NEAREST',dst) #nearest approximation of rotation


# remove padding of image
h = dst.shape[0]
w = dst.shape[1]
center = [h//2, w//2]
pad_left, pad_right, pad_top, pad_bottom = 0, w-1, 0, h-1
while(dst[center[0],pad_left] ==0):
    pad_left+=1
while(dst[center[0],pad_right] ==0):
    pad_right-=1
while(dst[pad_top, center[1]] ==0):
    pad_top+=1
while(dst[pad_bottom, center[1]] ==0):
    pad_bottom-=1


# Cropping to fit 800x800 Blocks

cropped = dst[pad_top+297:pad_bottom-297,pad_left+106:pad_right-105]
cv2.imshow('CROPPED',cropped)

square_size = 800
images = []
for i in range(0,8000,800):
    for j in range(0,4000,800):
        images.append(cropped[i:i+800,j:j+800])

cv2.imshow('Square',images[0])


# Setting low resolution

images_low = [cv2.resize(img,(400,400)) for img in images]
cv2.imshow('Square2',images_low[0])

# Writing files to folders
# TODO: png or tif?
output_directory_high = "highRes"
output_directory_low = "lowRes"

for i, img in enumerate(images):
    image_path_high = os.path.join(output_directory_high, f'image_high_{i}.png')
    cv2.imwrite(image_path_high, img)


for i, low_img in enumerate(images_low):
    image_path_low = os.path.join(output_directory_low, f'image_low_{i}.png')
    cv2.imwrite(image_path_low, low_img)


cv2.waitKey(0)
cv2.destroyAllWindows()