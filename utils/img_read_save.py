import numpy as np
import cv2
import os
from PIL import Image
from skimage.io import imsave

def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    # h, w, c = img.shape
    # print(img.shape)
    # img = cv2.resize(img, dsize=(img.shape[1]//2, img.shape[0]//2))
    return img

def img_save(image,imagename,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # Gray_pic
    image=Image.fromarray(image)
    if image.mode == "F":
        image = image.convert('RGB')
    # image.save(os.path.join(savepath, "{}".format(imagename)))
    # print(os.path.join(savepath, "{}.jpg".format(imagename)))
    image = np.asanyarray(image)
    imsave(os.path.join(savepath, "{}".format(imagename)),image)