import numpy as np
import matplotlib.pyplot as plt
from Helper import lRGB2XYZ  # Assuming loadVideo is not needed anymore
from scipy.signal import fftconvolve
from scipy.interpolate import interp2d
import cv2
import glob

def computeShifters(frame, template, template_X, template_Y):

    template = lRGB2XYZ(template)[:, :, 1]
    frame = lRGB2XYZ(frame)[:, :, 1]

    template_Selected = template[template_X:(template_X+800), template_Y:(template_Y+800)]
    template_Selected_mean = np.mean(template_Selected)
    image_part = frame[(template_X-1000):(template_X+1200), (template_Y-1000):(template_Y+1200)]
    image_part_mean = np.mean(image_part)
    result = (fftconvolve(image_part-image_part_mean, template_Selected-template_Selected_mean, mode='same')/np.sqrt(np.sum((template_Selected-template_Selected_mean)**2) * np.sum((image_part-image_part_mean)**2)))
    shifter_X, shifter_Y = np.unravel_index(np.argmax(result), result.shape)
    return shifter_X, shifter_Y

def interp(im, x, y):
    w = np.arange(im.shape[1])
    h = np.arange(im.shape[0])
    i = interp2d(w, h, im)
    int = i(w+(x), h+(y))
    return int

def refocusUnstructured(template, template_x, template_y, video):
    img = np.zeros_like(video[0])
    count = 0
    for frame in video:
        count += 1
        shifter_X, shifter_Y = computeShifters(frame, template, template_x, template_y)
        shiftedImage = np.dstack((interp(frame[:, :, 0], shifter_Y, shifter_X),
                             interp(frame[:, :, 1], shifter_Y, shifter_X),
                             interp(frame[:, :, 2], shifter_Y, shifter_X))).astype(np.uint16)
        img = (img + shiftedImage)
    return img/(count*255)

image_files = [
'Images/Camera1/image_1.jpg',
'Images/Camera2/image_1.jpg',
'Images/Camera3/image_1.jpg',
'Images/Camera4/image_1.jpg',
'Images/Camera1/image_2.jpg',
'Images/Camera2/image_2.jpg',
'Images/Camera3/image_2.jpg',
'Images/Camera4/image_2.jpg',
'Images/Camera1/image_3.jpg',
'Images/Camera2/image_3.jpg',
'Images/Camera3/image_3.jpg',
'Images/Camera4/image_3.jpg',
'Images/Camera1/image_4.jpg',
'Images/Camera2/image_4.jpg',
'Images/Camera3/image_4.jpg',
'Images/Camera4/image_4.jpg'
]
frames = [cv2.imread(img) for img in image_files]

# Select a template from the middle image
index = len(frames) // 2
template = frames[index]

result_1 = refocusUnstructured(template, 2670, 2000, frames)
plt.imshow(result_1)
plt.imsave('result.png', result_1)
plt.show()