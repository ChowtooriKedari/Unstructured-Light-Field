import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
from scipy import interpolate
from Helper import writeEXR,lRGB2XYZ,readEXR, XYZ2lRGB, writeHDR, read_colorchecker_gm,gammaDecoding
import cv2

img_path = 'data/chessboard_lightfield.png' 
img = np.array(Image.open(img_path))

lenslet_height = 16  # Number of pixels per lenslet block
lenslet_width = 16   # Number of pixels per lenslet block

num_lenslets_vertical = img.shape[0] // lenslet_height
num_lenslets_horizontal = img.shape[1] // lenslet_width

# Reshape the image to a 5D array: [u, v, s, t, c]
L = np.transpose((img.reshape(16, num_lenslets_vertical, 16, num_lenslets_horizontal, 3, order='F')), (2, 0, 1, 3, 4)).astype(np.uint8)

# fig, axes = plt.subplots(nrows=num_lenslets_vertical, ncols=num_lenslets_horizontal, figsize=(20, 20))
# plt.subplots_adjust(wspace=0, hspace=0)

# # Display each sub-aperture image in the corresponding subplot
# for u in range(num_lenslets_vertical):
#     for v in range(num_lenslets_horizontal):
#         sub_aperture_img = L[u, :, v, :, :]
#         ax = axes[u, v]
#         ax.imshow(sub_aperture_img)
#         ax.axis('off')  # Turn off the axis

L = np.transpose(L, (1, 0, 2, 3, 4))
grid_Image = np.transpose(L, (0, 2, 1, 3, 4)).reshape(16*L.shape[2], 16*L.shape[3], 3)
plt.imshow(grid_Image)
plt.show()

lensletSize = 16
maxUV = (lensletSize - 1) / 2


def mainCheck(a,u,v):
    if(np.abs(u)<=maxUV and np.abs(v)<=maxUV):
        return True
    return False

U = np.arange(lensletSize) - maxUV
V = np.arange(lensletSize) - maxUV

def interp(I,u,v,depth):
    p=np.arange(I.shape[1])
    q=np.arange(I.shape[0])
    image=interpolate.interp2d(p,q,I)
    return image(p+(depth*u),q-(depth*v))

def depthRefocus(lightfield,depth,a=1,aperture='max'):
    refocused_image =np.zeros((lightfield.shape[2], lightfield.shape[3], lightfield.shape[4]))
    aperture_size = (lensletSize) ** 2  # Define aperture_size based on maxUV

    for u in range(len(U)):
        for v in range(len(V)):
            # du = int(depth * u)
            # dv = int(depth * v)
            # valid_s = slice(max(0, -du), min(16, 16 - du))
            # valid_t = slice(max(0, -dv), min(16, 16 - dv))
            # shifted_s = slice(max(0, du), min(16, 16 + du))
            # shifted_t = slice(max(0, dv), min(16, 16 + dv))
            image = np.dstack((interp(lightfield[u, v, :, :, 0], U[u], V[v], depth),
                                interp(lightfield[u, v, :, :, 1], U[u], V[v], depth), 
                                interp(lightfield[u, v, :, :, 2], U[u], V[v], depth)))

            if (aperture == 'max' and mainCheck(U[u], V[v], maxUV) == False):
                continue

            refocused_image += image
    return (refocused_image / aperture_size)/255

plot = plt.figure()
image_1 = depthRefocus(L, -1.5)
plot.add_subplot(2,3,1)
plt.title('Depth=-1.5')
plt.imshow(image_1)
image_2 = depthRefocus(L, -1.0)
plot.add_subplot(2,3,2)
plt.title('Depth=-1.0')
plt.imshow(image_2)
image_3 = depthRefocus(L, -0.5)
plot.add_subplot(2,3,3)
plt.title('Depth=-0.5')
plt.imshow(image_3)
image_4 = depthRefocus(L,0)
plot.add_subplot(2,3,4)
plt.title('Depth=0')
plt.imshow(image_4)
image_5 = depthRefocus(L, 0.5)
plot.add_subplot(2,3,5)
plt.title('Depth=0.5')
plt.imshow(image_5)
image_6 = depthRefocus(L, 1.0)
plot.add_subplot(2,3,6)
plt.title('Depth=1.0')
plt.imshow(image_6)
plt.show()


def allInFocus(focalStack, kernel1, kernel2, sigma1, sigma2, depths):

    depths = np.asarray(depths + abs(np.min(depths)))
    depths = depths / np.max(depths)

    illuminant = np.zeros((focalStack.shape))
    for i in range(len(depths)):
        illuminant[:, :, :, i] = lRGB2XYZ(gammaDecoding(focalStack[:, :, :, i]))
    luminance = illuminant[:, :, 1, :]
    gaussianBlur = cv2.GaussianBlur(luminance, [kernel1, kernel1], sigma1)
    high_frequency = luminance - gaussianBlur
    weighted_Image = cv2.GaussianBlur((high_frequency**2), [kernel2, kernel2], sigma2)
    weighted_Image = np.stack((weighted_Image, weighted_Image, weighted_Image), axis = 2)
    focusedImage = np.nan_to_num(np.divide(np.sum(weighted_Image * focalStack, axis=-1), np.sum(weighted_Image, axis=-1)))
    depthImage = np.nan_to_num(np.divide(np.sum(weighted_Image * depths[None, None, None, :], axis=-1), np.sum(weighted_Image, axis=-1)))

    return focusedImage, depthImage

depths = np.linspace(-1.5, 0, 15)
focal_stack = np.zeros((L.shape[2], L.shape[3], L.shape[4], len(depths)))

for depth in range(len(depths)):
        focal_stack[:, :, :, depth] = depthRefocus(L, depths[depth])

image, depth = allInFocus(focal_stack, 7, 35, 0, 0, depths)
pltname='focusImageNew.exr'
pltname1='DepthMapNew.exr'

writeEXR(pltname,image)
writeEXR(pltname1,depth)


image, depth = allInFocus(focal_stack, 9, 37, 5, 5, depths)
pltname='focusImageNew_1.exr'
pltname1='DepthMapNew_1.exr'

writeEXR(pltname,image)
writeEXR(pltname1,depth)

image, depth = allInFocus(focal_stack, 3, 33, 3, 3, depths)
pltname='focusImageNew_2.exr'
pltname1='DepthMapNew_2.exr'

writeEXR(pltname,image)
writeEXR(pltname1,depth)