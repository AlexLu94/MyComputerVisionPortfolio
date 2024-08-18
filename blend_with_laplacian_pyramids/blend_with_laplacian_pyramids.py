import cv2   as cv
import numpy as np

def get_gaussian_pyramid(img, N):
    temp = img.copy()
    retv = [temp]
    for _ in range(N):
        temp = cv.pyrDown(temp)
        retv.append(temp)
    return retv

def get_laplacian_pyramid(img, N):
    gaussian_pyr = get_gaussian_pyramid(img, N)
    retv         = [gaussian_pyr[N-1]]
    for i in range(N-1,0,-1):
        retv.append(cv.subtract(gaussian_pyr[i-1], cv.pyrUp(gaussian_pyr[i])))
    return retv

def merge_masked(img1, img2, mask):
    img = cv.copyTo(img1, mask)
    cv.copyTo(img2, cv.bitwise_not(mask), img)
    return img

N = 9

img1 = cv.imread('./image1.jpg')
if img1 is None:
    print("Error reading img1")
    exit()
img2 = cv.imread('./image2.jpg')
if img2 is None:
    print("Error reading img2")
    exit()
mask = cv.imread('./mask.jpg')
if mask is None:
    print("Error reading mask")
    exit()
if img2.shape != img1.shape or mask.shape != img1.shape:
    print("Error, the two images and the mask must have same shape")
    exit()

sizex = 800*8
sizey = 800*8
img1 = cv.resize(img1, (sizex, sizey))
img2 = cv.resize(img2, (sizex, sizey))
mask = cv.resize(mask, (sizex, sizey))

laplacian_pyr1    = get_laplacian_pyramid(img1, N)
laplacian_pyr2    = get_laplacian_pyramid(img2, N)
mask_gaussian_pyr = get_gaussian_pyramid(mask, N)

# Merge every level of laplacian pyramid
out_laplacian_pyr = []
for i in range(N):
    out_laplacian_pyr.append(merge_masked(laplacian_pyr1[i], laplacian_pyr2[i], mask_gaussian_pyr[N-i-1]))

# now reconstruct
output = out_laplacian_pyr[0]
for i in range(1, N):
    output = cv.pyrUp(output)
    output = cv.add(output, out_laplacian_pyr[i])
 
cv.imwrite('output.jpg',output)
