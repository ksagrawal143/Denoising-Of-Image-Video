import cv2
import numpy as np
from matplotlib import pyplot as plt

#Denoising of Image:

img=cv2.imread("CamCann/Noisy.jpg")
#Image denoising can be done by either averaging the value of the pixels,or by normal blurring or by gaussian blurring or median blurring.

#Median blur produces the best denoised image but reduces the clearity.

#One can also apply morphological transformations such as erosion or dilation to perform the smoothing and denoising of the image /video.

#Basic of all is that we use mask or kernel of a required size convolve it by moving it over the image.
kernel=np.ones((5,5),np.float32)/25
denoised=cv2.filter2D(img,-1,kernel)

#Applying Gaussian Blur to filter noise in the image
gaussian_blr=cv2.GaussianBlur(img,(15,15),0)

#Applying median blur to filter noise in the image
median_blr=cv2.medianBlur(img,15)

#Output:
cv2.imshow("denoised",denoised)
cv2.imshow("Original",img)
cv2.imshow("Gaussian Blur",gaussian_blr)
cv2.imshow("Median Blur",median_blr)

cv2.waitKey(0)
cv2.destroyAllWindows()

#Denoising of Video:

vid=cv2.VideoCapture(0)
while True:
    _, frame = vid.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#Applying Gaussian Blur to filter noise in the video
    denoise_1=cv2.GaussianBlur(frame,(7,7),0)
#Applying median blur to filter noise in the video
    denoise_2=cv2.medianBlur(frame,7)
    cv2.imshow('Original',frame)
    cv2.imshow('Denoised_1',denoise_1)
    cv2.imshow('Denoised_2',denoise_2)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
