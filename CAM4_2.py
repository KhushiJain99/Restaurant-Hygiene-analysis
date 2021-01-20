
import cv2
import imutils
import numpy as np
import joblib
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Load testing image A and plot saved coordinates on it
imgA = cv2.imread("E:/ My Desktop/Softvan/NYB/CAM4_test.jpg",0)
imgA = imutils.resize(imgA , width=500)
pts = joblib.load(filename="config.pkl",mmap_mode=None)

mask = np.zeros(imgA.shape, np.uint8)
points = np.array(pts.get("ROI"), np.int32)
points = points.reshape((-1, 1, 2))
          
mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)

mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255)) # for ROI

status=cv2.imwrite('Mask_test.jpg', mask2)
print("Image written to disk", status)

imgA = cv2.bitwise_and(mask2, imgA)

# Load image B
imgB = cv2.imread("E:/ My Desktop/Softvan/NYB/ROI.jpg",0)
imgB = imutils.resize(imgB , width=500) 

# Binary threshold for removing noise on both images
imageA=cv2.adaptiveThreshold(imgA,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
imageB=cv2.adaptiveThreshold(imgB,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    
#Save cropped image A
cv2.imwrite("cropped_ROI.jpg",imageA)
cv2.imshow("Image1", imageA)
cv2.waitKey(0)

cv2.imshow("Image2", imageB)
cv2.waitKey(0)

#Mean Square error
def mse(imgA, imgB):
	
	err = np.sum((imgA.astype("float") - imgB.astype("float")) ** 2)
	err /= float(imgA.shape[0] * imgB.shape[1])
	
	return err
m = mse(imageA, imageB)

#SSID
def compare_images(imgA, imgB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	
	s = ssim(imgA, imgB)
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imgA, cmap = plt.cm.gray)
	plt.axis("off")
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imgB, cmap = plt.cm.gray)
	plt.axis("off")
	# show the images
	plt.show()
    
compare_images(imageA, imageB, "Comparison")

# Pixel counting
#from PIL import Image 
h, w = imageA.shape[0], imageA.shape[1]
#print('width: ', w)
#print('height:', h)

TP = w * h

n_white_pix = np.sum(imageA == 255)
print('Number of white pixels:', n_white_pix)

n_black_pix = TP - n_white_pix  
print('Number of black pixels:',n_black_pix)

Ratio= n_white_pix / TP 
print('Ratio in image A :')
print("%.5f" %Ratio)

# Find Pixel counting for Image B
h, w = imageB.shape[0], imageB.shape[1]

TP = w * h

n_white_pix = np.sum(imageB == 255)
print('Number of white pixels:', n_white_pix)

n_black_pix = TP - n_white_pix  
print('Number of black pixels:',n_black_pix)

Ratio= n_white_pix / TP 
print('Ratio in image B :' )
print("%.5f" %Ratio)
# Histogram 

cv2.imshow(" img A", imgA)
cv2.waitKey(0)

cv2.imshow(" img B", imgB)
cv2.waitKey(0)
cv2.destroyAllWindows()

histogram=cv2.calcHist([imgA],[0],None,[256],[0,256])

histogram1=cv2.calcHist([imgB],[0],None,[256],[0,256])
c1, c2 = 0, 0

# Euclidean Distace between data and test image
i = 0
while i<len(histogram) and i<len(histogram1): 
    c1+=(histogram[i]-histogram1[i])**2
    i+= 1
c1 = c1**(1 / 2) 
print ('Euclidean Distace', c1)

if m<140 and ssim>0.98 :
    print("*********************The Table is clean!**************************")
    
else:
    print("*******************The Table is not clean!************************")
    