import cv2
import imutils
import numpy as np
import joblib
from matplotlib import pyplot as plt

#  https://www.programmersought.com/article/3449903953/

# load image, initially BGR
img = cv2.imread("E:/ My Desktop/Softvan/NYB/CAM4_Base.jpg")
img= imutils.resize(img , width=500)
# show this image
cv2.imshow("original_img", img)

# convert it to grayscale, show
img=cv2.imread("E:/ My Desktop/Softvan/NYB/CAM4_Base.jpg", 0)
img= imutils.resize(img , width=500)
cv2.imshow("grayscale_img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# crop using coordinates and save cropped image.
pts = [] # for storing points

 # :mouse callback function

def draw_roi(event, x, y, flags, param):
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN: # Left click, select point
        pts.append((x, y))  

    if event == cv2.EVENT_RBUTTONDOWN: # Right click to save points
        
        mask = np.zeros(img.shape, np.uint8)
        points = np.array(pts, np.int32)
        points = points.reshape((-1, 1, 2))

                  
        mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)

        mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255)) # for ROI

        mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0)) # for displaying images on the desktop

        show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)

        cv2.imshow("mask", mask2)
        # saving the mask
        status=cv2.imwrite('E:/ My Desktop/Softvan/NYB/Mask.jpg', mask2)
        print("Image written to disk", status)
        
        cv2.imshow("show_img", show_image)

        ROI = cv2.bitwise_and(mask2, img)
        #show and save image
        cv2.imshow("ROI", ROI)
        cv2.imwrite("cropped_ROI.jpg",ROI)
        status=cv2.imwrite('E:/ My Desktop/Softvan/NYB/ROI.jpg', ROI)
        print("ROI img written to disk", status)
        
        #hold window
        cv2.waitKey(0)
        


    if len(pts) > 0:
                 # Draw the last point in pts
        cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)
    if len(pts) > 1:
                 # 
        for i in range(len(pts) - 1):
            cv2.circle(img2, pts[i], 5, (0, 0, 255), -1) # x ,y is the coordinates of the mouse click place
            cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)
    cv2.imshow('image', img2)

 #Create images and windows and bind windows to callback functions
img = cv2.imread("E:/ My Desktop/Softvan/NYB/CAM4_Base.jpg")
img = imutils.resize(img , width=500)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_roi)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord("s"):
        saved_data = {
            "ROI": pts
        }

        joblib.dump(value=saved_data, filename="config.pkl")
        print("[INFO] ROI coordinates have been saved to local.")
        break
 
 # Module 2    
data = joblib.load(filename="config.pkl",mmap_mode=None)
print (data)
img = cv2.imread("cropped_ROI.jpg",0)
img = cv2.medianBlur(img,5)
ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

cv2.destroyAllWindows()



