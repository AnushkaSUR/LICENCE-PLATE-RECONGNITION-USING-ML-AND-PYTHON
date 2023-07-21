import numpy as np
import cv2
import imutils
import sys
import pytesseract
import pandas as pd
import time

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image = cv2.imread(r"F:\SEM 4\SDP ML\Mini project SDP ML\License Plate Recognition\archive\images\Cars158.png")

for angle in np.arange(0, 15, 15):
    rotated = imutils.rotate_bound(image, angle)
    cv2.imwrite("rotatedimage.jpg", rotated)

image = imutils.resize(rotated, width=500)
cv2.imwrite("resized.jpg", image)

# Plate Localization
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("Gray.jpg", gray)
cv2.imshow('Gray', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray = cv2.bilateralFilter(gray, 11, 17, 17)

edged = cv2.Canny(gray, 170, 200)
cv2.imwrite("Edged.jpg", edged)
cv2.imshow('Edged', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]
NumberPlateCnt = None

count = 0
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        NumberPlateCnt = approx
        break
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [NumberPlateCnt], 0, 55, -1)
new_image = cv2.bitwise_and(image, image, mask=mask)

cv2.imwrite("Final_image.jpg", new_image)
cv2.imshow('Final', new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Character Segmentation
(x, y) = np.where(mask == 55)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

cv2.imwrite("Cropped.jpg", Cropped)
cv2.imshow('NoPlate', Cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Character Recognition
config = ('-l eng --oem 1 --psm 3')
text = pytesseract.image_to_string(Cropped, config=config)
raw_data = {'date': [time.asctime(time.localtime(time.time()))], 'No plate': [text]}

df = pd.DataFrame(raw_data)
df.to_csv('data.csv', mode='a')
print("Detected Number is: ", text)
