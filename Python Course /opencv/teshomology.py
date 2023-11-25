import cv2
import numpy as np

# อ่านภาพ
image = cv2.imread('Cat03.jpg')

# แปลงภาพเป็น grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# กำหนดค่า threshold เพื่อแปลงเป็นภาพ binary
# ในที่นี้ใช้ค่า 127 เป็น threshold และ 255 เป็นค่าสูงสุด
ret, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# สร้าง kernel สำหรับการดำเนินการ morphological
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# การใช้งาน dilation บนภาพ binary
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

# การใช้งาน erosion บนภาพ binary
eroded_image = cv2.erode(binary_image, kernel, iterations=1)

# การใช้งาน opening (erosion ตามด้วย dilation) บนภาพ binary
opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# การใช้งาน closing (dilation ตามด้วย erosion) บนภาพ binary
closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# แสดงภาพ
cv2.imshow('Original', image)
cv2.imshow('Binary', binary_image)
cv2.imshow('Dilated', dilated_image)
cv2.imshow('Eroded', eroded_image)
cv2.imshow('Opened', opened_image)
cv2.imshow('Closed', closed_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
