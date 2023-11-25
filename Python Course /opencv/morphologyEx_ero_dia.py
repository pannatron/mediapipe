import cv2
import numpy as np

cap = cv2.VideoCapture(1)  # ใช้กล้องคอมพิวเตอร์หลัก

# ปรับขนาดและรูปร่างของ kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ทำ Erosion ด้วย iterations มากขึ้น
    eroded = cv2.erode(gray, kernel, iterations=2)
    
    # ทำ Dilation ด้วย iterations มากขึ้น
    dilated = cv2.dilate(gray, kernel, iterations=2)
    
    # ใช้ morphologyEx เพื่อทำ closing (dilation ตามด้วย erosion)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    
    # ต่อภาพทั้งหมดเข้าด้วยกันแนวนอน
    combined = np.hstack((gray, eroded, dilated, closed))
    
    cv2.imshow('Original | Eroded | Dilated | Closed', combined)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
