import cv2
import numpy as np

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้!")
    exit()

# สร้างตัวลบพื้นหลัง
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    frame=cv2.resize(frame,(640,480))
    if not ret:
        print("ไม่สามารถรับภาพจากกล้อง!")
        break

    # แปลงภาพเป็นสีเทา
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับขอบด้วย Canny
    edges = cv2.Canny(grey, 50, 150)

    # ลบพื้นหลัง
    fgmask = fgbg.apply(frame)

    # สร้างภาพที่เป็นสี RGB จาก fgmask และ edges (เพื่อให้สามารถซ้อนกันแบบเดียวกันได้)
    fgmask_colored = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # รวมภาพ frame, fgmask_colored, และ edges_colored เข้าด้วยกัน
    stacked_images = np.hstack((frame, fgmask_colored, edges_colored))

    # แสดงภาพผลลัพธ์
    cv2.imshow('Original, Background Subtraction and Edges', stacked_images)

    # หากกด 'q' ให้ออกจาก loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
