import cv2
import numpy as np

# สร้าง connection กับกล้อง
cap = cv2.VideoCapture(1)

# ตรวจสอบว่าเปิดกล้องสำเร็จหรือไม่
if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้!")
    exit()

while True:
    # รับภาพจากกล้อง
    ret, frame = cap.read()
    
    # ถ้าไม่สามารถรับภาพได้, จบ loop
    if not ret:
        break

    # ตั้งค่า threshold สำหรับแต่ละ channel
    lower = np.array([30, 30, 30])  # ค่าสีต่ำสุดที่ต้องการคัดกรอง
    upper = np.array([150, 150, 150])  # ค่าสีสูงสุดที่ต้องการคัดกรอง

    mask = cv2.inRange(frame, lower, upper)

    # แสดงภาพ
    cv2.imshow('RGB Threshold', mask)
    
    # กด 'q' เพื่อปิดหน้าต่าง
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
