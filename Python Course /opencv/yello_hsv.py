import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้!")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("ไม่สามารถรับภาพจากกล้องได้!")
        break

    # แปลงภาพจาก BGR ไปยัง HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # กำหนดช่วงของสีเหลืองในภาพรูปแบบ HSV
    lower_red1 = np.array([0, 0, 0])
    upper_red1 = np.array([80, 195, 110])

    mask = cv2.inRange(hsv, lower_red1, upper_red1)

    # ปรับปรุง mask ด้วยการทำ morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # นำ mask มาใช้กับภาพเดิม
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # รวมภาพ BGR, Mask และ Result เข้าด้วยกันแนวนอน
    combined = np.hstack((frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result))

    cv2.imshow('BGR | Mask | Result', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
