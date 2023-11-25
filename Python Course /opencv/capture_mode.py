import cv2
import numpy as np

# กำหนดโหมดรูปภาพ
modes = {0: 'Normal', 1: 'Grayscale', 2: 'Edges'}
mode = 0  # โหมดเริ่มต้นคือโหมดปกติ

# เริ่มต้นกล้อง
cap = cv2.VideoCapture(0)

while True:
    # อ่านภาพจากกล้อง
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # แปลงภาพตามโหมดที่เลือก
    if mode == 1:
        # โหมด Grayscale
        displayed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # แปลงภาพเป็นสีเพื่อทำให้สามารถใส่ข้อความสีได้
        displayed_frame = cv2.cvtColor(displayed_frame, cv2.COLOR_GRAY2BGR)
    elif mode == 2:
        # โหมด Edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        displayed_frame = cv2.Canny(gray, 100, 200)
        # แปลงภาพเป็นสีเพื่อทำให้สามารถใส่ข้อความสีได้
        displayed_frame = cv2.cvtColor(displayed_frame, cv2.COLOR_GRAY2BGR)
    else:
        # โหมดปกติ
        displayed_frame = frame

    # ใส่ข้อความเพื่อบอกโหมดที่ใช้
    # ปรับตำแหน่งข้อความในแนวแกน Y และขนาดของตัวอักษร
    cv2.putText(displayed_frame, f"Mode: {modes[mode]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # แสดงภาพ
    cv2.imshow('frame', displayed_frame)

    # รอการกดปุ่ม
    key = cv2.waitKey(1) & 0xFF

    # ถ้ากด space bar, ถ่ายรูป
    if key == 32:  # 32 คือ ASCII Code ของ space bar
        img_name = f"opencv_frame_{mode}_{modes[mode]}.png"
        cv2.imwrite(img_name, displayed_frame)
        print(f"{img_name} saved!")

    # ถ้ากด 'a', สลับโหมด
    if key == ord('a'):
        mode = (mode + 1) % 3
        print(f"Mode changed to {modes[mode]}")

    # ถ้ากด 'q', ออกจากโปรแกรม
    if key == ord('q'):
        break

# ปิดกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
