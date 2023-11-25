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

    # แยก channel H (Hue) มาจากภาพ HSV
    h, s, v = cv2.split(hsv)

    # แปลงภาพ channel H ให้เป็นภาพสีเทา
    h_colored = cv2.applyColorMap(h, cv2.COLORMAP_JET)

    # รวมภาพ BGR และ channel H เข้าด้วยกันแนวนอน
    combined = np.hstack((frame, h_colored))

    cv2.imshow('BGR and Hue', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
