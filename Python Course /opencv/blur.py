import cv2
import numpy as np

# ตั้งค่ากล้อง
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from webcam.")
        break

    # แปลงภาพเป็น grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Box Blur (Average Blur)
    # cv2.blur() ใช้ kernel size 5x5 ซึ่งหมายความว่าค่าเฉลี่ยจะถูกคำนวณจากพิกเซล 5x5 บริเวณรอบๆ
    blur_box = cv2.blur(gray, (33,33))

    # 2. Gaussian Blur
    # cv2.GaussianBlur() ใช้ kernel size 5x5 และ standard deviation ในทิศทาง X เป็น 0
    # ซึ่ง OpenCV จะคำนวณค่า standard deviation จาก kernel size
    blur_gaussian = cv2.GaussianBlur(gray, (33,33), 0)

    # 3. Median Blur
    # cv2.medianBlur() ใช้ kernel size 5 ซึ่งหมายความว่าค่ามัธยฐานจะถูกคำนวณจากพิกเซล 5x5 บริเวณรอบๆ
    blur_median = cv2.medianBlur(gray, 33)

    # รวมภาพเบลอแบบต่างๆ เข้าด้วยกัน
    combined = np.hstack((blur_box, blur_gaussian, blur_median))

    # ใส่ตัวเลขบนภาพเพื่อบ่งชี้ประเภทของการเบลอ
    cv2.putText(combined, 'Box Blur', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.putText(combined, 'Gaussian Blur', (gray.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.putText(combined, 'Median Blur', (2 * gray.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # แสดงผลลัพธ์
    cv2.imshow('Blurs Comparison', combined)

    # รอการกดปุ่ม 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปล่อยกล้องและปิดหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()
