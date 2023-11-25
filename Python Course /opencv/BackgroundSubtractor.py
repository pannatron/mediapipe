import cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้!")
    exit()

# สร้างตัวลบพื้นหลัง
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=80, detectShadows=True)

while True:
    ret, frame = cap.read()
    if not ret:
        print("ไม่สามารถรับภาพจากกล้อง!")
        break

    # ใช้ตัวลบพื้นหลัง
    fgmask = fgbg.apply(frame)

    # ใช้ morphology เพื่อปรับปรุงภาพ (เพื่อลด noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    
    # แปลง mask ให้เป็นสีเพื่อนำไปแสดงผลเป็นภาพสี
    fgmask_colored = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)

    # รวมภาพ frame และ fgmask_colored เข้าด้วยกัน
    stacked_images = cv2.hconcat([frame, fgmask_colored])

    # แสดงภาพผลลัพธ์
    cv2.imshow('Original and Background Subtraction', stacked_images)

    # หากกด 'q' ให้ออกจาก loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
