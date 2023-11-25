import cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้!")
    exit()

# สร้างตัวลบพื้นหลัง
fgbg = cv2.createBackgroundSubtractorMOG2()

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
    
    # ใช้ mask ในการลบพื้นหลังจากเฟรมต้นฉบับ
    # ต้องทำการให้มิติของ fgmask กับ frame เท่ากันก่อน
    fgmask_3d = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    # รวมเฟรมและ mask ด้วย bitwise_and
    fg = cv2.bitwise_and(frame, fgmask_3d)

    # รวมภาพ frame และ fg เข้าด้วยกัน
    stacked_images = cv2.hconcat([frame, fg])

    # แสดงภาพผลลัพธ์
    cv2.imshow('Original and Foreground', stacked_images)

    # หากกด 'q' ให้ออกจาก loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
