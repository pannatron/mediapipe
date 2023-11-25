import cv2
import time

cap = cv2.VideoCapture(0)  # เปิดกล้องของคอมพิวเตอร์

if not cap.isOpened():
    print("Error: ไม่สามารถเปิดกล้องได้!")
    exit()

while True:
    start_time = time.time()
    
    ret, frame = cap.read()  # ดึงภาพจากกล้อง
    if not ret:
        break
    
    # แปลงภาพเป็น greyscale และเปลี่ยนกลับเป็น BGR เพื่อให้มีขนาดและ channel เท่ากับ frame
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey_bgr = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    
    end_time = time.time()
    fps = 1.0 / (end_time - start_time)
    fps_text = f"FPS: {fps:.2f}"

    # เขียนข้อความ FPS ลงบนรูปภาพของแต่ละฟีด
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(grey_bgr, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # ต่อภาพแนวนอน
    stacked_image = cv2.hconcat([frame, grey_bgr])
    
    # แสดงภาพ
    cv2.imshow('Camera Feed', stacked_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
