import cv2
import mediapipe as mp
import numpy as np

# สร้างโอบเจคท์สำหรับการตรวจจับ Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# เปิดกล้อง
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # ประมวลผลภาพด้วย MediaPipe Selfie Segmentation
    results = selfie_segmentation.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # สร้าง mask โดยใช้ผลลัพธ์ของการ segmentation
    # ทำการเปลี่ยนสีของพื้นหลังโดยใช้ mask
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(frame.shape, dtype=np.uint8)  # สามารถเปลี่ยนเป็นรูปภาพพื้นหลังอื่นๆ
    bg_image[:] = (192, 192, 192)  # สีพื้นหลังที่ต้องการ (ในที่นี้เป็นสีเทา)
    output_image = np.where(condition, frame, bg_image)

    # แสดงผลลัพธ์
    cv2.imshow('Selfie Segmentation', output_image)

    # ปิดโปรแกรมเมื่อกดปุ่ม 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# ปิดกล้องและปล่อยทรัพยากร
cap.release()
cv2.destroyAllWindows()
