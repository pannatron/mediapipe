import cv2
import mediapipe as mp

# สร้างโอบเจคท์สำหรับการตรวจจับใบหน้า
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.5)

# สร้างโอบเจคท์สำหรับการวาด
mp_drawing = mp.solutions.drawing_utils

# กำหนดฟังก์ชันสำหรับการประมวลผลภาพ
# def process_image(image):
#     # ประมวลผลภาพด้วย MediaPipe Face Detection
#     results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     # ตรวจสอบว่ามีใบหน้าในภาพหรือไม่
#     if results.detections:
#         for detection in results.detections:
#             # วาดกรอบรอบใบหน้า
#             mp_drawing.draw_detection(image, detection)

#     return image
def process_image(image):
    # ประมวลผลภาพด้วย MediaPipe Face Detection
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # ตรวจสอบว่ามีใบหน้าในภาพหรือไม่
    if results.detections:
        for detection in results.detections:
            # กำหนดสีและความหนาของกรอบ
            bboxC = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            
            # วาดกรอบรอบใบหน้าด้วยสีและความหนาที่กำหนด
            mp_drawing.draw_detection(image, detection, bboxC)

    return image


# เปิดกล้อง
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # ประมวลผลภาพและวาดใบหน้า
    frame = process_image(frame)

    # แสดงผลลัพธ์
    cv2.imshow('MediaPipe Face Detection', frame)

    # ปิดโปรแกรมเมื่อกดปุ่ม 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและปล่อยทรัพยากร
cap.release()
cv2.destroyAllWindows()
