import cv2
import mediapipe as mp

# สร้างโอบเจคท์สำหรับ Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# สร้างโอบเจคท์สำหรับการวาด
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# กำหนดฟังก์ชันสำหรับการประมวลผลภาพ
def process_image(image):
    # ประมวลผลภาพด้วย MediaPipe Holistic
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # วาดจุด landmark สำหรับใบหน้า, มือและท่าทาง
    # ใช้ 'FACEMESH_TESSELATION' หากต้องการวาดเมชของใบหน้า
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.pose_landmarks,
            connections=mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.left_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
    
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=results.right_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

    return image


# เปิดกล้อง
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # ประมวลผลภาพและวาดจุด landmark สำหรับใบหน้า, มือและท่าทาง
    frame = process_image(frame)

    # แสดงผลลัพธ์
    cv2.imshow('MediaPipe Holistic', frame)

    # ปิดโปรแกรมเมื่อกดปุ่ม 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# ปิดกล้องและปล่อยทรัพยากร
cap.release()
cv2.destroyAllWindows()
