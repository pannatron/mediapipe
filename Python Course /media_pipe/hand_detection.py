import cv2
import mediapipe as mp

# สร้างโอบเจคท์สำหรับการตรวจจับมือ
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.1)

# สร้างโอบเจคท์สำหรับการวาด
mp_drawing = mp.solutions.drawing_utils

# def process_image(image):
#     # Convert the image from BGR to RGB
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Process the image and find hand landmarks
#     results = hands.process(image_rgb)
    
#     # If landmarks are found, draw them
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Specify the color (red) and thickness for landmarks and connections
#             landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=3)
#             connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5)
            
#             # Draw the landmarks and the connections in red
#             mp_drawing.draw_landmarks(
#                 image,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 landmark_drawing_spec,
#                 connection_drawing_spec)
    
#     return image
def process_image(image):
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hand landmarks
    results = hands.process(image_rgb)
    
    # If landmarks are found, draw them
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Create specifications for drawing the landmarks
            # The first spec is for a larger, semi-transparent glow
            glow_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=10, circle_radius=7)
            # The second spec is for the solid red lines and dots
            solid_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=2)
            
            # Draw the glow first
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                glow_spec,
                glow_spec)
            
            # Draw the solid lines and dots on top of the glow
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                solid_spec,
                solid_spec)
    
    return image


# เปิดกล้อง
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # ประมวลผลภาพและวาดมือ
    frame = process_image(frame)

    # แสดงผลลัพธ์
    cv2.imshow('MediaPipe Hands', frame)

    # ปิดโปรแกรมเมื่อกดปุ่ม 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและปล่อยทรัพยากร
cap.release()
cv2.destroyAllWindows()
