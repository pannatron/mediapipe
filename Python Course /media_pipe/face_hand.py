import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Drawing specs for face mesh and hands
mp_drawing = mp.solutions.drawing_utils
face_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
hand_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

def process_image(image):
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find face landmarks
    face_results = face_mesh.process(image_rgb)
    # Process the image and find hands landmarks
    hand_results = hands.process(image_rgb)
    
    # If face landmarks are found, draw them
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=face_drawing_spec,
                connection_drawing_spec=face_drawing_spec)

    # If hand landmarks are found, draw them
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=hand_drawing_spec,
                connection_drawing_spec=hand_drawing_spec)

    return image

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = process_image(frame)

    cv2.imshow('MediaPipe Face Mesh and Hands', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources.
cap.release()
cv2.destroyAllWindows

