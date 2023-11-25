import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def process_image(image):
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find face landmarks
    results = face_mesh.process(image_rgb)
    
    # If face landmarks are found, draw them
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Specify the color (for example, green) and thickness for landmarks
            landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)
            # Specify a different color (for example, blue) and thickness for connections
            connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            
            # Draw the face mesh landmarks and connections with the specified colors and thicknesses
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=landmark_drawing_spec,
                connection_drawing_spec=connection_drawing_spec)
    
    return image


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = process_image(frame)

    cv2.imshow('MediaPipe Face Mesh', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
