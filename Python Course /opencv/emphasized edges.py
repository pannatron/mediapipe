import cv2

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 100, 200)
    
    # Emphasize edges
    emphasized_edges = cv2.addWeighted(frame, 0.9, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.1, 0)

    # Convert grayscale image back to BGR for stacking
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Stack the images horizontally
    result = cv2.hconcat([frame, gray_bgr, emphasized_edges])

    cv2.imshow('Comparison', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
