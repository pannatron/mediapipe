import cv2
import numpy as np

# Callback function for trackbars
def nothing(x):
    pass

cap = cv2.VideoCapture(1)

# Create trackbars for color change in RGB and HSV
cv2.namedWindow("RGB Adjust")
cv2.namedWindow("HSV Adjust")

# Trackbars for lower bounds of RGB
cv2.createTrackbar("R_low", "RGB Adjust", 0, 255, nothing)
cv2.createTrackbar("G_low", "RGB Adjust", 0, 255, nothing)
cv2.createTrackbar("B_low", "RGB Adjust", 0, 255, nothing)

# Trackbars for upper bounds of RGB
cv2.createTrackbar("R_high", "RGB Adjust", 255, 255, nothing)
cv2.createTrackbar("G_high", "RGB Adjust", 255, 255, nothing)
cv2.createTrackbar("B_high", "RGB Adjust", 255, 255, nothing)

# Trackbars for lower bounds of HSV
cv2.createTrackbar("H_low", "HSV Adjust", 0, 180, nothing)
cv2.createTrackbar("S_low", "HSV Adjust", 0, 255, nothing)
cv2.createTrackbar("V_low", "HSV Adjust", 0, 255, nothing)

# Trackbars for upper bounds of HSV
cv2.createTrackbar("H_high", "HSV Adjust", 180, 180, nothing)
cv2.createTrackbar("S_high", "HSV Adjust", 255, 255, nothing)
cv2.createTrackbar("V_high", "HSV Adjust", 255, 255, nothing)

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get RGB values from trackbars for lower bounds
    r_low = cv2.getTrackbarPos("R_low", "RGB Adjust")
    g_low = cv2.getTrackbarPos("G_low", "RGB Adjust")
    b_low = cv2.getTrackbarPos("B_low", "RGB Adjust")

    # Get RGB values from trackbars for upper bounds
    r_high = cv2.getTrackbarPos("R_high", "RGB Adjust")
    g_high = cv2.getTrackbarPos("G_high", "RGB Adjust")
    b_high = cv2.getTrackbarPos("B_high", "RGB Adjust")

    # RGB mask
    lower_bound_rgb = np.array([b_low, g_low, r_low]) # Note that OpenCV uses BGR format
    upper_bound_rgb = np.array([b_high, g_high, r_high]) 
    mask_rgb = cv2.inRange(frame, lower_bound_rgb, upper_bound_rgb)
    result_rgb = cv2.bitwise_and(frame, frame, mask=mask_rgb)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get HSV values from trackbars for lower bounds
    h_low = cv2.getTrackbarPos("H_low", "HSV Adjust")
    s_low = cv2.getTrackbarPos("S_low", "HSV Adjust")
    v_low = cv2.getTrackbarPos("V_low", "HSV Adjust")

    # Get HSV values from trackbars for upper bounds
    h_high = cv2.getTrackbarPos("H_high", "HSV Adjust")
    s_high = cv2.getTrackbarPos("S_high", "HSV Adjust")
    v_high = cv2.getTrackbarPos("V_high", "HSV Adjust")

    # Create the lower and upper bounds for the HSV mask
    lower_bound = np.array([h_low, s_low, v_low])
    upper_bound = np.array([h_high, s_high, v_high])

    # Create a mask that identifies pixels in the HSV range and apply it
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Background subtraction
    fgmask = fgbg.apply(frame)

    fgmask_colored = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    gray_colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    edges = cv2.Canny(gray, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Concatenate images
    upper_row = np.hstack((frame, gray_colored, edges_colored))
    lower_row = np.hstack((result_rgb, result, fgmask_colored))  # Using the masked results here.
    final_display = np.vstack((upper_row, lower_row))

    cv2.imshow('Final Display', final_display)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
