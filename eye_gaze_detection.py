# import cv2

# # Load Haar Cascade classifiers for face and eye detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# # Start video capture from webcam
# cap = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert to grayscale for better detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around face

#         # Region of interest for eyes
#         roi_gray = gray[y:y + h, x:x + w]
#         roi_color = frame[y:y + h, x:x + w]

#         # Detect eyes
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         eye_positions = []

#         for (ex, ey, ew, eh) in eyes:
#             cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # Draw rectangle around eyes
#             eye_positions.append((ex + ew / 2, ey + eh / 2))  # Store center of each eye

#         if len(eye_positions) == 2:  # Ensure both eyes are detected
#             left_eye = eye_positions[0]
#             right_eye = eye_positions[1]
#             eye_center_x = (left_eye[0] + right_eye[0]) / 2

#             # Determine gaze direction
#             if eye_center_x < x + w / 3:
#                 gaze_direction = "Looking Left"
#             elif eye_center_x > x + 2 * w / 3:
#                 gaze_direction = "Looking Right"
#             else:
#                 gaze_direction = "Looking Center"

#             # Display gaze direction on the frame
#             cv2.putText(frame, gaze_direction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     # Display the resulting frame
#     cv2.imshow('Eye Gaze Detection', frame)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the capture and close windows
# cap.release()
# cv2.destroyAllWindows()

import cv2

# Load Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start video capture from webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Region of interest for eyes
            roi_gray = gray[y:y + h, x:x + w]

            # Detect eyes
            eyes = eye_cascade.detectMultiScale(roi_gray)
            eye_positions = []

            for (ex, ey, ew, eh) in eyes:
                eye_positions.append((ex + ew / 2, ey + eh / 2))  # Store center of each eye

            if len(eye_positions) == 2:  # Ensure both eyes are detected
                left_eye = eye_positions[0]
                right_eye = eye_positions[1]
                eye_center_x = (left_eye[0] + right_eye[0]) / 2

                # Determine gaze direction
                if eye_center_x < x + w / 3:
                    gaze_direction = "Looking Left"
                elif eye_center_x > x + 2 * w / 3:
                    gaze_direction = "Looking Right"
                else:
                    gaze_direction = "Looking Center"

                print(gaze_direction)  # Print the gaze direction to the console

except KeyboardInterrupt:
    # Allow the user to exit the loop with Ctrl+C
    print("Exiting...")

# Release the capture
cap.release()
