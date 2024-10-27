import cv2
import numpy as np
import math
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks

def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    point_3d = []
    dist_coeffs = np.zeros((4, 1))
    
    # Define rear and front 3D box sizes
    rear_size, rear_depth, front_size, front_depth = val
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    
    point_3d = np.array(point_3d, dtype=np.float64).reshape(-1, 3)

    # Map to 2D image points
    (point_2d, _) = cv2.projectPoints(point_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    return np.int32(point_2d.reshape(-1, 2))

def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix, rear_size=1, rear_depth=0, front_size=None, front_depth=None, color=(255, 255, 0), line_width=2):
    if front_size is None:
        front_size = img.shape[1]
    if front_depth is None:
        front_depth = front_size * 2
    
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    
    # Draw lines from rear to front
    for i in range(4):
        cv2.line(img, tuple(point_2d[i + 5]), tuple(point_2d[i]), color, line_width, cv2.LINE_AA)

def detect_mouth_open(marks):
    # Simple mouth open detection based on vertical distance between mouth corners
    mouth_open_distance = abs(marks[54][1] - marks[48][1])  # Right mouth corner and left mouth corner
    return mouth_open_distance > 15  # Threshold for detecting mouth open

face_model = get_face_detector()
landmark_model = get_landmark_model()

# 3D model points for head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype="double")

def detect_head_pose():
    cap = cv2.VideoCapture(0)  # Use webcam
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, img = cap.read()
        if not ret:
            break
        
        size = img.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)

        # Camera matrix
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        faces = find_faces(img, face_model)
        for face in faces:
            marks = detect_marks(img, landmark_model, face)
            image_points = np.array([
                marks[30],     # Nose tip
                marks[8],      # Chin
                marks[36],     # Left eye left corner
                marks[45],     # Right eye right corner
                marks[48],     # Left Mouth corner
                marks[54]      # Right mouth corner
            ], dtype="double")

            dist_coeffs = np.zeros((4, 1))
            success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
            
            # Draw the annotation box
            draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix)

            # Detect mouth open/close
            if detect_mouth_open(marks):
                cv2.putText(img, 'Mouth Open', (30, 30), font, 2, (0, 0, 255), 3)
            else:
                cv2.putText(img, 'Mouth Closed', (30, 30), font, 2, (0, 255, 0), 3)

        cv2.imshow('Head Pose Detection', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_head_pose()
