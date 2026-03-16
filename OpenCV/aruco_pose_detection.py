import cv2 
import numpy as np 

cap = cv2.VideoCapture(0) 

marker_length = 0.05

import numpy as np

camera_matrix = np.array([
    [400.0, 0.0, 320.0],
    [0.0, 400.0, 240.0],
    [0.0, 0.0, 1.0]
], dtype=float)

dist_coeffs = np.zeros((5, 1))


aruco_dicts = [
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50),
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
]

parameters = cv2.aruco.DetectorParameters()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for dictionary in aruco_dicts:

        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                marker_length,
                camera_matrix,
                dist_coeffs
            )

            for i in range(len(ids)):

                cv2.drawFrameAxes(
                    frame,
                    camera_matrix,
                    dist_coeffs,
                    rvecs[i],
                    tvecs[i],
                    0.03
                )

                depth = tvecs[i][0][2]

                text = f"ID:{ids[i][0]} Depth:{depth:.2f}m"

                corner = corners[i][0][0]

                cv2.putText(
                    frame,
                    text,
                    (int(corner[0]), int(corner[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2
                )

    cv2.imshow("Aruco Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print(f"Rotational vectors are: {rvecs[i].flatten()}")
print(f"Translation vectors are: {tvecs[i].flatten()}")




