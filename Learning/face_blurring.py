import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()

    H, W, _ = frame.shape

    # detect face
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(
        min_detection_confidence=0.5, # minimum confidence value for successful detection
        model_selection=0 # detect faces within 2 meters of the camera
    ) as face_detection:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = face_detection.process(frame_rgb)

        if out.detections:

            for detection in out.detections:
                
                bbox = detection.location_data.relative_bounding_box

                x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

                x1 = int(x1 * W)
                w = int(w * W)
                y1 = int(y1 * H)
                h = int(h * H)

                # cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

                # blurring the face is as simple as applying a blur to only the matrix values inside the bounding box
                frame[y1:y1+h, x1:x1+w, :] = cv2.blur(frame[y1:y1+h, x1:x1+w, :], (30, 30))

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()