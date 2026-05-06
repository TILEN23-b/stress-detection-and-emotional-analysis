import cv2
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Try multiple camera indexes
cap = None
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera working at index {i}")
        break

if not cap or not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.putText(frame, emotion, (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
