import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

prev_thumb_tip = None

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(image)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                thumb_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y])
                index_finger_base = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y])
                
                if prev_thumb_tip is not None and np.linalg.norm(thumb_tip - prev_thumb_tip) > 0.01:
                    cv2.putText(image, 'Ola!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                prev_thumb_tip = thumb_tip
        
        cv2.imshow('MediaPipe Hands', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
