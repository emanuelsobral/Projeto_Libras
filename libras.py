import cv2
import mediapipe as mp
import numpy as np

mp_desenho = mp.solutions.drawing_utils
mp_maos = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_maos.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as maos: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        resultados = maos.process(imagem)
        
        imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR)
        
        if resultados.multi_hand_landmarks:
            for marcos_mao in resultados.multi_hand_landmarks:
                mp_desenho.draw_landmarks(imagem, marcos_mao, mp_maos.HAND_CONNECTIONS)
                
                #dicionario de dedos
                pontas_dedos = {
                    'polegar': np.array([marcos_mao.landmark[mp_maos.HandLandmark.THUMB_TIP].x, marcos_mao.landmark[mp_maos.HandLandmark.THUMB_TIP].y]),
                    'indicador': np.array([marcos_mao.landmark[mp_maos.HandLandmark.INDEX_FINGER_TIP].x, marcos_mao.landmark[mp_maos.HandLandmark.INDEX_FINGER_TIP].y]),
                    'medio': np.array([marcos_mao.landmark[mp_maos.HandLandmark.MIDDLE_FINGER_TIP].x, marcos_mao.landmark[mp_maos.HandLandmark.MIDDLE_FINGER_TIP].y]),
                    'anelar': np.array([marcos_mao.landmark[mp_maos.HandLandmark.RING_FINGER_TIP].x, marcos_mao.landmark[mp_maos.HandLandmark.RING_FINGER_TIP].y]),
                    'mindinho': np.array([marcos_mao.landmark[mp_maos.HandLandmark.PINKY_TIP].x, marcos_mao.landmark[mp_maos.HandLandmark.PINKY_TIP].y])
                }
                
                #Checa se o dedo esta em pe ou dobrado
                distancias_dedos = {
                    'polegar': np.linalg.norm(pontas_dedos['polegar'] - np.array([marcos_mao.landmark[mp_maos.HandLandmark.THUMB_MCP].x, marcos_mao.landmark[mp_maos.HandLandmark.THUMB_MCP].y])),
                    'indicador': np.linalg.norm(pontas_dedos['indicador'] - np.array([marcos_mao.landmark[mp_maos.HandLandmark.INDEX_FINGER_MCP].x, marcos_mao.landmark[mp_maos.HandLandmark.INDEX_FINGER_MCP].y])),
                    'medio': np.linalg.norm(pontas_dedos['medio'] - np.array([marcos_mao.landmark[mp_maos.HandLandmark.MIDDLE_FINGER_MCP].x, marcos_mao.landmark[mp_maos.HandLandmark.MIDDLE_FINGER_MCP].y])),
                    'anelar': np.linalg.norm(pontas_dedos['anelar'] - np.array([marcos_mao.landmark[mp_maos.HandLandmark.RING_FINGER_MCP].x, marcos_mao.landmark[mp_maos.HandLandmark.RING_FINGER_MCP].y])),
                    'mindinho': np.linalg.norm(pontas_dedos['mindinho'] - np.array([marcos_mao.landmark[mp_maos.HandLandmark.PINKY_MCP].x, marcos_mao.landmark[mp_maos.HandLandmark.PINKY_MCP].y]))
                }
                
                #gesto JOINHA
                gesto_joinha = distancias_dedos['polegar'] > 0.1 and all(distancia < 0.1 for dedo, distancia in distancias_dedos.items() if dedo != 'polegar')
                
                #gesto OLA
                gesto_ola = all(distancia > 0.1 for dedo, distancia in distancias_dedos.items() if dedo in ['polegar', 'indicador']) and all(distancia < 0.1 for dedo, distancia in distancias_dedos.items() if dedo not in ['polegar', 'indicador'])
                
                #Mostrar mensagem na tela de acordo com o gesto
                if gesto_joinha:
                    cv2.putText(imagem, 'Joinha!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                elif gesto_ola:
                    cv2.putText(imagem, 'Ola!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('MediaPipe Hands', imagem)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


