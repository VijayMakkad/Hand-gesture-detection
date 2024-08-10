import cv2
import os
import pickle
import numpy as np
import mediapipe as mp
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model from pickle file
model_dict = pickle.load(open('./model.pickle', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Determine the feature length used during training
expected_feature_length = 84  # Replace with the actual length used during training

# Load class names (replace with actual class names used during training)
class_names = ['a','ThumbsUp', 'L', 'Y','Chutt']  # Update with actual class names

cap = cv2.VideoCapture(0)
while True:
    data_aux = []
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)

        # Pad or truncate the data to ensure it matches the expected feature length
        if len(data_aux) < expected_feature_length:
            data_aux = np.pad(data_aux, (0, expected_feature_length - len(data_aux)), 'constant')
        elif len(data_aux) > expected_feature_length:
            data_aux = data_aux[:expected_feature_length]

        # Make prediction
        prediction = model.predict([np.asarray(data_aux)])[0]
        
        # Check if prediction is a class label or index
        if isinstance(prediction, (int, np.int64)):
            predicted_class = class_names[int(prediction)]
        else:
            predicted_class = prediction
        
        # Display the predicted class on the frame
        cv2.putText(frame, f'Prediction: {predicted_class}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

    # Show the frame with prediction
    cv2.imshow('frame', frame)
    
    # Break loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
