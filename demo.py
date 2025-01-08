import cv2
import mediapipe as mp
import streamlit as st
from PIL import Image

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Streamlit app
st.title("Hand Gesture Detection")
st.text("Real-time hand gesture detection using MediaPipe and Streamlit.")

# Start webcam feed
run = st.checkbox("Run Hand Gesture Detection")
FRAME_SKIP = 2
FRAME_CTR = 0

# Streamlit placeholders
frame_placeholder = st.empty()
gesture_placeholder = st.empty()

cap = cv2.VideoCapture(0)

while run and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video feed.")
        break

    FRAME_CTR += 1
    if FRAME_CTR % FRAME_SKIP != 0:
        continue

    # Mirror the frame
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    detected_gestures = []

    # Draw Hand Landmarks and Perform Gesture Recognition
    if result.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get Handedness (Left or Right)
            hand_label = result.multi_handedness[hand_idx].classification[0].label  # "Left" or "Right"

            # Extract Landmark Data
            h, w, c = frame.shape
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            WRIST = landmarks[0]
            THUMB_TIP = landmarks[4]
            INDEX_TIP = landmarks[8]
            MIDDLE_TIP = landmarks[12]
            RING_TIP = landmarks[16]
            PINKY_TIP = landmarks[20]

            # Detect "Palm Open"
            if (
                THUMB_TIP[1] < landmarks[3][1] and
                INDEX_TIP[1] < landmarks[5][1] and
                MIDDLE_TIP[1] < landmarks[9][1] and
                RING_TIP[1] < landmarks[13][1] and
                PINKY_TIP[1] < landmarks[17][1]
            ):
                detected_gestures.append(f"{hand_label} Palm Open")

            # Detect "Thumbs Up"
            if (
                THUMB_TIP[1] < landmarks[3][1] and
                THUMB_TIP[1] < INDEX_TIP[1] and
                THUMB_TIP[1] < MIDDLE_TIP[1]
            ):
                detected_gestures.append(f"{hand_label} Thumbs Up")

    # Convert BGR to RGB for Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    # Display the frame in Streamlit
    frame_placeholder.image(frame, use_column_width=True)
    gesture_placeholder.text(f"Detected Gestures: {', '.join(detected_gestures) if detected_gestures else 'None'}")

cap.release()
