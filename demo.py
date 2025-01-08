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
            THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = landmarks[1:5]
            INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = landmarks[5:9]
            MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = landmarks[9:13]
            RING_MCP, RING_PIP, RING_DIP, RING_TIP = landmarks[13:17]
            PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = landmarks[17:21]

            # Detect Palm
            if (
                    THUMB_TIP[1] < THUMB_MCP[1]
                    and INDEX_TIP[1] < INDEX_MCP[1]
                    and MIDDLE_TIP[1] < MIDDLE_MCP[1]
                    and RING_TIP[1] < RING_MCP[1]
                    and PINKY_TIP[1] < PINKY_MCP[1]
                    and abs(INDEX_TIP[0] - PINKY_TIP[0]) > w * 0.1  # Ensure fingers are spread
            ):
                if hand_label == 'Left':
                    if THUMB_TIP[0] > INDEX_TIP[0]:
                        detected_gestures.append(f"{hand_label} Palm Open")
                else:
                    if THUMB_TIP[0] < INDEX_TIP[0]:
                        detected_gestures.append(f"{hand_label} Palm Open")

            # Detect "Thumbs Down"
            elif (
                    THUMB_TIP[1] > THUMB_MCP[1]
                    and THUMB_MCP[1] > INDEX_MCP[1]
                    and THUMB_MCP[1] > MIDDLE_MCP[1]
                    and THUMB_MCP[1] > RING_MCP[1]
                    and THUMB_MCP[1] > PINKY_MCP[1]
            ):
                detected_gestures.append(f"{hand_label} Thumbs Down")

            # Detect "Thumbs Up"
            elif (
                    THUMB_TIP[1] < THUMB_MCP[1]
                    and THUMB_MCP[1] < INDEX_MCP[1]
                    and THUMB_MCP[1] < MIDDLE_MCP[1]
                    and THUMB_MCP[1] < RING_MCP[1]
                    and THUMB_MCP[1] < PINKY_MCP[1]
            ):
                detected_gestures.append(f"{hand_label} Thumbs Up")

            # Detect "Index Pointing" or "L"
            elif (
                    INDEX_TIP[1] < INDEX_MCP[1]
                    and MIDDLE_TIP[1] > MIDDLE_MCP[1]
                    and RING_TIP[1] > RING_MCP[1]
                    and PINKY_TIP[1] > PINKY_MCP[1]
            ):
                if hand_label == 'Left':
                    if THUMB_TIP[0] < INDEX_TIP[0]:
                        detected_gestures.append(f"{hand_label} Index Pointing")
                    else:
                        detected_gestures.append(f"{hand_label} L")
                else:
                    if THUMB_TIP[0] > INDEX_TIP[0]:
                        detected_gestures.append(f"{hand_label} Index Pointing")
                    else:
                        detected_gestures.append(f"{hand_label} L")

            # Detect "Fist"
            elif (
                    INDEX_TIP[1] > INDEX_MCP[1]
                    and MIDDLE_TIP[1] > MIDDLE_MCP[1]
                    and RING_TIP[1] > RING_MCP[1]
                    and PINKY_TIP[1] > PINKY_MCP[1]
            ):
                detected_gestures.append(f"{hand_label} Fist")

            # Detect "Victory" (Peace Sign)
            elif (
                    INDEX_TIP[1] < INDEX_MCP[1]
                    and MIDDLE_TIP[1] < MIDDLE_MCP[1]
                    and RING_TIP[1] > RING_MCP[1]
                    and PINKY_TIP[1] > PINKY_MCP[1]
            ):
                detected_gestures.append(f"{hand_label} Peace")

            # Detect "OK" Gesture
            elif (
                    abs(THUMB_TIP[0] - INDEX_TIP[0]) < w * 0.1
                    and abs(THUMB_TIP[1] - INDEX_TIP[1]) < h * 0.1
                    and MIDDLE_TIP[1] < MIDDLE_MCP[1]
                    and RING_TIP[1] < RING_MCP[1]
                    and PINKY_TIP[1] < PINKY_MCP[1]
            ):
                detected_gestures.append(f"{hand_label} Ok")

    # Convert BGR to RGB for Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    # Display the frame in Streamlit
    frame_placeholder.image(frame, use_container_width=True)
    gesture_placeholder.text(f"Detected Gestures: {', '.join(detected_gestures) if detected_gestures else 'None'}")

cap.release()
