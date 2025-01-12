import cv2
import mediapipe as mp

FRAME_CTR = 0
FRAME_SKIP = 2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start Video Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    FRAME_CTR += 1
    if FRAME_CTR % FRAME_SKIP != 0:
        continue

    # Mirror the frame
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

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

            # Set text position based on hand label
            text_x = 10
            text_y = 50 if hand_label == "Left" else 100

            # Detect "Palm Open"
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
                        cv2.putText(frame, f"{hand_label} Palm", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    if THUMB_TIP[0] < INDEX_TIP[0]:
                        cv2.putText(frame, f"{hand_label} Palm", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Detect "Thumbs Down"
            elif (
                THUMB_TIP[1] > THUMB_MCP[1]
                and THUMB_MCP[1] > INDEX_MCP[1]
                and THUMB_MCP[1] > MIDDLE_MCP[1]
                and THUMB_MCP[1] > RING_MCP[1]
                and THUMB_MCP[1] > PINKY_MCP[1]
            ):
                cv2.putText(frame, f"{hand_label} Thumbs Down", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Detect "Thumbs Up"
            elif (
                THUMB_TIP[1] < THUMB_MCP[1]
                and THUMB_MCP[1] < INDEX_MCP[1]
                and THUMB_MCP[1] < MIDDLE_MCP[1]
                and THUMB_MCP[1] < RING_MCP[1]
                and THUMB_MCP[1] < PINKY_MCP[1]
            ):
                cv2.putText(frame, f"{hand_label} Thumbs Up", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Detect "Index Pointing" or "L"
            elif (
                INDEX_TIP[1] < INDEX_MCP[1]
                and MIDDLE_TIP[1] > MIDDLE_MCP[1]
                and RING_TIP[1] > RING_MCP[1]
                and PINKY_TIP[1] > PINKY_MCP[1]
            ):
                if hand_label == 'Left':
                    if THUMB_TIP[0] < INDEX_TIP[0]:
                        cv2.putText(frame, f"{hand_label} Index Pointing", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    else:
                        cv2.putText(frame, f"{hand_label} L", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                else:
                    if THUMB_TIP[0] > INDEX_TIP[0]:
                        cv2.putText(frame, f"{hand_label} Index Pointing", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    else:
                        cv2.putText(frame, f"{hand_label} L", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Detect "Fist"
            elif (
                INDEX_TIP[1] > INDEX_MCP[1]
                and MIDDLE_TIP[1] > MIDDLE_MCP[1]
                and RING_TIP[1] > RING_MCP[1]
                and PINKY_TIP[1] > PINKY_MCP[1]
            ):
                cv2.putText(frame, f"{hand_label} Fist", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Detect "Victory" (Peace Sign)
            elif (
                INDEX_TIP[1] < INDEX_MCP[1]
                and MIDDLE_TIP[1] < MIDDLE_MCP[1]
                and RING_TIP[1] > RING_MCP[1]
                and PINKY_TIP[1] > PINKY_MCP[1]
            ):
                cv2.putText(frame, f"{hand_label} Peace", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Detect "OK" Gesture
            elif (
                abs(THUMB_TIP[0] - INDEX_TIP[0]) < w * 0.1
                and abs(THUMB_TIP[1] - INDEX_TIP[1]) < h * 0.1
                and MIDDLE_TIP[1] < MIDDLE_MCP[1]
                and RING_TIP[1] < RING_MCP[1]
                and PINKY_TIP[1] < PINKY_MCP[1]
            ):
                cv2.putText(frame, f"{hand_label} OK", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Display Frame
    cv2.imshow('Hand Gesture Recognition (Mirrored)', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
