# Hand Gesture Recognition with OpenCV and MediaPipe

This project uses **OpenCV** and **MediaPipe Hands** to recognize and display basic hand gestures in real-time using a webcam. The script detects gestures such as "Palm Open," "Thumbs Up," "Thumbs Down," "Index Pointing," "L," "Fist," "Victory (Peace Sign)," and "OK" gestures. It also identifies whether the detected hand is "Left" or "Right."

---
## Features
- Real-time hand tracking and gesture recognition.
- Detects basic hand gestures with classification as "Left" or "Right."
- Draws landmarks and hand connections on the detected hands.
- Provides visual feedback by displaying recognized gestures on the video frame.

---
## How It Works
1. Video Capture: The script uses OpenCV to capture frames from your webcam (cv2.VideoCapture).
2. Hand Detection: MediaPipe's Hands solution detects hands in each frame and provides 21 hand landmarks for each detected hand.
3. Gesture Recognition: Specific hand gestures are recognized based on the relative positions of landmarks. For example:
   - Palm Open: All fingers are extended, and the thumb and pinky are sufficiently spread apart.
   - Thumbs Up: Thumb is raised above the other fingers.
   - Victory (Peace Sign): Index and middle fingers are extended while others are folded.
4. Display Feedback: Detected gestures and the hand label ("Left" or "Right") are displayed on the video feed using OpenCV's cv2.putText.
5. Optimization: The script processes every other frame (FRAME_SKIP = 2) to reduce computational load.

---
## Gesture Details
The script recognizes the following gestures:
- Palm Open: All fingers extended and spread.
- Thumbs Up: Thumb raised above all other fingers.
- Thumbs Down: Thumb lowered below all other fingers.
- Index Pointing: Only the index finger extended.
- L Gesture: Thumb and index finger extended in an "L" shape.
- Fist: All fingers folded.
- Victory (Peace Sign): Index and middle fingers extended.
- OK Gesture: Thumb and index finger touch to form a circle.

