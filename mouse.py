
import cv2
import mediapipe as mp
import pyautogui
from pynput.mouse import Button, Controller
import math
import random

# Initialize global variables
screen_width, screen_height = pyautogui.size()
mouse = Controller()

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

mp_draw = mp.solutions.drawing_utils

# Helper functions
def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calculate_angle(a, b, c):
    """Calculate angle between three points (a, b, c) using vector math."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    cosine_angle = (
        (ba[0] * bc[0] + ba[1] * bc[1]) /
        (math.sqrt(ba[0] ** 2 + ba[1] ** 2) * math.sqrt(bc[0] ** 2 + bc[1] ** 2))
    )
    return math.degrees(math.acos(max(min(cosine_angle, 1), -1)))

def move_mouse(index_finger_tip):
    """Move the mouse pointer based on the index finger tip position."""
    if index_finger_tip:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y * screen_height)
        pyautogui.moveTo(x, y, duration=0.1)

def save_screenshot():
    """Capture and save a screenshot."""
    filename = f"screenshot_{random.randint(1, 1000)}.png"
    pyautogui.screenshot(filename)
    print(f"Screenshot saved: {filename}")

# Gesture detection
def detect_gesture(landmarks):
    """Detect gestures based on hand landmarks and trigger actions."""
    if not landmarks:
        return

    # Extract key landmarks
    index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_base = landmarks[mp_hands.HandLandmark.THUMB_CMC]
    
    # Calculate distances and angles
    thumb_index_distance = calculate_distance((thumb_tip.x, thumb_tip.y), (index_finger_tip.x, index_finger_tip.y))
    thumb_angle = calculate_angle(
        (thumb_base.x, thumb_base.y),
        (thumb_tip.x, thumb_tip.y),
        (index_finger_tip.x, index_finger_tip.y)
    )

    # Define gesture actions
    if thumb_index_distance < 0.05:
        save_screenshot()
    elif thumb_angle < 30:
        mouse.click(Button.left)
    else:
        move_mouse(index_finger_tip)

# Main application loop
def main():
    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip and process the frame
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    detect_gesture(hand_landmarks.landmark)

            # Display the frame
            cv2.imshow('Gesture Control', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
