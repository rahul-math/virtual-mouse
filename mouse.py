import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
from pathlib import Path
from pynput.mouse import Button, Listener as MouseListener
from pynput import mouse
from datetime import datetime
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure pyautogui for safety
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

class GestureMouseController:
    def __init__(self):
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            max_num_hands=1
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Mouse controller
        self.mouse_controller = mouse.Controller()
        
        # Gesture state management
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.5  # 500ms cooldown between gestures
        self.last_click_time = 0
        self.click_cooldown = 0.3  # 300ms cooldown between clicks
        
        # Mouse movement smoothing
        self.mouse_history = []
        self.smoothing_factor = 0.7
        self.movement_threshold = 5
        
        # Create screenshots directory
        self.screenshots_dir = Path("screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)
        
        # Gesture detection parameters
        self.gesture_params = {
            'angle_threshold': 50,
            'distance_threshold': 50,
            'movement_sensitivity': 1.2
        }
        
        logger.info("GestureMouseController initialized successfully")
    
    def get_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        try:
            # Vector from point2 to point1
            v1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
            # Vector from point2 to point3
            v2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
            
            # Calculate angle using dot product
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)  # Ensure valid range
            angle = math.degrees(math.acos(cos_angle))
            
            return angle
        except (ZeroDivisionError, ValueError):
            return 180  # Default angle if calculation fails
    
    def smooth_mouse_movement(self, x, y):
        """Apply smoothing to mouse movement"""
        self.mouse_history.append((x, y))
        
        # Keep only recent history
        if len(self.mouse_history) > 5:
            self.mouse_history.pop(0)
        
        # Calculate weighted average
        if len(self.mouse_history) >= 2:
            weights = np.linspace(0.1, 1.0, len(self.mouse_history))
            avg_x = np.average([pos[0] for pos in self.mouse_history], weights=weights)
            avg_y = np.average([pos[1] for pos in self.mouse_history], weights=weights)
            
            return int(avg_x), int(avg_y)
        
        return x, y
    
    def find_finger_tip(self, processed):
        """Find index finger tip coordinates"""
        if processed.multi_hand_landmarks:
            hand_landmarks = processed.multi_hand_landmarks[0]
            index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            return index_finger_tip
        return None
    
    def move_mouse(self, index_finger_tip):
        """Move mouse based on finger tip position with smoothing"""
        if index_finger_tip is not None:
            # Convert normalized coordinates to screen coordinates
            x = int(index_finger_tip.x * self.screen_width)
            y = int(index_finger_tip.y * self.screen_height)
            
            # Apply smoothing
            smooth_x, smooth_y = self.smooth_mouse_movement(x, y)
            
            # Only move if movement is significant enough
            current_pos = self.mouse_controller.position
            distance = self.get_distance(current_pos, (smooth_x, smooth_y))
            
            if distance > self.movement_threshold:
                try:
                    pyautogui.moveTo(smooth_x, smooth_y, duration=0.01)
                except pyautogui.FailSafeException:
                    logger.warning("PyAutoGUI fail-safe triggered")
    
    def is_gesture_allowed(self):
        """Check if enough time has passed since last gesture"""
        current_time = time.time()
        return current_time - self.last_gesture_time > self.gesture_cooldown
    
    def is_click_allowed(self):
        """Check if enough time has passed since last click"""
        current_time = time.time()
        return current_time - self.last_click_time > self.click_cooldown
    
    def is_left_click(self, landmark_list, thumb_index_dist):
        """Detect left click gesture (index finger extended, others folded)"""
        try:
            index_extended = self.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < self.gesture_params['angle_threshold']
            middle_folded = self.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90
            thumb_separated = thumb_index_dist > self.gesture_params['distance_threshold']
            
            return index_extended and middle_folded and thumb_separated
        except (IndexError, TypeError):
            return False
    
    def is_right_click(self, landmark_list, thumb_index_dist):
        """Detect right click gesture (middle finger extended, index folded)"""
        try:
            middle_extended = self.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < self.gesture_params['angle_threshold']
            index_folded = self.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90
            thumb_separated = thumb_index_dist > self.gesture_params['distance_threshold']
            
            return middle_extended and index_folded and thumb_separated
        except (IndexError, TypeError):
            return False
    
    def is_double_click(self, landmark_list, thumb_index_dist):
        """Detect double click gesture (both index and middle extended)"""
        try:
            index_extended = self.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < self.gesture_params['angle_threshold']
            middle_extended = self.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < self.gesture_params['angle_threshold']
            thumb_separated = thumb_index_dist > self.gesture_params['distance_threshold']
            
            return index_extended and middle_extended and thumb_separated
        except (IndexError, TypeError):
            return False
    
    def is_screenshot(self, landmark_list, thumb_index_dist):
        """Detect screenshot gesture (index and middle extended with thumb close)"""
        try:
            index_extended = self.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < self.gesture_params['angle_threshold']
            middle_extended = self.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < self.gesture_params['angle_threshold']
            thumb_close = thumb_index_dist < self.gesture_params['distance_threshold']
            
            return index_extended and middle_extended and thumb_close
        except (IndexError, TypeError):
            return False
    
    def is_mouse_mode(self, landmark_list):
        """Detect mouse movement mode (thumb and index close, index extended)"""
        try:
            thumb_index_close = self.get_distance(landmark_list[4], landmark_list[5]) < self.gesture_params['distance_threshold']
            index_extended = self.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90
            
            return thumb_index_close and index_extended
        except (IndexError, TypeError):
            return False
    
    def execute_click(self, button, frame, text, color):
        """Execute a mouse click with visual feedback"""
        if self.is_click_allowed():
            try:
                self.mouse_controller.press(button)
                self.mouse_controller.release(button)
                self.last_click_time = time.time()
                cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                logger.info(f"Executed: {text}")
            except Exception as e:
                logger.error(f"Click execution failed: {e}")
    
    def take_screenshot(self, frame):
        """Take a screenshot with timestamp"""
        if self.is_gesture_allowed():
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = self.screenshots_dir / f"screenshot_{timestamp}.png"
                
                screenshot = pyautogui.screenshot()
                screenshot.save(str(filename))
                
                self.last_gesture_time = time.time()
                cv2.putText(frame, f"Screenshot: {filename.name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                logger.info(f"Screenshot saved: {filename}")
            except Exception as e:
                logger.error(f"Screenshot failed: {e}")
    
    def detect_gesture(self, frame, landmark_list, processed):
        """Main gesture detection and execution logic"""
        if len(landmark_list) < 21:
            return
        
        try:
            index_finger_tip = self.find_finger_tip(processed)
            thumb_index_dist = self.get_distance(landmark_list[4], landmark_list[5])
            
            # Mouse movement mode
            if self.is_mouse_mode(landmark_list):
                self.move_mouse(index_finger_tip)
                cv2.putText(frame, "Mouse Mode", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Gesture detection
            elif self.is_left_click(landmark_list, thumb_index_dist):
                self.execute_click(Button.left, frame, "Left Click", (0, 255, 0))
            
            elif self.is_right_click(landmark_list, thumb_index_dist):
                self.execute_click(Button.right, frame, "Right Click", (0, 0, 255))
            
            elif self.is_double_click(landmark_list, thumb_index_dist):
                if self.is_click_allowed():
                    pyautogui.doubleClick()
                    self.last_click_time = time.time()
                    cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    logger.info("Executed: Double Click")
            
            elif self.is_screenshot(landmark_list, thumb_index_dist):
                self.take_screenshot(frame)
                
        except Exception as e:
            logger.error(f"Gesture detection error: {e}")
    
    def add_ui_elements(self, frame):
        """Add UI elements to the frame"""
        # Add instructions
        instructions = [
            "Gestures:",
            "Thumb+Index close: Mouse mode",
            "Index extended: Left click",
            "Middle extended: Right click",
            "Both extended (thumb far): Double click",
            "Both extended (thumb close): Screenshot",
            "Press 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, frame.shape[0] - 20 - (len(instructions) - i - 1) * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add frame rate
        cv2.putText(frame, f"FPS: {int(cv2.getTickFrequency() / (cv2.getTickCount() - self.last_gesture_time)) if self.last_gesture_time > 0 else 0}", 
                   (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def run(self):
        """Main execution loop"""
        cap = cv2.VideoCapture(0)
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            logger.error("Cannot open camera")
            return
        
        logger.info("Starting gesture control. Press 'q' to quit.")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                processed = self.hands.process(frame_rgb)
                
                # Extract landmarks
                landmark_list = []
                if processed.multi_hand_landmarks:
                    hand_landmarks = processed.multi_hand_landmarks[0]
                    
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract landmark coordinates
                    for landmark in hand_landmarks.landmark:
                        landmark_list.append((landmark.x, landmark.y))
                
                # Detect and execute gestures
                self.detect_gesture(frame, landmark_list, processed)
                
                # Add UI elements
                self.add_ui_elements(frame)
                
                # Display frame
                cv2.imshow('Gesture Mouse Control', frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Gesture control stopped")

def main():
    """Main function"""
    try:
        controller = GestureMouseController()
        controller.run()
    except Exception as e:
        logger.error(f"Failed to start application: {e}")

if __name__ == '__main__':
    main()
