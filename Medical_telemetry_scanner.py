import cv2
import mediapipe as mp
import time
import math
import numpy as np

# --- CONFIGURATION ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Visuals
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_TEXT = (255, 255, 255)
COLOR_WARN = (0, 0, 255)
COLOR_OK = (0, 255, 0)


class MedicalScanner:
    def __init__(self):
        self.state = "MENU"  # MENU, STABILITY, ROM, REFLEX, REPORT
        self.start_time = 0
        self.data_buffer = []
        self.results = {}

        # Test specific variables
        self.target_visible = False
        self.reaction_start = 0

    def get_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def draw_ui(self, img, text, subtext=""):
        # Dark overlay for text
        cv2.rectangle(img, (0, 0), (640, 100), (0, 0, 0), -1)
        cv2.putText(img, text, (20, 40), FONT, 0.8, COLOR_TEXT, 2)
        cv2.putText(img, subtext, (20, 80), FONT, 0.6, (200, 200, 200), 1)

    def run(self):
        cap = cv2.VideoCapture(0)

        while True:
            success, img = cap.read()
            if not success: break

            img = cv2.flip(img, 1)
            h, w, c = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(img_rgb)

            # --- STATE MACHINE ---

            # 0. MAIN MENU
            if self.state == "MENU":
                self.draw_ui(img, "DIAGNOSTIC MODE", "Press 'S' to Start Screening")
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    self.state = "STABILITY"
                    self.start_time = time.time()
                    self.data_buffer = []

            # 1. STABILITY TEST (Tremor/Parkinson's)
            elif self.state == "STABILITY":
                elapsed = time.time() - self.start_time
                countdown = 5 - int(elapsed)

                if hand_results.multi_hand_landmarks:
                    lm = hand_results.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)

                    # Track Index Tip (8) movement
                    cx, cy = int(lm.landmark[8].x * w), int(lm.landmark[8].y * h)
                    self.data_buffer.append((cx, cy))
                    cv2.circle(img, (cx, cy), 5, COLOR_WARN, -1)

                self.draw_ui(img, f"TEST 1: STABILITY ({countdown}s)", "Hold your hand COMPLETEY STILL")

                if elapsed > 5:
                    # Calculate Jitter (Standard Deviation)
                    if len(self.data_buffer) > 10:
                        x_vals = [p[0] for p in self.data_buffer]
                        y_vals = [p[1] for p in self.data_buffer]
                        jitter = np.std(x_vals) + np.std(y_vals)
                        self.results['Tremor_Score'] = round(jitter, 2)
                    else:
                        self.results['Tremor_Score'] = 0.0

                    self.state = "ROM"
                    self.start_time = time.time()
                    self.data_buffer = []  # Reset for next test

            # 2. ROM TEST (Arthritis/Mobility)
            elif self.state == "ROM":
                elapsed = time.time() - self.start_time
                countdown = 5 - int(elapsed)

                if hand_results.multi_hand_landmarks:
                    lm = hand_results.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)

                    # Calculate Pinch Distance (Thumb 4 to Index 8)
                    dist = self.get_distance(lm.landmark[4], lm.landmark[8])
                    self.data_buffer.append(dist)

                    # Visual feedback
                    cv2.line(img, (int(lm.landmark[4].x * w), int(lm.landmark[4].y * h)),
                             (int(lm.landmark[8].x * w), int(lm.landmark[8].y * h)), COLOR_OK, 2)

                self.draw_ui(img, f"TEST 2: MOBILITY ({countdown}s)", "Open and Close hand fully (FAST)")

                if elapsed > 5:
                    if len(self.data_buffer) > 0:
                        min_open = min(self.data_buffer)
                        max_open = max(self.data_buffer)
                        self.results['ROM_Score'] = round((max_open - min_open) * 100, 2)  # Normalized approx
                    else:
                        self.results['ROM_Score'] = 0

                    self.state = "REFLEX"
                    self.start_time = time.time()
                    self.target_visible = False

            # 3. REFLEX TEST (Concussion/Cognitive)
            elif self.state == "REFLEX":
                # Wait random time then show stimulus
                if not self.target_visible:
                    self.draw_ui(img, "TEST 3: REFLEX", "Wait for GREEN SCREEN then Pinch")
                    if time.time() - self.start_time > 2.0:  # 2 second delay
                        self.target_visible = True
                        self.reaction_start = time.time()
                else:
                    # Stimulus Active (Green Overlay)
                    cv2.rectangle(img, (0, 100), (w, h), (0, 255, 0), -1)
                    cv2.putText(img, "PINCH NOW!", (w // 2 - 100, h // 2), FONT, 2, (0, 0, 0), 3)

                    if hand_results.multi_hand_landmarks:
                        lm = hand_results.multi_hand_landmarks[0]
                        dist = self.get_distance(lm.landmark[4], lm.landmark[8])

                        # Detect Pinch
                        if dist < 0.05:
                            reaction_time = (time.time() - self.reaction_start) * 1000  # ms
                            self.results['Reaction_Time'] = int(reaction_time)
                            self.state = "REPORT"

            # 4. REPORT CARD
            elif self.state == "REPORT":
                # Draw Report Background
                cv2.rectangle(img, (50, 50), (w - 50, h - 50), (20, 20, 20), -1)
                cv2.rectangle(img, (50, 50), (w - 50, h - 50), (255, 255, 255), 2)

                y_start = 120
                cv2.putText(img, "DIAGNOSTIC REPORT", (w // 2 - 120, 90), FONT, 0.8, COLOR_OK, 2)

                # Analyze Tremor
                tremor = self.results.get('Tremor_Score', 0)
                t_status = "Healthy" if tremor < 5.0 else "Possible Tremor"
                cv2.putText(img, f"Stability Jitter: {tremor} ({t_status})", (80, y_start), FONT, 0.6, COLOR_TEXT, 1)

                # Analyze ROM
                rom = self.results.get('ROM_Score', 0)
                r_status = "Good" if rom > 20 else "Restricted"
                cv2.putText(img, f"Range of Motion: {rom} ({r_status})", (80, y_start + 40), FONT, 0.6, COLOR_TEXT, 1)

                # Analyze Reflex
                reflex = self.results.get('Reaction_Time', 0)
                rx_status = "Normal" if reflex < 400 else "Delayed"
                if reflex < 150: rx_status = "Impulsive/Early"
                cv2.putText(img, f"Reaction Time: {reflex}ms ({rx_status})", (80, y_start + 80), FONT, 0.6, COLOR_TEXT,
                            1)

                # Disclaimers
                cv2.putText(img, "POTENTIAL SCREENINGS:", (80, y_start + 140), FONT, 0.6, (0, 255, 255), 1)
                cv2.putText(img, "- Parkinson's / Essential Tremor (Stability)", (80, y_start + 170), FONT, 0.5,
                            (200, 200, 200), 1)
                cv2.putText(img, "- Arthritis / Carpal Tunnel (ROM)", (80, y_start + 195), FONT, 0.5, (200, 200, 200),
                            1)
                cv2.putText(img, "- Concussion / Fatigue / ADHD (Reflex)", (80, y_start + 220), FONT, 0.5,
                            (200, 200, 200), 1)

                cv2.putText(img, "NOT MEDICAL ADVICE. FOR RESEARCH ONLY.", (w // 2 - 180, h - 70), FONT, 0.5,
                            (0, 0, 255), 1)
                cv2.putText(img, "Press 'R' to Restart", (w // 2 - 80, h - 30), FONT, 0.6, COLOR_OK, 1)

                if cv2.waitKey(1) & 0xFF == ord('r'):
                    self.state = "MENU"

            cv2.imshow("Bio-Telemetry Scanner", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = MedicalScanner()
    app.run()