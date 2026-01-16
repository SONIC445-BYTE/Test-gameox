import pygame
import cv2
import mediapipe as mp
import random
import math
import numpy as np
import time
import os


# --- 1. ROBUST CAMERA INITIALIZATION ---
def get_working_camera():
    for i in range(3):
        print(f"Testing Camera Index {i}...")
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Force DirectShow
        if cap.isOpened():
            time.sleep(1.0)  # Warmup
            ret, frame = cap.read()
            if ret:
                print(f"✅ Success! Using Camera Index {i}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap
            else:
                cap.release()
    return None


# --- 2. CONFIGURATION ---
pygame.init()
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
except:
    print("Audio disabled.")

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Bio-Metric Assessment Hub (Eye + Hand)")
clock = pygame.time.Clock()

# Camera
cap = get_working_camera()
camera_active = (cap is not None)

# --- MEDIAPIPE SETUP (HANDS + FACE) ---
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
# Refine landmarks=True gives us detailed Iris/Eye tracking
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
DARK_GREY = (20, 20, 20)
GREY = (100, 100, 100)

font_ui = pygame.font.Font(None, 30)
font_title = pygame.font.Font(None, 50)
font_report = pygame.font.SysFont("Courier New", 16)


# --- 3. SOUND ENGINE ---
def generate_sound(freq_start, freq_end, duration, vol=0.5):
    try:
        sample_rate = 44100
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, False)
        freqs = np.linspace(freq_start, freq_end, n_samples)
        waveform = np.sin(2 * np.pi * freqs * t) * vol
        waveform *= np.linspace(1.0, 0.0, n_samples)
        waveform = (waveform * 32767).astype(np.int16)
        stereo = np.column_stack((waveform, waveform)).copy(order='C')
        return pygame.sndarray.make_sound(stereo)
    except:
        return None


snd_shoot = generate_sound(800, 200, 0.15, 0.3)
snd_hit = generate_sound(100, 50, 0.2, 0.5)
snd_scan = generate_sound(300, 600, 0.1, 0.1)  # Scanning sound
snd_start = generate_sound(400, 800, 0.5, 0.4)


def play_snd(sound):
    if sound: sound.play()


# --- 4. DIAGNOSTIC ENGINE (UPDATED) ---
class HealthMonitor:
    def __init__(self):
        self.start_time = time.time()

        # Motor
        self.hand_positions = []
        self.tremor_scores = []
        self.pinch_history = []
        self.enemy_spawn_times = {}
        self.reaction_times = []
        self.shots_fired = 0
        self.shots_hit = 0

        # Ocular (Eye)
        self.blink_count = 0
        self.eye_closed_frames = 0
        self.head_rotation_scores = []  # Stability of head
        self.is_blinking = False

    # --- MOTOR UPDATES ---
    def update_movement(self, x, y):
        self.hand_positions.append((x, y))
        if len(self.hand_positions) > 30: self.hand_positions.pop(0)
        if len(self.hand_positions) > 10:
            x_vals = [p[0] for p in self.hand_positions]
            y_vals = [p[1] for p in self.hand_positions]
            jitter = float(np.std(x_vals) + np.std(y_vals))
            if jitter < 25: self.tremor_scores.append(jitter)

    def update_rom(self, d):
        self.pinch_history.append(d)

    def log_shot(self):
        self.shots_fired += 1

    def log_hit(self):
        self.shots_hit += 1

    def log_spawn(self, eid):
        self.enemy_spawn_times[eid] = time.time()

    def calculate_reflex(self, tid):
        now = time.time()
        if tid in self.enemy_spawn_times:
            reaction = (now - self.enemy_spawn_times[tid]) * 1000
            if reaction < 3000: self.reaction_times.append(reaction)
            del self.enemy_spawn_times[tid]

    # --- OCULAR UPDATES ---
    def update_eyes(self, landmarks, w, h):
        # 1. Blink Detection (EAR - Eye Aspect Ratio)
        # Left Eye Indices: 33, 160, 158, 133, 153, 144
        # Vertical Dist / Horizontal Dist
        def get_ear(eye_indices):
            p1 = landmarks[eye_indices[1]]
            p2 = landmarks[eye_indices[5]]
            p3 = landmarks[eye_indices[2]]
            p4 = landmarks[eye_indices[4]]
            p0 = landmarks[eye_indices[0]]  # Corner
            p5 = landmarks[eye_indices[3]]  # Corner

            # Vertical
            v1 = math.hypot(p1.x - p5.x, p1.y - p5.y)  # Simplified logic for speed
            v_dist = abs(p1.y - p5.y) + abs(p2.y - p4.y)  # Approx vertical
            h_dist = abs(p0.x - p3.x)
            return v_dist / (2.0 * h_dist) if h_dist > 0 else 0

        left_ear = get_ear([33, 160, 158, 133, 153, 144])
        right_ear = get_ear([362, 385, 387, 263, 373, 380])
        avg_ear = (left_ear + right_ear) / 2.0

        # EAR Threshold for Blink is usually 0.02 - 0.03 in MediaPipe simplified
        # Adjusting threshold for 2D distances
        blink_thresh = 0.08

        if avg_ear < blink_thresh:
            self.eye_closed_frames += 1
        else:
            if self.eye_closed_frames > 1:  # Valid blink (not just flicker)
                self.blink_count += 1
                self.is_blinking = True
            self.eye_closed_frames = 0
            self.is_blinking = False

        # 2. Head Stability (Nose Tracking)
        nose = landmarks[1]
        self.head_rotation_scores.append((nose.x, nose.y))

    def generate_report(self):
        # Calculations
        avg_jitter = float(np.mean(self.tremor_scores)) if self.tremor_scores else 0.0
        avg_reflex = float(np.mean(self.reaction_times)) if self.reaction_times else 0.0
        accuracy = (self.shots_hit / self.shots_fired * 100) if self.shots_fired > 0 else 0.0

        duration_min = (time.time() - self.start_time) / 60.0
        if duration_min < 0.1: duration_min = 0.1
        bpm = self.blink_count / duration_min  # Blinks Per Minute

        # Screenings
        screenings = []
        if avg_jitter > 5.0: screenings.append("• Parkinson's/Tremor: POSITIVE")
        if avg_reflex > 450: screenings.append("• Concussion/Fatigue: POSITIVE")
        if avg_reflex < 150 and avg_reflex > 0: screenings.append("• ADHD (Impulsivity): POSITIVE")
        if bpm < 10: screenings.append("• Computer Vision Syndrome (Low Blink): POSITIVE")
        if bpm > 40: screenings.append("• Ocular Fatigue/Dryness (High Blink): POSITIVE")
        if not screenings: screenings.append("• No significant anomalies detected.")

        return [
            "--- DIAGNOSTIC REPORT ---",
            "MOTOR METRICS:",
            f" Stability:       {avg_jitter:.2f} (Lower is better)",
            f" Reaction Time:   {int(avg_reflex)}ms",
            f" Accuracy:        {int(accuracy)}%",
            "",
            "OCULAR METRICS:",
            f" Blink Rate:      {int(bpm)} per min",
            f" Focus Status:    {'Strained' if bpm < 10 else 'Normal'}",
            "",
            "SCREENING FLAGS:",
            *screenings,
            "",
            "NOTE: Research purposes only."
        ]


# --- 5. GAME VARIABLES ---
game_state = "START"
doctor = HealthMonitor()
score = 0
lives = 3
bullets = []
enemies = []
enemy_counter = 0
last_shot = 0

# Cursor & Eye Tracking
hand_x, hand_y = WIDTH // 2, HEIGHT // 2
prev_hand_x, prev_hand_y = WIDTH // 2, HEIGHT // 2
SMOOTHING = 0.5
pinching = False
pinch_strength = 0.0
eye_lock_timer = 0  # To count how long user stares at start

# --- 6. MAIN LOOP ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN and game_state == "GAMEOVER":
            if event.key == pygame.K_r:
                score, lives, enemies, bullets = 0, 3, [], []
                doctor = HealthMonitor()
                game_state = "START"  # Go back to eye check
                eye_lock_timer = 0

    # --- WEBCAM PROCESSING ---
    face_detected = False

    if camera_active:
        success, img = cap.read()
        if success:
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 1. PROCESS HANDS
            res_hands = hands.process(img_rgb)
            if res_hands.multi_hand_landmarks:
                lm = res_hands.multi_hand_landmarks[0]

                # Cursor Logic
                tx, ty = lm.landmark[8].x * WIDTH, lm.landmark[8].y * HEIGHT
                nx, ny = prev_hand_x + (tx - prev_hand_x) * SMOOTHING, prev_hand_y + (ty - prev_hand_y) * SMOOTHING
                prev_hand_x, prev_hand_y = nx, ny
                hand_x, hand_y = int(nx), int(ny)

                # Pinch Logic
                x1, y1 = lm.landmark[8].x, lm.landmark[8].y
                x2, y2 = lm.landmark[4].x, lm.landmark[4].y
                t_dist = math.hypot(x2 - x1, y2 - y1)
                p_size = math.hypot(lm.landmark[9].x - lm.landmark[0].x, lm.landmark[9].y - lm.landmark[0].y)
                ratio = t_dist / p_size if p_size > 0 else 1.0
                pinching = ratio < 0.25
                pinch_strength = max(0.0, min(1.0, 1.0 - (ratio / 0.5)))

                if game_state == "PLAYING":
                    doctor.update_movement(hand_x, hand_y)
                    doctor.update_rom(t_dist)

            # 2. PROCESS EYES (FACE MESH)
            res_face = face_mesh.process(img_rgb)
            if res_face.multi_face_landmarks:
                face_lm = res_face.multi_face_landmarks[0]
                face_detected = True

                if game_state == "PLAYING":
                    doctor.update_eyes(face_lm.landmark, WIDTH, HEIGHT)

                # Draw Eyes on Preview
                # Left Eye Center approx index 468 (Iris)
                if len(face_lm.landmark) > 468:
                    ir_x = int(face_lm.landmark[468].x * 200) + (WIDTH - 210)
                    ir_y = int(face_lm.landmark[468].y * 150) + (HEIGHT - 160)
                    # We can store this coordinate to draw later

    screen.fill(BLACK)

    # --- STATE: START (RETINAL LOCK) ---
    if game_state == "START":
        # Draw Scanning UI
        title = font_title.render("RETINAL SCAN REQUIRED", True, CYAN)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 100))

        # Draw Face Placeholder
        center_x, center_y = WIDTH // 2, HEIGHT // 2
        pygame.draw.rect(screen, DARK_GREY, (center_x - 100, center_y - 100, 200, 200), 2)
        pygame.draw.line(screen, DARK_GREY, (center_x - 20, center_y), (center_x + 20, center_y), 2)
        pygame.draw.line(screen, DARK_GREY, (center_x, center_y - 20), (center_x, center_y + 20), 2)

        if face_detected:
            eye_lock_timer += 1
            # Loading Bar
            bar_w = min(200, eye_lock_timer * 4)
            pygame.draw.rect(screen, GREEN, (center_x - 100, center_y + 120, bar_w, 10))

            status = font_ui.render("SUBJECT IDENTIFIED... HOLD STEADY", True, GREEN)
            screen.blit(status, (WIDTH // 2 - status.get_width() // 2, center_y + 140))

            if eye_lock_timer % 10 == 0: play_snd(snd_scan)  # Beep

            if eye_lock_timer > 60:  # ~1 second hold
                play_snd(snd_start)
                game_state = "PLAYING"
                doctor = HealthMonitor()  # Reset doctor stats
        else:
            eye_lock_timer = 0
            warn = font_ui.render("Look at the Camera to Start", True, RED)
            screen.blit(warn, (WIDTH // 2 - warn.get_width() // 2, center_y + 140))

    # --- STATE: PLAYING ---
    elif game_state == "PLAYING":
        # Logic
        now = pygame.time.get_ticks()
        if pinching and now - last_shot > 200:
            bullets.append([hand_x, hand_y])
            last_shot = now
            doctor.log_shot()
            play_snd(snd_shoot)

        for b in bullets[:]:
            b[1] -= 20
            pygame.draw.circle(screen, GREEN, (int(b[0]), int(b[1])), 5)
            if b[1] < 0: bullets.remove(b)

        if random.randint(1, 60) == 1:
            enemy_counter += 1
            enemies.append([random.randint(50, WIDTH - 50), -50, random.randint(3, 6), 30, enemy_counter])
            doctor.log_spawn(enemy_counter)

        for e in enemies[:]:
            e[1] += e[2]
            # Draw Alien
            ex, ey = int(e[0]), int(e[1])
            pygame.draw.circle(screen, RED, (ex, ey), 30)
            pygame.draw.circle(screen, WHITE, (ex - 12, ey - 8), 8)  # Left Eye
            pygame.draw.circle(screen, WHITE, (ex + 12, ey - 8), 8)  # Right Eye
            pygame.draw.circle(screen, BLACK, (ex - 12, ey - 8), 3)  # Pupil
            pygame.draw.circle(screen, BLACK, (ex + 12, ey - 8), 3)  # Pupil

            rect = pygame.Rect(ex - 30, ey - 30, 60, 60)
            hit = False
            for b in bullets[:]:
                if rect.collidepoint(int(b[0]), int(b[1])):
                    enemies.remove(e)
                    bullets.remove(b)
                    score += 10
                    doctor.log_hit()
                    doctor.calculate_reflex(e[4])
                    play_snd(snd_hit)
                    hit = True
                    break
            if not hit and e[1] > HEIGHT:
                enemies.remove(e)
                lives -= 1
                if lives <= 0: game_state = "GAMEOVER"

        # Cursor
        color = RED if pinching else CYAN
        pygame.draw.circle(screen, color, (hand_x, hand_y), 15, 2)
        # Pinch Bar
        pygame.draw.rect(screen, GREY, (hand_x - 20, hand_y + 30, 40, 6))
        pygame.draw.rect(screen, GREEN if pinching else YELLOW, (hand_x - 20, hand_y + 30, int(40 * pinch_strength), 6))

        # HUD
        screen.blit(font_ui.render(f"Score: {score}", True, WHITE), (10, 10))
        screen.blit(font_ui.render(f"Lives: {lives}", True, RED), (WIDTH - 100, 10))

    # --- STATE: GAMEOVER ---
    elif game_state == "GAMEOVER":
        screen.fill(DARK_GREY)
        report = doctor.generate_report()

        # Report Card
        pygame.draw.rect(screen, BLACK, (50, 50, WIDTH - 100, HEIGHT - 100))
        pygame.draw.rect(screen, CYAN, (50, 50, WIDTH - 100, HEIGHT - 100), 2)

        head = font_title.render("MEDICAL SUMMARY", True, WHITE)
        screen.blit(head, (WIDTH // 2 - head.get_width() // 2, 70))

        dy = 130
        for line in report:
            c = WHITE
            if "POSITIVE" in line:
                c = RED
            elif "METRICS" in line:
                c = CYAN
            elif "No significant" in line:
                c = GREEN
            screen.blit(font_report.render(line, True, c), (80, dy))
            dy += 25

        rest = font_ui.render("Press 'R' for Re-Assessment", True, YELLOW)
        screen.blit(rest, (WIDTH // 2 - rest.get_width() // 2, HEIGHT - 60))

    # --- REFERENCE FRAME (Preview) ---
    if 'img' in locals() and success:
        try:
            prev = cv2.resize(img, (200, 150))
            prev = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
            surf = pygame.image.frombuffer(prev.tobytes(), prev.shape[1::-1], "RGB")
            screen.blit(surf, (WIDTH - 210, HEIGHT - 160))
            pygame.draw.rect(screen, CYAN, (WIDTH - 210, HEIGHT - 160, 200, 150), 2)

            # Eye Tracking Overlay on Preview
            if face_detected:
                txt = font_report.render("EYE TRACKING: ON", True, GREEN)
                screen.blit(txt, (WIDTH - 200, HEIGHT - 180))
        except:
            pass

    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()