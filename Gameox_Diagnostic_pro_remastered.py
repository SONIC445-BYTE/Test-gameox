import pygame
import cv2
import mediapipe as mp
import random
import math
import numpy as np
import time


# --- 1. CAMERA INITIALIZATION ---
def get_working_camera():
    for i in range(3):
        print(f"Testing Camera Index {i}...")
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            time.sleep(1.0)
            ret, frame = cap.read()
            if ret:
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
    pass

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neuro-Motor Elite Assessment Hub")
clock = pygame.time.Clock()

cap = get_working_camera()
camera_active = (cap is not None)

# MediaPipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
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
TRANS_GREEN = (0, 255, 0, 50)

font_ui = pygame.font.Font(None, 30)
font_tech = pygame.font.SysFont("Courier New", 20, bold=True)  # Tech font
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
snd_scan = generate_sound(300, 600, 0.05, 0.1)
snd_boot = generate_sound(100, 1000, 1.0, 0.3)
snd_alert = generate_sound(800, 800, 0.1, 0.2)


def play_snd(sound):
    if sound: sound.play()


# --- 4. DIAGNOSTIC ENGINE ---
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
        # Ocular
        self.blink_count = 0
        self.eye_closed_frames = 0
        self.is_blinking = False

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

    def update_eyes(self, landmarks):
        # Blink Detection
        def get_ear(eye_indices):
            p1, p5 = landmarks[eye_indices[1]], landmarks[eye_indices[3]]
            p2, p4 = landmarks[eye_indices[5]], landmarks[eye_indices[4]]
            p0, p3 = landmarks[eye_indices[0]], landmarks[eye_indices[2]]
            v_dist = abs(p1.y - p5.y) + abs(p2.y - p4.y)
            h_dist = abs(p0.x - p3.x)
            return v_dist / (2.0 * h_dist) if h_dist > 0 else 0

        left_ear = get_ear([33, 160, 158, 133, 153, 144])
        right_ear = get_ear([362, 385, 387, 263, 373, 380])
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < 0.08:
            self.eye_closed_frames += 1
        else:
            if self.eye_closed_frames > 1:
                self.blink_count += 1
            self.eye_closed_frames = 0

    def generate_report(self):
        avg_jitter = float(np.mean(self.tremor_scores)) if self.tremor_scores else 0.0
        avg_reflex = float(np.mean(self.reaction_times)) if self.reaction_times else 0.0
        accuracy = (self.shots_hit / self.shots_fired * 100) if self.shots_fired > 0 else 0.0
        duration_min = max(0.1, (time.time() - self.start_time) / 60.0)
        bpm = self.blink_count / duration_min

        screenings = []
        if avg_jitter > 5.0: screenings.append("• Parkinson's/Tremor: POSITIVE")
        if avg_reflex > 450: screenings.append("• Concussion/Fatigue: POSITIVE")
        if avg_reflex < 150 and avg_reflex > 0: screenings.append("• ADHD (Impulsivity): POSITIVE")
        if bpm < 10: screenings.append("• Computer Vision Syndrome: POSITIVE")
        if bpm > 40: screenings.append("• Ocular Fatigue: POSITIVE")
        if not screenings: screenings.append("• No significant anomalies detected.")

        return [
            "--- DIAGNOSTIC REPORT ---",
            "MOTOR METRICS:",
            f" Stability:       {avg_jitter:.2f}",
            f" Reaction Time:   {int(avg_reflex)}ms",
            f" Accuracy:        {int(accuracy)}%",
            "",
            "OCULAR METRICS:",
            f" Blink Rate:      {int(bpm)} per min",
            "",
            "SCREENING FLAGS:",
            *screenings,
            "",
            "NOTE: Research purposes only."
        ]


# --- 5. GAME VARIABLES ---
game_state = "BOOT"  # New starting state
doctor = HealthMonitor()
score = 0
lives = 3
bullets, enemies = [], []
enemy_counter = 0
last_shot = 0

# Cursor
hand_x, hand_y = WIDTH // 2, HEIGHT // 2
prev_hand_x, prev_hand_y = WIDTH // 2, HEIGHT // 2
SMOOTHING = 0.5
pinching, pinch_strength = False, 0.0
eye_lock_timer = 0
boot_timer = 0  # For boot sequence

# --- 6. MAIN LOOP ---
running = True
# Play Boot Sound
play_snd(snd_boot)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN and game_state == "GAMEOVER":
            if event.key == pygame.K_r:
                score, lives, enemies, bullets = 0, 3, [], []
                doctor = HealthMonitor()
                game_state = "START"
                eye_lock_timer = 0

    # --- WEBCAM PROCESSING ---
    face_detected = False
    hand_detected = False

    if camera_active:
        success, img = cap.read()
        if success:
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # A. PROCESS HANDS
            res_hands = hands.process(img_rgb)
            if res_hands.multi_hand_landmarks:
                hand_detected = True
                lm = res_hands.multi_hand_landmarks[0]

                # Update Hand Pos
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

            # B. PROCESS EYES
            res_face = face_mesh.process(img_rgb)
            if res_face.multi_face_landmarks:
                face_lm = res_face.multi_face_landmarks[0]
                face_detected = True
                if game_state == "PLAYING": doctor.update_eyes(face_lm.landmark)

    screen.fill(BLACK)

    # --- STATE: BOOT SEQUENCE (New!) ---
    if game_state == "BOOT":
        boot_timer += 1

        # 1. Show Raw Camera Feed as background
        if 'img' in locals():
            bg_frame = cv2.resize(img, (WIDTH, HEIGHT))
            bg_frame = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)
            bg_surf = pygame.image.frombuffer(bg_frame.tobytes(), bg_frame.shape[1::-1], "RGB")
            # Darken it
            bg_surf.set_alpha(100)
            screen.blit(bg_surf, (0, 0))

        # 2. Draw Tech Overlay
        pygame.draw.rect(screen, CYAN, (50, 50, WIDTH - 100, HEIGHT - 100), 2)
        pygame.draw.line(screen, CYAN, (WIDTH // 2, 50), (WIDTH // 2, HEIGHT - 50), 1)
        pygame.draw.line(screen, CYAN, (50, HEIGHT // 2), (WIDTH - 50, HEIGHT // 2), 1)

        # 3. Scrolling Text
        lines = ["INITIALIZING NEURAL LINK...", "LOADING BIOMETRIC DRIVERS...", "CALIBRATING OPTICAL SENSORS...",
                 "ESTABLISHING BASELINE..."]
        y_pos = 100
        for i, line in enumerate(lines):
            if boot_timer > i * 60:  # Reveal line every second
                txt = font_tech.render(f"> {line} [OK]", True, GREEN)
                screen.blit(txt, (70, y_pos))
                y_pos += 40

        # 4. Check Connection
        status_y = HEIGHT - 100
        if boot_timer > 240:
            cam_status = "CAMERA: ONLINE" if camera_active else "CAMERA: ERROR"
            face_status = "FACE: DETECTED" if face_detected else "FACE: SEARCHING..."
            hand_status = "HAND: DETECTED" if hand_detected else "HAND: SEARCHING..."

            c_col = GREEN if camera_active else RED
            f_col = GREEN if face_detected else YELLOW
            h_col = GREEN if hand_detected else YELLOW

            screen.blit(font_tech.render(cam_status, True, c_col), (70, status_y))
            screen.blit(font_tech.render(face_status, True, f_col), (70, status_y + 30))
            screen.blit(font_tech.render(hand_status, True, h_col), (70, status_y + 60))

            # Transition Condition
            if boot_timer > 350:
                game_state = "START"

        # Scan Line Effect
        scan_y = (boot_timer * 5) % HEIGHT
        pygame.draw.line(screen, GREEN, (0, scan_y), (WIDTH, scan_y), 2)

    # --- STATE: START (RETINAL LOCK) ---
    elif game_state == "START":
        title = font_title.render("RETINAL CALIBRATION", True, CYAN)
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 100))

        center_x, center_y = WIDTH // 2, HEIGHT // 2
        pygame.draw.rect(screen, DARK_GREY, (center_x - 100, center_y - 100, 200, 200), 2)

        if face_detected:
            eye_lock_timer += 1
            bar_w = min(200, eye_lock_timer * 4)
            pygame.draw.rect(screen, GREEN, (center_x - 100, center_y + 120, bar_w, 10))

            if eye_lock_timer % 10 == 0: play_snd(snd_scan)

            if eye_lock_timer > 60:
                play_snd(snd_shoot)
                game_state = "PLAYING"
                doctor = HealthMonitor()
        else:
            eye_lock_timer = 0
            warn = font_ui.render("Look at Camera to Unlock", True, RED)
            screen.blit(warn, (WIDTH // 2 - warn.get_width() // 2, center_y + 140))

    # --- STATE: PLAYING ---
    elif game_state == "PLAYING":
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

        # --- NEW DIFFICULTY LOGIC ---
        # Base speed 3-6. Add +1 speed for every 30 points scored.
        # Max speed cap of +15 to prevent impossible speeds
        speed_bonus = min(15, score // 30)

        if random.randint(1, max(10, 60 - speed_bonus * 2)) == 1:  # Spawn faster too
            enemy_counter += 1
            ex = random.randint(50, WIDTH - 50)
            base_speed = random.randint(3, 6)
            final_speed = base_speed + speed_bonus

            enemies.append([ex, -50, final_speed, 30, enemy_counter])
            doctor.log_spawn(enemy_counter)

        for e in enemies[:]:
            e[1] += e[2]  # Fall

            ex, ey = int(e[0]), int(e[1])
            pygame.draw.circle(screen, RED, (ex, ey), 30)
            pygame.draw.circle(screen, WHITE, (ex - 10, ey - 5), 8)
            pygame.draw.circle(screen, WHITE, (ex + 10, ey - 5), 8)

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
        pygame.draw.rect(screen, GREY, (hand_x - 20, hand_y + 30, 40, 6))
        pygame.draw.rect(screen, GREEN if pinching else YELLOW, (hand_x - 20, hand_y + 30, int(40 * pinch_strength), 6))

        # Display Difficulty Level
        lvl = 1 + (score // 30)
        screen.blit(font_ui.render(f"Score: {score}", True, WHITE), (10, 10))
        screen.blit(font_ui.render(f"Level: {lvl}", True, YELLOW), (10, 40))
        screen.blit(font_ui.render(f"Lives: {lives}", True, RED), (WIDTH - 100, 10))

    # --- STATE: GAMEOVER ---
    elif game_state == "GAMEOVER":
        screen.fill(DARK_GREY)
        report = doctor.generate_report()

        pygame.draw.rect(screen, BLACK, (50, 50, WIDTH - 100, HEIGHT - 100))
        pygame.draw.rect(screen, CYAN, (50, 50, WIDTH - 100, HEIGHT - 100), 2)

        head = font_title.render("ASSESSMENT COMPLETE", True, WHITE)
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

        rest = font_ui.render("Press 'R' to Restart", True, YELLOW)
        screen.blit(rest, (WIDTH // 2 - rest.get_width() // 2, HEIGHT - 60))

    pygame.display.flip()
    clock.tick(60)

if cap: cap.release()
pygame.quit()