import pygame
import cv2
import mediapipe
import random
import math
import numpy as np
import time

# --- 1. CONFIGURATION ---
pygame.init()
# Game Resolution (High Quality Display)
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neuro-Motor Assessment Unit (Pro Optimized)")
clock = pygame.time.Clock()

mp_hands = mediapipe.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,  # Lowered slightly for speed
    min_tracking_confidence=0.5
)

# --- OPTIMIZATION FIX: Lower Camera Processing Resolution ---
# We process a small image (fast) but map coordinates to the big screen (HD)
CAM_W, CAM_H = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, CAM_W)
cap.set(4, CAM_H)

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
font_report = pygame.font.SysFont("Courier New", 16)


# --- 2. DIAGNOSTIC ENGINE (Unchanged Logic) ---
class HealthMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.hand_positions = []
        self.tremor_scores = []
        self.pinch_history = []
        self.enemy_spawn_times = {}
        self.reaction_times = []
        self.shots_fired = 0
        self.shots_hit = 0

    def update_movement(self, x, y):
        self.hand_positions.append((x, y))
        if len(self.hand_positions) > 30: self.hand_positions.pop(0)

        # Only calc jitter if we have enough data
        if len(self.hand_positions) > 10:
            x_vals = [p[0] for p in self.hand_positions]
            y_vals = [p[1] for p in self.hand_positions]
            jitter = float(np.std(x_vals) + np.std(y_vals))
            # Filter out intentional large movements (aiming)
            if jitter < 25: self.tremor_scores.append(jitter)

    def update_rom(self, d):
        self.pinch_history.append(d)

    def log_shot(self):
        self.shots_fired += 1

    def log_hit(self):
        self.shots_hit += 1

    def log_spawn(self, enemy_id):
        self.enemy_spawn_times[enemy_id] = time.time()

    def calculate_reflex(self, target_id):
        now = time.time()
        if target_id in self.enemy_spawn_times:
            reaction = (now - self.enemy_spawn_times[target_id]) * 1000
            if reaction < 3000: self.reaction_times.append(reaction)
            del self.enemy_spawn_times[target_id]

    def generate_report(self):
        avg_jitter = float(np.mean(self.tremor_scores)) if self.tremor_scores else 0.0
        avg_reflex = float(np.mean(self.reaction_times)) if self.reaction_times else 0.0
        accuracy = (self.shots_hit / self.shots_fired * 100) if self.shots_fired > 0 else 0.0

        rom_fatigue = False
        if len(self.pinch_history) > 50:
            first_q = float(np.mean(self.pinch_history[:20]))
            last_q = float(np.mean(self.pinch_history[-20:]))
            if last_q < (first_q * 0.8): rom_fatigue = True

        screenings = []
        if avg_jitter > 5.0: screenings.append("• Parkinson's / Essential Tremor: POSITIVE")
        if avg_reflex > 450: screenings.append("• Concussion / Brain Injury: POSITIVE")
        if avg_reflex < 150 and avg_reflex > 0: screenings.append("• ADHD (Impulsivity): POSITIVE")

        if avg_reflex > 500 and avg_jitter < 3.0: screenings.append("• Depression (Psychomotor Slowness): POSITIVE")
        if avg_jitter > 4.0 and accuracy < 60: screenings.append("• Multiple Sclerosis (Intention Tremor): POSITIVE")
        if avg_jitter > 10.0: screenings.append("• Huntington's (Chorea/Jerkiness): POSITIVE")
        if rom_fatigue: screenings.append("• ALS / Myasthenia (Muscle Fatigue): POSITIVE")
        if avg_reflex < 400 and accuracy < 40: screenings.append("• Dyspraxia (Coordination Deficit): POSITIVE")
        if avg_jitter > 4.0 and avg_reflex > 500 and accuracy < 50: screenings.append(
            "• Alcohol / Substance Intoxication: POSITIVE")

        if not screenings: screenings.append("• No significant motor anomalies detected.")

        return [
            "--- NEURO-MOTOR REPORT ---",
            f"Stability (Jitter):  {avg_jitter:.2f}",
            f"Avg Reaction Time:   {int(avg_reflex)}ms",
            f"Coordination (Acc):  {int(accuracy)}%",
            f"Fatigue Detected:    {'YES' if rom_fatigue else 'NO'}",
            "",
            "SCREENING RESULTS:",
            *screenings,
            "",
            "DISCLAIMER: For research/entertainment only.",
            "Not a substitute for clinical diagnosis."
        ]


# --- 3. GAME STATE ---
doctor = HealthMonitor()
score = 0
lives = 3
game_over = False
bullets = []
enemies = []
enemy_counter = 0
last_shot = 0

# Cursor Smoothing Variables
hand_x, hand_y = WIDTH // 2, HEIGHT // 2
prev_hand_x, prev_hand_y = WIDTH // 2, HEIGHT // 2
SMOOTHING_FACTOR = 0.5  # Higher = Smoother but more lag (0.0 - 1.0)

pinching = False
pinch_strength = 0  # New visualization variable

# --- 4. MAIN LOOP ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN and game_over:
            if event.key == pygame.K_r:  # Restart
                score, lives, enemies, bullets, game_over = 0, 3, [], [], False
                doctor = HealthMonitor()
                enemy_counter = 0

    if not game_over:
        success, img = cap.read()
        if success:
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]

                # --- A. CURSOR SMOOTHING ---
                # Get raw target position
                target_x = int(lm.landmark[8].x * WIDTH)
                target_y = int(lm.landmark[8].y * HEIGHT)

                # Smooth the movement
                hand_x = int(prev_hand_x + (target_x - prev_hand_x) * SMOOTHING_FACTOR)
                hand_y = int(prev_hand_y + (target_y - prev_hand_y) * SMOOTHING_FACTOR)
                prev_hand_x, prev_hand_y = hand_x, hand_y

                doctor.update_movement(hand_x, hand_y)

                # --- B. DYNAMIC PINCH LOGIC ---
                #
                # 1. Get Tip Distance (Index to Thumb)
                x1, y1 = lm.landmark[8].x, lm.landmark[8].y
                x2, y2 = lm.landmark[4].x, lm.landmark[4].y
                tip_dist = math.hypot(x2 - x1, y2 - y1)

                # 2. Get Palm Scale (Wrist to Middle Knuckle)
                # This measures how "big" the hand is (Z-depth)
                x_base, y_base = lm.landmark[0].x, lm.landmark[0].y
                x_mid, y_mid = lm.landmark[9].x, lm.landmark[9].y
                palm_size = math.hypot(x_mid - x_base, y_mid - y_base)

                # 3. Calculate Normalized Pinch Ratio
                if palm_size > 0:
                    pinch_ratio = tip_dist / palm_size
                else:
                    pinch_ratio = 1.0

                # Threshold ~0.25 is usually a solid pinch
                if pinch_ratio < 0.25:
                    pinching = True
                else:
                    pinching = False

                doctor.update_rom(tip_dist)

                # Calculate strength for UI (0.0 to 1.0)
                # Maps ratio 0.5->0.0 to strength 0.0->1.0
                pinch_strength = max(0, min(1, 1 - (pinch_ratio / 0.5)))

        # Game Logic
        current_time = pygame.time.get_ticks()
        if pinching and current_time - last_shot > 200:
            bullets.append([hand_x, hand_y])
            last_shot = current_time
            doctor.log_shot()

        for b in bullets[:]:
            b[1] -= 20  # Faster bullets
            if b[1] < 0: bullets.remove(b)

        if random.randint(1, 60) == 1:
            enemy_counter += 1
            ex = random.randint(50, WIDTH - 50)
            enemies.append([ex, -50, random.randint(3, 6), 30, enemy_counter])
            doctor.log_spawn(enemy_counter)

        for e in enemies[:]:
            e[1] += e[2]
            e_rect = pygame.Rect(e[0] - 30, e[1] - 30, 60, 60)

            hit = False
            for b in bullets[:]:
                if e_rect.collidepoint(int(b[0]), int(b[1])):
                    enemies.remove(e)
                    bullets.remove(b)
                    score += 10
                    doctor.log_hit()
                    doctor.calculate_reflex(e[4])
                    hit = True
                    break

            if not hit and e[1] > HEIGHT:
                enemies.remove(e)
                lives -= 1
                if lives <= 0: game_over = True

    # Drawing
    screen.fill(BLACK)
    if not game_over:
        for b in bullets:
            pygame.draw.circle(screen, GREEN, (int(b[0]), int(b[1])), 5)
        for e in enemies:
            pygame.draw.circle(screen, RED, (e[0], e[1]), 30)
            # Simple Eye Graphics
            pygame.draw.circle(screen, WHITE, (e[0] - 10, e[1] - 5), 8)
            pygame.draw.circle(screen, WHITE, (e[0] + 10, e[1] - 5), 8)

        # Draw Cursor & Pinch Strength Bar
        color = RED if pinching else CYAN
        pygame.draw.circle(screen, color, (hand_x, hand_y), 15, 2)

        # --- NEW: VISUAL PINCH BAR ---
        # Draws a bar next to the cursor showing pinch readiness
        bar_len = 40
        bar_h = 6
        bar_x = hand_x - 20
        bar_y = hand_y + 30
        # Background
        pygame.draw.rect(screen, GREY, (bar_x, bar_y, bar_len, bar_h))
        # Fill (based on strength)
        fill_w = int(bar_len * pinch_strength)
        fill_c = GREEN if pinching else YELLOW
        pygame.draw.rect(screen, fill_c, (bar_x, bar_y, fill_w, bar_h))

        # HUD
        screen.blit(font_ui.render(f"Score: {score}", True, WHITE), (10, 10))
        screen.blit(font_ui.render(f"Lives: {lives}", True, RED), (WIDTH - 100, 10))
    else:
        screen.fill(DARK_GREY)
        report = doctor.generate_report()

        pygame.draw.rect(screen, BLACK, (50, 50, WIDTH - 100, HEIGHT - 100))
        pygame.draw.rect(screen, CYAN, (50, 50, WIDTH - 100, HEIGHT - 100), 2)

        y_offset_draw = 80
        for line in report:
            c = RED if "POSITIVE" in line else GREEN if "No significant" in line else WHITE
            screen.blit(font_report.render(line, True, c), (70, y_offset_draw))
            y_offset_draw += 30

        screen.blit(font_ui.render("Press 'R' to Restart", True, YELLOW), (WIDTH // 2 - 100, HEIGHT - 40))

    pygame.display.flip()
    # Optimize: Cap FPS at 60 but don't force it if lagging
    clock.tick(60)

cap.release()
pygame.quit()