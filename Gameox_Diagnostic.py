import pygame
import cv2
import mediapipe
import random
import math
import numpy as np
import time

# --- 1. SETUP & CONFIGURATION ---
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural-Cam Shooter (Diagnostic Mode)")
clock = pygame.time.Clock()

# MediaPipe
mp_hands = mediapipe.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
DARK_GREY = (30, 30, 30)

# Fonts
font_ui = pygame.font.Font(None, 30)
font_big = pygame.font.Font(None, 60)
font_report = pygame.font.SysFont("Courier New", 18)  # Monospaced for report


# --- 2. DIAGNOSTIC ENGINE (The "Doctor" in the background) ---
class HealthMonitor:
    def __init__(self):
        # 1. Tremor Data
        self.hand_positions = []  # Stores (x, y) for jitter calc
        self.tremor_scores = []  # List of jitter measurements

        # 2. Range of Motion (ROM)
        self.max_pinch_dist = 0.0
        self.min_pinch_dist = 1.0

        # 3. Reflex / Cognitive
        self.enemy_spawn_times = {}  # ID -> timestamp
        self.reaction_times = []

    def update_position(self, x, y):
        """Called every frame to track steadiness."""
        self.hand_positions.append((x, y))
        if len(self.hand_positions) > 30:  # Maintain 1 sec window
            self.hand_positions.pop(0)

        # Calculate Jitter (Standard Deviation of movement)
        if len(self.hand_positions) > 10:
            x_vals = [p[0] for p in self.hand_positions]
            y_vals = [p[1] for p in self.hand_positions]
            # If standard deviation is high, hand is shaking/moving fast
            # We filter for 'Resting Tremor' by ignoring huge movements (aiming)
            jitter = np.std(x_vals) + np.std(y_vals)
            if jitter < 20:  # Only log if they aren't aggressively aiming
                self.tremor_scores.append(jitter)

    def update_rom(self, dist):
        """Tracks how wide/closed the hand gets."""
        if dist > self.max_pinch_dist: self.max_pinch_dist = dist
        if dist < self.min_pinch_dist: self.min_pinch_dist = dist

    def log_spawn(self, enemy_id):
        """Starts timer when enemy appears."""
        self.enemy_spawn_times[enemy_id] = time.time()

    def log_shot(self, enemies_on_screen):
        """Calculates reaction time when player shoots."""
        now = time.time()
        # Find the oldest enemy on screen (the one likely being shot at)
        if enemies_on_screen:
            target_id = enemies_on_screen[0][4]  # ID is index 4
            if target_id in self.enemy_spawn_times:
                reaction = (now - self.enemy_spawn_times[target_id]) * 1000  # ms
                if reaction < 2000:  # Filter out AFK moments
                    self.reaction_times.append(reaction)
                del self.enemy_spawn_times[target_id]  # Clear to prevent double counting

    def generate_report(self):
        """Compiles data into a medical summary."""
        # A. Tremor Analysis
        avg_jitter = np.mean(self.tremor_scores) if self.tremor_scores else 0
        tremor_status = "Steady"
        if avg_jitter > 2.0: tremor_status = "Mild Micro-Tremor"
        if avg_jitter > 5.0: tremor_status = "High Jitter (Fatigue/Nerves)"

        # B. ROM Analysis
        rom_range = (self.max_pinch_dist - self.min_pinch_dist) * 100
        rom_status = "Flexible" if rom_range > 20 else "Restricted/Stiff"

        # C. Reflex Analysis
        avg_reflex = np.mean(self.reaction_times) if self.reaction_times else 0
        reflex_status = "Normal"
        if avg_reflex < 250: reflex_status = "Athlete Level (Fast)"
        if avg_reflex > 450: reflex_status = "Delayed (Fatigue?)"
        if avg_reflex > 0 and avg_reflex < 150: reflex_status = "Impulsive (Guessing)"

        # D. Diagnoses Mapping
        screenings = []
        if avg_jitter > 5.0: screenings.append("• Parkinson's/Essential Tremor Screen: POSITIVE")
        if rom_range < 15: screenings.append("• Arthritis/Carpal Tunnel Screen: POSITIVE")
        if avg_reflex > 500: screenings.append("• Cognitive Delay/Fatigue Screen: POSITIVE")
        if not screenings: screenings.append("• No significant motor anomalies detected.")

        return [
            "--- BIOMETRIC TELEMETRY REPORT ---",
            f"1. Stability Index: {avg_jitter:.2f} ({tremor_status})",
            f"2. Motion Range:    {rom_range:.1f}% ({rom_status})",
            f"3. Avg Reaction:    {int(avg_reflex)}ms ({reflex_status})",
            "",
            "POTENTIAL SCREENING FLAGS:",
            *screenings,
            "",
            "NOTE: This is not a medical diagnosis.",
            "Consult a neurologist for confirmation."
        ]


# Initialize Monitor
doctor = HealthMonitor()

# --- 3. GAME VARIABLES ---
score = 0
lives = 3
game_over = False
bullets = []
enemies = []  # [x, y, speed, size, unique_id]
enemy_counter = 0  # To assign unique IDs
last_shot = 0
hand_x, hand_y = WIDTH // 2, HEIGHT // 2
pinching = False

# --- 4. MAIN LOOP ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN and game_over:
            if event.key == pygame.K_r:  # Restart
                score = 0
                lives = 3
                enemies = []
                bullets = []
                game_over = False
                doctor = HealthMonitor()  # Reset doctor

    if not game_over:
        # A. Vision Processing
        success, img = cap.read()
        if success:
            img = cv2.flip(img, 1)
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]

                # Update Hand Position
                hand_x = int(lm.landmark[8].x * WIDTH)
                hand_y = int(lm.landmark[8].y * HEIGHT)

                # --- DIAGNOSTIC LOGGING ---
                doctor.update_position(hand_x, hand_y)

                # Pinch Detection
                idx_x, idx_y = lm.landmark[8].x, lm.landmark[8].y
                thb_x, thb_y = lm.landmark[4].x, lm.landmark[4].y
                dist = math.sqrt((idx_x - thb_x) ** 2 + (idx_y - thb_y) ** 2)

                # --- DIAGNOSTIC LOGGING ---
                doctor.update_rom(dist)

                if dist < 0.05:
                    pinching = True
                else:
                    pinching = False

        # B. Game Logic
        # Shoot
        if pinching and pygame.time.get_ticks() - last_shot > 200:
            bullets.append([hand_x, hand_y])
            last_shot = pygame.time.get_ticks()
            # --- DIAGNOSTIC LOGGING ---
            doctor.log_shot(enemies)

        # Move Bullets
        for b in bullets[:]:
            b[1] -= 15
            if b[1] < 0: bullets.remove(b)

        # Spawn Enemy
        if random.randint(1, 60) == 1:
            ex = random.randint(50, WIDTH - 50)
            ey = -50
            speed = random.randint(3, 6)
            enemy_counter += 1
            enemies.append([ex, ey, speed, 30, enemy_counter])
            # --- DIAGNOSTIC LOGGING ---
            doctor.log_spawn(enemy_counter)

        # Move Enemies & Collision
        player_hit = False
        for e in enemies[:]:
            e[1] += e[2]

            # Hit by bullet
            e_rect = pygame.Rect(e[0] - 30, e[1] - 30, 60, 60)
            for b in bullets[:]:
                if e_rect.collidepoint(b[0], b[1]):
                    enemies.remove(e)
                    bullets.remove(b)
                    score += 10
                    break

            # Hit Player (Bottom)
            if e[1] > HEIGHT:
                enemies.remove(e)
                lives -= 1
                if lives <= 0: game_over = True

    # --- 5. DRAWING ---
    screen.fill(BLACK)

    if not game_over:
        # Draw Game Elements
        for b in bullets:
            pygame.draw.circle(screen, GREEN, (b[0], b[1]), 5)
        for e in enemies:
            pygame.draw.circle(screen, RED, (e[0], e[1]), 30)
            # Eyes to look like aliens
            pygame.draw.circle(screen, WHITE, (e[0] - 10, e[1] - 5), 8)
            pygame.draw.circle(screen, WHITE, (e[0] + 10, e[1] - 5), 8)
            pygame.draw.circle(screen, BLACK, (e[0] - 10, e[1] - 5), 3)
            pygame.draw.circle(screen, BLACK, (e[0] + 10, e[1] - 5), 3)

        # Crosshair
        color = RED if pinching else CYAN
        pygame.draw.line(screen, color, (hand_x - 20, hand_y), (hand_x + 20, hand_y), 2)
        pygame.draw.line(screen, color, (hand_x, hand_y - 20), (hand_x, hand_y + 20), 2)
        pygame.draw.circle(screen, color, (hand_x, hand_y), 15, 2)

        # HUD
        score_t = font_ui.render(f"Score: {score}", True, WHITE)
        lives_t = font_ui.render(f"Lives: {lives}", True, RED)
        screen.blit(score_t, (10, 10))
        screen.blit(lives_t, (WIDTH - 100, 10))

        # Webcam Preview (Small)
        if 'img' in locals():
            img_s = cv2.resize(img, (160, 120))
            img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
            img_s = pygame.image.frombuffer(img_s.tobytes(), img_s.shape[1::-1], "RGB")
            screen.blit(img_s, (WIDTH - 170, HEIGHT - 130))

    else:
        # --- GAME OVER / DIAGNOSIS SCREEN ---
        screen.fill(DARK_GREY)

        # Generate Report Once
        report_lines = doctor.generate_report()

        # Draw Title
        go_text = font_big.render("GAME OVER", True, RED)
        score_text = font_big.render(f"Final Score: {score}", True, WHITE)
        screen.blit(go_text, (WIDTH // 2 - go_text.get_width() // 2, 30))
        screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 90))

        # Draw Medical Report
        y_offset = 180
        # Box background
        pygame.draw.rect(screen, BLACK, (50, 160, WIDTH - 100, HEIGHT - 200))
        pygame.draw.rect(screen, CYAN, (50, 160, WIDTH - 100, HEIGHT - 200), 2)

        for line in report_lines:
            # Color coding lines
            c = WHITE
            if "POSITIVE" in line:
                c = RED
            elif "Healthy" in line or "Normal" in line:
                c = GREEN
            elif "DIAGNOSTIC" in line:
                c = CYAN
            elif "NOTE" in line:
                c = (150, 150, 150)

            line_s = font_report.render(line, True, c)
            screen.blit(line_s, (70, y_offset))
            y_offset += 30

        restart_text = font_ui.render("Press 'R' to Restart Diagnosis", True, YELLOW)
        screen.blit(restart_text, (WIDTH // 2 - restart_text.get_width() // 2, HEIGHT - 30))

    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()