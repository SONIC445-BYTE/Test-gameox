import pygame
import cv2
import mediapipe  # FIX: Import the full package name explicitly
import random
import math

# --- 1. SETUP & CONFIGURATION ---
# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Gesture Shooter")
clock = pygame.time.Clock()

# FIX: Initialize MediaPipe using the direct path to avoid AttributeError
mp_hands = mediapipe.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mediapipe.solutions.drawing_utils

# Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)  # Height

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
NEON_BLUE = (20, 255, 255)
BLACK = (0, 0, 0)

# Game Variables
score = 0
bullets = []
enemies = []
enemy_timer = 0
last_shot_time = 0
hand_x, hand_y = WIDTH // 2, HEIGHT // 2
is_pinching = False


# --- 2. GAME FUNCTIONS ---

def spawn_enemy():
    x = random.randint(50, WIDTH - 50)
    y = random.randint(-100, -40)
    # Enemy structure: [x, y, speed, size]
    enemies.append([x, y, random.randint(2, 5), 30])


def draw_hand_cursor(surface, x, y, pinching):
    # Draw a futuristic crosshair
    color = RED if pinching else NEON_BLUE
    size = 20 if pinching else 30

    # Outer ring
    pygame.draw.circle(surface, color, (x, y), size, 2)
    # Inner dot
    pygame.draw.circle(surface, color, (x, y), 5)
    # Cross lines
    pygame.draw.line(surface, color, (x - size, y), (x + size, y), 2)
    pygame.draw.line(surface, color, (x, y - size), (x, y + size), 2)


def detect_pinch(landmarks):
    # Get coordinates of Index Finger Tip (8) and Thumb Tip (4)
    index_tip = landmarks[8]
    thumb_tip = landmarks[4]

    # Calculate Euclidean distance
    distance = math.sqrt((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2)

    # Threshold for "pinch" (adjust if needed)
    return distance < 0.05


# --- 3. MAIN GAME LOOP ---
running = True
while running:
    # A. Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # B. OpenCV & MediaPipe Processing
    success, img = cap.read()
    if not success:
        break

    # Flip image specifically for processing (so movement mirrors correctly)
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # C. Hand Tracking Logic
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get Index Finger Tip coordinates for aiming
            # x and y are normalized (0.0 to 1.0), so we multiply by screen dimensions
            hand_x = int(hand_landmarks.landmark[8].x * WIDTH)
            hand_y = int(hand_landmarks.landmark[8].y * HEIGHT)

            # Check for shooting gesture (Pinch)
            if detect_pinch(hand_landmarks.landmark):
                is_pinching = True
            else:
                is_pinching = False

    # D. Game Logic

    # Shooting (Auto-fire when pinching)
    current_time = pygame.time.get_ticks()
    if is_pinching and current_time - last_shot_time > 200:  # 200ms cooldown
        bullets.append([hand_x, hand_y])
        last_shot_time = current_time

    # Move Bullets
    for b in bullets[:]:
        b[1] -= 10  # Move bullet up
        if b[1] < 0:
            bullets.remove(b)

    # Spawn Enemies
    enemy_timer += 1
    if enemy_timer > 60:  # Spawn every 60 frames (approx 1 sec)
        spawn_enemy()
        enemy_timer = 0

    # Move Enemies & Collision Detection
    for e in enemies[:]:
        e[1] += e[2]  # Move enemy down

        # Check collision with bullets
        enemy_rect = pygame.Rect(e[0] - e[3], e[1] - e[3], e[3] * 2, e[3] * 2)
        for b in bullets[:]:
            bullet_rect = pygame.Rect(b[0] - 5, b[1] - 5, 10, 10)
            if enemy_rect.colliderect(bullet_rect):
                try:
                    bullets.remove(b)
                    enemies.remove(e)
                    score += 10
                except ValueError:
                    pass  # Handle rare list removal errors
                break

        # Remove if off screen
        if e[1] > HEIGHT:
            enemies.remove(e)

    # E. Drawing
    screen.fill(BLACK)  # Clear screen

    # Draw Enemies
    for e in enemies:
        pygame.draw.circle(screen, RED, (e[0], e[1]), e[3])

    # Draw Bullets
    for b in bullets:
        pygame.draw.circle(screen, GREEN, (b[0], b[1]), 5)

    # Draw Hand Cursor
    draw_hand_cursor(screen, hand_x, hand_y, is_pinching)

    # Draw Score
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

    # Show small webcam feed in corner (Optional)
    # Resize cam frame to fit in corner
    img_small = cv2.resize(img, (200, 150))
    # Rotate for Pygame (OpenCV uses BGR, Pygame uses RGB and different axis)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    img_small = pygame.image.frombuffer(img_small.tobytes(), img_small.shape[1::-1], "RGB")
    screen.blit(img_small, (WIDTH - 210, HEIGHT - 160))

    pygame.display.flip()
    clock.tick(60)

# Cleanup
cap.release()
pygame.quit()