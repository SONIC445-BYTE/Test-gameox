import pygame
import cv2
import mediapipe
import random
import math

# --- 1. SETUP & CONFIGURATION ---
# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Gesture Shooter v2.0")
clock = pygame.time.Clock()

# Initialize MediaPipe
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
YELLOW = (255, 255, 0)

# Game Variables
score = 0
player_lives = 3  # NEW: Start with 3 lives
difficulty_speed = 0  # NEW: Speed increase over time
game_over = False  # NEW: Game state

bullets = []
enemies = []
enemy_timer = 0
last_shot_time = 0
hand_x, hand_y = WIDTH // 2, HEIGHT // 2
is_pinching = False

# Fonts
font_small = pygame.font.Font(None, 36)
font_large = pygame.font.Font(None, 74)


# --- 2. GAME FUNCTIONS ---

def spawn_enemy():
    x = random.randint(50, WIDTH - 50)
    y = random.randint(-100, -40)
    # NEW: Base speed (2-5) + Difficulty Bonus
    speed = random.randint(2, 5) + difficulty_speed
    enemies.append([x, y, speed, 30])


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

    # Threshold for "pinch"
    return distance < 0.05


# --- 3. MAIN GAME LOOP ---
running = True
while running:
    # A. Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not game_over:
        # B. OpenCV & MediaPipe Processing
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # C. Hand Tracking Logic
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_x = int(hand_landmarks.landmark[8].x * WIDTH)
                hand_y = int(hand_landmarks.landmark[8].y * HEIGHT)

                if detect_pinch(hand_landmarks.landmark):
                    is_pinching = True
                else:
                    is_pinching = False

        # D. Game Logic

        # Shooting
        current_time = pygame.time.get_ticks()
        if is_pinching and current_time - last_shot_time > 200:
            bullets.append([hand_x, hand_y])
            last_shot_time = current_time

        # Move Bullets
        for b in bullets[:]:
            b[1] -= 10
            if b[1] < 0:
                bullets.remove(b)

        # Spawn Enemies
        enemy_timer += 1
        if enemy_timer > 60:
            spawn_enemy()
            enemy_timer = 0

        # Move Enemies & Logic
        for e in enemies[:]:
            e[1] += e[2]  # Move down

            # Collision with Bullets
            enemy_rect = pygame.Rect(e[0] - e[3], e[1] - e[3], e[3] * 2, e[3] * 2)
            for b in bullets[:]:
                bullet_rect = pygame.Rect(b[0] - 5, b[1] - 5, 10, 10)
                if enemy_rect.colliderect(bullet_rect):
                    try:
                        bullets.remove(b)
                        enemies.remove(e)
                        score += 10

                        # NEW: Increase difficulty every 100 points
                        if score % 100 == 0:
                            difficulty_speed += 1

                    except ValueError:
                        pass
                    break

            # NEW: Enemy Hits Bottom (Player loses life)
            if e[1] > HEIGHT:
                enemies.remove(e)
                player_lives -= 1
                if player_lives <= 0:
                    game_over = True

    # E. Drawing
    screen.fill(BLACK)

    # Draw Enemies
    for e in enemies:
        pygame.draw.circle(screen, RED, (e[0], e[1]), e[3])

    # Draw Bullets
    for b in bullets:
        pygame.draw.circle(screen, GREEN, (b[0], b[1]), 5)

    # Draw Cursor (Only if game is active)
    if not game_over:
        draw_hand_cursor(screen, hand_x, hand_y, is_pinching)

    # UI: Score
    score_text = font_small.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

    # UI: Lives (Draw hearts/squares in top right)
    for i in range(player_lives):
        pygame.draw.rect(screen, RED, (WIDTH - 40 - (i * 40), 10, 30, 30))

    # UI: Webcam Feed
    if not game_over and 'img' in locals():
        img_small = cv2.resize(img, (200, 150))
        img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        img_small = pygame.image.frombuffer(img_small.tobytes(), img_small.shape[1::-1], "RGB")
        screen.blit(img_small, (WIDTH - 210, HEIGHT - 160))

    # NEW: Game Over Screen
    if game_over:
        text_go = font_large.render("GAME OVER", True, RED)
        text_score = font_small.render(f"Final Score: {score}", True, WHITE)

        # Center the text
        rect_go = text_go.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 20))
        rect_score = text_score.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 40))

        screen.blit(text_go, rect_go)
        screen.blit(text_score, rect_score)

    pygame.display.flip()

    # If Game Over, wait a bit then close (or just pause)
    if game_over:
        pygame.time.wait(3000)  # Show screen for 3 seconds
        running = False  # Exit loop

    clock.tick(60)

# Cleanup
cap.release()
pygame.quit()