import pygame
import cv2
import mediapipe
import random
import math
import os  # Needed to check for file existence

# --- 1. SETUP & CONFIGURATION ---
# Initialize Pygame and Sound Mixer
pygame.init()
pygame.mixer.init()  # Initialize sound engine

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Gesture Shooter - REAL Version")
clock = pygame.time.Clock()

# Initialize MediaPipe
mp_hands = mediapipe.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mediapipe.solutions.drawing_utils

# Initialize Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Colors (Fallbacks)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
NEON_BLUE = (20, 255, 255)
BLACK = (0, 0, 0)

# --- ASSET LOADING (Robust Version) ---
assets_dir = "assets"
os.makedirs(assets_dir, exist_ok=True)

# Placeholders
spaceship_img = None
alien_img = None
bullet_img = None
shoot_sound = None
explosion_sound = None

print("--- Checking for Assets ---")
try:
    # 1. Load Images
    # Spaceship
    if os.path.exists(os.path.join(assets_dir, "spaceship.png")):
        spaceship_img = pygame.image.load(os.path.join(assets_dir, "spaceship.png")).convert_alpha()
        spaceship_img = pygame.transform.scale(spaceship_img, (60, 60))
        print("✅ Loaded: spaceship.png")

    # Alien (Checks for 'alien.png' OR 'aliens.png')
    if os.path.exists(os.path.join(assets_dir, "alien.png")):
        alien_img = pygame.image.load(os.path.join(assets_dir, "alien.png")).convert_alpha()
    elif os.path.exists(os.path.join(assets_dir, "aliens.png")):
        alien_img = pygame.image.load(os.path.join(assets_dir, "aliens.png")).convert_alpha()

    if alien_img:
        alien_img = pygame.transform.scale(alien_img, (60, 60))
        print("✅ Loaded: alien.png")
    else:
        print("⚠️ alien.png/aliens.png not found. Using red circles.")

    # Bullet
    if os.path.exists(os.path.join(assets_dir, "bullet.png")):
        bullet_img = pygame.image.load(os.path.join(assets_dir, "bullet.png")).convert_alpha()
        bullet_img = pygame.transform.scale(bullet_img, (20, 40))
        print("✅ Loaded: bullet.png")

except Exception as e:
    print(f"⚠️ Image Loading Issue: {e}. Using shapes.")

try:
    # 2. Load Sounds (Checks for .wav OR .mp3)
    # Shoot Sound
    if os.path.exists(os.path.join(assets_dir, "shoot.wav")):
        shoot_sound = pygame.mixer.Sound(os.path.join(assets_dir, "shoot.wav"))
    elif os.path.exists(os.path.join(assets_dir, "shoot.mp3")):
        shoot_sound = pygame.mixer.Sound(os.path.join(assets_dir, "shoot.mp3"))

    if shoot_sound:
        shoot_sound.set_volume(0.4)
        print("✅ Loaded: shoot sound")

    # Explosion Sound
    if os.path.exists(os.path.join(assets_dir, "explosion.wav")):
        explosion_sound = pygame.mixer.Sound(os.path.join(assets_dir, "explosion.wav"))
    elif os.path.exists(os.path.join(assets_dir, "explosion.mp3")):
        explosion_sound = pygame.mixer.Sound(os.path.join(assets_dir, "explosion.mp3"))

    if explosion_sound:
        explosion_sound.set_volume(0.6)
        print("✅ Loaded: explosion sound")

except Exception as e:
    print(f"⚠️ Sound Loading Issue: {e}. Game will be silent.")
print("---------------------------")

# Game Variables
score = 0
player_lives = 3
difficulty_speed = 0
game_over = False

bullets = []
enemies = []  # Structure: [x, y, speed, size (radius for fallback)]
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
    speed = random.randint(2, 5) + difficulty_speed
    enemies.append([x, y, speed, 30])  # 30 is radius used for fallback collision


def draw_hand_cursor(surface, x, y, pinching):
    if spaceship_img:
        # Draw the spaceship image centered on the hand position
        # We subtract half width/height to center it
        rect = spaceship_img.get_rect(center=(x, y))
        surface.blit(spaceship_img, rect)
    else:
        # Fallback: Draw the old futuristic crosshair
        color = RED if pinching else NEON_BLUE
        size = 20 if pinching else 30
        pygame.draw.circle(surface, color, (x, y), size, 2)
        pygame.draw.circle(surface, color, (x, y), 5)
        pygame.draw.line(surface, color, (x - size, y), (x + size, y), 2)
        pygame.draw.line(surface, color, (x, y - size), (x, y + size), 2)


def detect_pinch(landmarks):
    index_tip = landmarks[8]
    thumb_tip = landmarks[4]
    distance = math.sqrt((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2)
    return distance < 0.05


# --- 3. MAIN GAME LOOP ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not game_over:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_x = int(hand_landmarks.landmark[8].x * WIDTH)
                hand_y = int(hand_landmarks.landmark[8].y * HEIGHT)
                is_pinching = detect_pinch(hand_landmarks.landmark)

        # --- Game Logic ---

        # Shooting
        current_time = pygame.time.get_ticks()
        if is_pinching and current_time - last_shot_time > 250:  # Slightly slower firerate for sounds
            bullets.append([hand_x, hand_y])
            last_shot_time = current_time
            # Play Sound
            if shoot_sound: shoot_sound.play()

        # Move Bullets
        for b in bullets[:]:
            b[1] -= 12  # Faster bullets
            if b[1] < -50: bullets.remove(b)

        # Spawn Enemies
        enemy_timer += 1
        if enemy_timer > 60:
            spawn_enemy()
            enemy_timer = 0

        # Move Enemies & Collision
        for e in enemies[:]:
            e[1] += e[2]

            # Collision Logic (Works for both images and circles)
            # We define hitboxes based on the center position e[0], e[1]
            enemy_hitbox = pygame.Rect(e[0] - e[3], e[1] - e[3], e[3] * 2, e[3] * 2)

            for b in bullets[:]:
                # Bullet hitbox is small
                bullet_hitbox = pygame.Rect(b[0] - 10, b[1] - 20, 20, 40)
                if enemy_hitbox.colliderect(bullet_hitbox):
                    try:
                        bullets.remove(b)
                        enemies.remove(e)
                        score += 10
                        # Play Sound
                        if explosion_sound: explosion_sound.play()
                        if score % 100 == 0: difficulty_speed += 1
                    except ValueError:
                        pass
                    break

            if e[1] > HEIGHT + 50:
                enemies.remove(e)
                player_lives -= 1
                if player_lives <= 0: game_over = True

    # --- E. Drawing ---
    screen.fill(BLACK)  # You could replace this with a background image!

    # Draw Enemies
    for e in enemies:
        if alien_img:
            # Blit image centered on coordinates
            rect = alien_img.get_rect(center=(e[0], e[1]))
            screen.blit(alien_img, rect)
        else:
            # Fallback circle
            pygame.draw.circle(screen, RED, (e[0], e[1]), e[3])

    # Draw Bullets
    for b in bullets:
        if bullet_img:
            rect = bullet_img.get_rect(center=(b[0], b[1]))
            screen.blit(bullet_img, rect)
        else:
            pygame.draw.circle(screen, GREEN, (b[0], b[1]), 5)

    if not game_over:
        draw_hand_cursor(screen, hand_x, hand_y, is_pinching)

    # UI Elements
    score_text = font_small.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

    # Draw Lives (Using spaceship image if available, else red squares)
    for i in range(player_lives):
        x_pos = WIDTH - 40 - (i * 40)
        if spaceship_img:
            life_icon = pygame.transform.scale(spaceship_img, (30, 30))
            screen.blit(life_icon, (x_pos, 10))
        else:
            pygame.draw.rect(screen, RED, (x_pos, 10, 30, 30))

    # Webcam feed
    if not game_over and 'img' in locals():
        img_small = cv2.resize(img, (200, 150))
        img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        img_small = pygame.image.frombuffer(img_small.tobytes(), img_small.shape[1::-1], "RGB")
        screen.blit(img_small, (WIDTH - 210, HEIGHT - 160))

    if game_over:
        text_go = font_large.render("GAME OVER", True, RED)
        text_score = font_small.render(f"Final Score: {score}", True, WHITE)
        rect_go = text_go.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 20))
        rect_score = text_score.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 40))
        screen.blit(text_go, rect_go)
        screen.blit(text_score, rect_score)

    pygame.display.flip()

    if game_over:
        pygame.time.wait(3000)
        running = False

    clock.tick(60)

cap.release()
pygame.quit()