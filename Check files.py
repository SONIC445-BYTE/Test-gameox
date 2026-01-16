import os

folder = "assets"

if not os.path.exists(folder):
    print(f"ERROR: The folder '{folder}' does not exist!")
else:
    print(f"--- Files found in '{folder}' ---")
    files = os.listdir(folder)
    for f in files:
        print(f" - {f}")

    print("\n--- Diagnostic Checks ---")
    if "spaceship.png" in files:
        print("✅ Spaceship is named correctly.")
    else:
        print("❌ spaceship.png is MISSING or named wrong.")

    if "alien.png" in files:
        print("✅ Alien is named correctly.")
    else:
        print("❌ alien.png is MISSING. Did you name it 'alien.png.jpg'?")

    if "bullet.png" in files:
        print("✅ Bullet is named correctly.")
    else:
        print("❌ bullet.png is MISSING.")