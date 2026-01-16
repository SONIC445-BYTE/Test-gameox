import mediapipe as mp
try:
    print(mp.solutions)
    print("SUCCESS! Solutions found.")
except AttributeError:
    print("STILL FAILING: Solutions missing.")