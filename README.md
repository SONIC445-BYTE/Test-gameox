# 🧬 Project GAMEOX: The Neuro-Motor Assessment Unit

[**Play Web Version**](https://SONIC445-BYTE.github.io/Test-gameox/index_v0.2.html)

Project GAMEOX is an experimental fusion of arcade-style gaming and real-time healthcare diagnostics. By leveraging Computer Vision and AI-driven hand tracking, GAMEOX transforms a classic space shooter into a sophisticated diagnostic tool capable of screening for neuro-motor anomalies.

## 🚀 The Vision: Gaming Meets Healthcare
Why choose between entertainment and health? GAMEOX utilizes the engagement of gaming to perform passive medical screenings. As you play, the system tracks subtle tremors, reaction times, and muscle fatigue, providing a comprehensive "Neuro-Motor Report" at the end of every session.

## 🧠 Core Diagnostic Capabilities (Beta)
The Diagnostic Engine monitors your performance to screen for potential indicators of:

*   **Stability & Tremors:** Analyzing hand jitter for early signs of Parkinson’s or Essential Tremors.
*   **Reflex & Cognition:** Measuring reaction times in milliseconds to detect signs of Concussions, ADHD (Impulsivity), or Depression (Psychomotor Slowness).
*   **Range of Motion (ROM):** Tracking pinch strength and movement fluidity to identify muscle fatigue related to ALS or Myasthenia.
*   **Coordination:** Assessing accuracy and motor control for Dyspraxia screening.

## 🕹️ Main Experience: Diagnostic Pro
The flagship version of the project is `Gameox_Diagnostic_pro.py`. It offers high-fidelity tracking and the most advanced diagnostic reporting available in the suite.

### Key Features
*   **AI Hand Tracking:** Powered by MediaPipe for sub-millimeter precision.
*   **Dynamic Pinch Logic:** Uses palm-scale normalization to detect gestures regardless of hand distance from the camera.
*   **Smooth Motion Engine:** Predictive cursor smoothing for a high-performance gaming feel.
*   **Real-time HUD:** Visual pinch-strength bars and telemetry data.

## 🛠️ Getting Started
### Prerequisites
To run the Pro version locally, you will need:
*   Python 3.10+
*   A webcam
*   The following libraries:
    ```bash
    pip install pygame opencv-python mediapipe numpy
    ```

### Running the Game
1.  Clone the repository.
2.  Navigate to the root directory.
3.  Launch the diagnostic unit:
    ```bash
    python "Gameox_Diagnostic_pro.py"
    ```

## 🎮 How to Play
1.  **Calibrate:** Ensure your hand is visible to the webcam. A cyan cursor will follow your index finger.
2.  **Move:** Move your hand to guide your unit across the screen.
3.  **Shoot:** Pinch your thumb and index finger together to fire.
4.  **Engage:** Neutralize incoming threats. If they pass your defense, your "vitals" (lives) will decrease.
5.  **Review:** Once the session ends, wait for the Neuro-Motor Report to generate your results.

## 🚧 Roadmap & Contributions
Project GAMEOX is an open-source experiment, and viewers are encouraged to become contributors!

*   [ ] **Mobile Optimization:** Enhancing the WebGL/MediaPipe bridge for mobile browsers.
*   [ ] **Expanded Diagnostics:** Adding tests for Carpal Tunnel and Arthritis.
*   [ ] **Global Leaderboards:** Comparing reaction times and stability scores globally.
*   [ ] **Lore Integration:** Fusing medical data into a deeper sci-fi narrative.

Want to help? Check out the issues or submit a PR with your enhancements!

## ⚠️ Medical Disclaimer
Project GAMEOX is intended for research and entertainment purposes only. The screenings provided are based on experimental algorithms and do NOT constitute a clinical diagnosis. Always consult a medical professional for health concerns.

Developed with ❤️ for the future of Interactive Health.
