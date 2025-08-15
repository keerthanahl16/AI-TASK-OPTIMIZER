The AI-Powered Task Optimizer is a data science and machine learning project designed to enhance workplace productivity and well-being by analyzing employees' emotions in real-time.
Using facial expression analysis via webcam, the system detects moods, recommends personalized tasks, tracks emotional trends, identifies stress, and provides team-level insights. 
It ensures data privacy through anonymization and supports a healthier, empathetic work environment.

This project was developed as part of a personal exploration into computer vision and emotion recognition, inspired by the need to align tasks with emotional states for better performance.

Table of Contents

Features

1.Real-Time Emotion Detection:
Captures webcam images to detect emotions (happy, sad, angry, neutral, surprise, fear, disgust, crying) using DeepFace.
Handles blank images and failed detections with retries and fallback to "neutral" mood.
Preprocesses images (lighting normalization, face alignment) for improved accuracy.

2.Task Recommendation:
Suggests tasks based on detected emotions (e.g., "Work on a creative project" for happy, "Take a break" for sad).
Includes custom recommendations for "crying" to support emotional well-being.

3.Historical Mood Tracking:
Stores mood data (employee ID, team ID, timestamp, mood) in mood_history.csv.
Maintains a timeline for analyzing long-term emotional trends.

4.Stress Management Alerts:
Monitors for prolonged negative moods (sad, angry, fear, crying).
Alerts HR if 3+ negative moods are detected within 24 hours.

5.Team Mood Analytics:
Aggregates mood data by team, providing insights into morale (e.g., mood distribution, average moods per day).
Outputs analytics in JSON format for easy integration.

6.Data Privacy:
Anonymizes employee IDs using SHA-256 hashing.
Deletes temporary image files after processing.

7.User-Friendly Capture:
3-second countdown with live webcam preview before capturing.
Displays captured images until a key is pressed, ensuring visibility during execution.
Retries up to 3 times for blank images or no-face detections.

Demo

You can try the live app here: [AI Task Optimizer – Streamlit](https://ai-task-optimizer-czalhjd9age8idtzdefjlx.streamlit.app/)


Example terminal output:
![Screenshot (15)](https://github.com/user-attachments/assets/22ea9312-da32-4acf-beac-6dbb9385dbc1)

![Screenshot (12)](https://github.com/user-attachments/assets/30a27c58-81d8-4887-92e8-3b23103f64da)

![Screenshot (13)](https://github.com/user-attachments/assets/b429ea95-fed0-411e-8f2b-d74fa3319e13)
<img width="1366" height="768" alt="Screenshot (52)" src="https://github.com/user-attachments/assets/db947d66-7229-471f-94b4-4ff7f95205f7" />
<img width="1366" height="768" alt="Screenshot (50)" src="https://github.com/user-attachments/assets/ab4fbdd1-9838-451f-9e6c-9d5aeef9d853" />
<img width="1366" height="768" alt="Screenshot (51)" src="https://github.com/user-attachments/assets/9fabacb4-7568-48a2-8ec0-cd66592111ad" />

Prerequisites

Operating System: Windows 10/11, macOS, or Linux.
Python: Version 3.8 or higher (3.10 recommended).
Webcam: Built-in or external, with good lighting for accurate face detection.
Internet: Required for DeepFace model downloads on first run.
Hardware: Basic CPU (GPU optional for faster DeepFace processing).
Software: Visual Studio Code (recommended) or any Python IDE.

Installation

1.Clone the Repository:

git clone https://github.com/keerthanahl16/ai-powered-task-optimizer.git
cd ai-powered-task-optimizer

2.Set Up a Virtual Environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

3.Install Dependencies:

pip install opencv-python deepface numpy pandas

opencv-python: Webcam capture and image display.
deepface: Emotion detection.
numpy: Image processing.
pandas: Mood tracking and analytics.

4.Verify Webcam:

Open your webcam in an app (e.g., Windows Camera) to ensure it works.
Close other apps using the webcam (e.g., Zoom).

Usage

1.Open in VS Code:

Launch VS Code.
Go to File > Open Folder and select the project directory (e.g., ai-powered-task-optimizer).

2.Select Python Interpreter:

Press Ctrl+Shift+P, type Python: Select Interpreter.
Choose the virtual environment’s Python (e.g., ./venv/Scripts/python.exe).

3.Run the Script:

Option 1: Open task_optimizer.py, click the “Run Python File” button (triangle, top-right).
Option 2: In terminal:

python task_optimizer.py

4.Interact:

Countdown: For each frame, a 3-second countdown shows a live “Webcam Preview” window.
Capture: Face the webcam, smile for “happy”, etc. The captured image appears in “Captured Frame (Press any key to continue)”.
Press Key: Press any key to proceed after inspecting the image.
Output: Terminal shows emotions, tasks, analytics, and saves data to mood_history.csv.

5.Check Results:

Terminal: Displays emotions (with confidence scores), tasks, stress alerts, and analytics.
mood_history.csv: Open in Excel/VS Code to view mood records.
Image Windows: Images stay visible until you press a key.

Acknowledgements

DeepFace: For emotion detection models.
OpenCV: For webcam capture and image processing.
Pandas: For data management.
Inspired by workplace well-being research and computer vision applications.

Created by Keerthana H L in April 2025.
