# Import required libraries
import cv2
import numpy as np
from deepface import DeepFace
from datetime import datetime
import hashlib
import time
import pandas as pd
import random
import os
import json

# Task recommendation function
def recommend_task(emotion):
    task_recommendations = {
        "happy": [
            "Work on a creative project or brainstorm new ideas.",
            "Collaborate with teammates on a fun task.",
            "Tackle a challenging task that excites you.",
            "Share your positive energy by helping a colleague."
        ],
        "sad": [
            "Take a break and listen to calming music.",
            "Chat with a friend or colleague.",
            "Journal your thoughts to reflect.",
            "Engage in a light, enjoyable task."
        ],
        "angry": [
            "Take a walk to cool down.",
            "Practice deep breathing or mindfulness.",
            "Write down what’s bothering you.",
            "Do a physical activity to release tension."
        ],
        "neutral": [
            "Focus on routine tasks or organize your workspace.",
            "Plan your day for productivity.",
            "Learn a new tool or skill.",
            "Review your goals and progress."
        ],
        "surprise": [
            "Explore a new tool or idea.",
            "Take on a creative challenge.",
            "Brainstorm with your team.",
            "Reflect on what caused the surprise."
        ],
        "fear": [
            "Prioritize important tasks.",
            "Break tasks into manageable steps.",
            "Seek support from a colleague.",
            "Focus on confidence-building tasks."
        ],
        "disgust": [
            "Clean your workspace.",
            "Take a break for a refreshing activity.",
            "Address what caused the feeling.",
            "Switch to a value-aligned task."
        ],
        "crying": [
            "Rest and practice self-care.",
            "Reach out for support.",
            "Try meditation or stretching.",
            "Journal your feelings privately."
        ]
    }
    return random.choice(task_recommendations.get(emotion.lower(), ["No specific recommendation."]))
def anonymize_employee_id(employee_id):
    return hashlib.sha256(employee_id.encode()).hexdigest()

def is_image_blank(frame, threshold=10):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)
    mean_intensity = np.mean(gray)
    return variance < threshold or mean_intensity < 20

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    frame_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return frame_rgb

def take_photo(filename='photo.jpg', countdown_seconds=3, max_attempts=3):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    attempt = 1
    while attempt <= max_attempts:
        print(f"Attempt {attempt}/{max_attempts} to capture image...")
        for i in range(countdown_seconds, 0, -1):
            print(f"Capturing in {i} seconds...")
            ret, frame = cap.read()
            if ret:
                cv2.imshow("Webcam Preview", frame)
                cv2.waitKey(1000)
            else:
                print("Warning: Could not read frame during countdown.")
        cv2.destroyAllWindows()
    
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture image.")
            attempt += 1
            continue
        
        if is_image_blank(frame):
            print("Warning: Captured image appears blank.")
            attempt += 1
            if attempt <= max_attempts:
                print("Retrying capture...")
                time.sleep(1)
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            print("Warning: No face detected in image.")
            attempt += 1
            if attempt <= max_attempts:
                print("Retrying capture...")
                time.sleep(1)
            continue
        
        frame_processed = preprocess_image(frame)
        cv2.imwrite(filename, frame_processed)
        
        cv2.imshow("Captured Frame (Press any key to continue)", frame)
        print(f"Image saved as {filename} - displayed in window.")
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        
        cap.release()
        return filename
    
    print("Error: Failed to capture a valid image after maximum attempts.")
    cap.release()
    return None
def detect_emotion(image_path, max_samples=3):
    try:
        emotions = []
        confidences = []
        for _ in range(max_samples):
            result = DeepFace.analyze(
                img_path=image_path,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            emotion = result[0]['dominant_emotion']
            if emotion != "unknown":
                emotion_dist = result[0]['emotion']
                confidence = emotion_dist.get(emotion, 0)
                emotions.append(emotion)
                confidences.append(confidence)
            time.sleep(0.7)  
        
        if not emotions:
            print("Warning: No recognizable emotion detected across samples.")
            return "neutral"
        
        max_idx = np.argmax(confidences)
        final_emotion = emotions[max_idx]
        print(f"Emotion samples: {emotions}, Confidences: {[round(c, 2) for c in confidences]}")
        return final_emotion
    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return "neutral"

class MoodTracker:
    def __init__(self, csv_file="mood_history.csv"):
        self.csv_file = csv_file
        self.mood_data = []
        if os.path.exists(csv_file):
            self.mood_data = pd.read_csv(csv_file).to_dict('records')

    def track_mood(self, employee_id, team_id, mood):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.mood_data.append({
            "employee_id": employee_id,
            "team_id": team_id,
            "timestamp": timestamp,
            "mood": mood
        })
        pd.DataFrame(self.mood_data).to_csv(self.csv_file, index=False)

    def check_stress_level(self, employee_id, threshold=3, time_window_hours=24):
        df = pd.DataFrame(self.mood_data)
        if df.empty or employee_id not in df['employee_id'].values:
            return
        
        cutoff_time = pd.to_datetime(datetime.now()) - pd.Timedelta(hours=time_window_hours)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        recent_moods = df[
            (df['employee_id'] == employee_id) &
            (df['timestamp'] >= cutoff_time)
        ]
        
        negative_moods = recent_moods[recent_moods['mood'].isin(["sad", "angry", "fear", "crying"])]
        if len(negative_moods) >= threshold:
            print(f"Alert: Employee {employee_id[:8]}... may be experiencing stress. Notify HR.")

    def get_team_mood_analytics(self, team_id):
        df = pd.DataFrame(self.mood_data)
        if df.empty or team_id not in df['team_id'].values:
            return {"message": "No data for this team."}
        
        team_data = df[df['team_id'] == team_id]
        mood_counts = team_data['mood'].value_counts().to_dict()
        avg_moods_per_day = team_data.groupby(team_data['timestamp'].apply(lambda x: pd.to_datetime(x).date()))['mood'].count().mean()
        
        return {
            "team_id": team_id,
            "mood_distribution": mood_counts,
            "avg_moods_per_day": round(avg_moods_per_day, 2),
            "total_records": len(team_data)
        }

def real_time_task_optimizer(employee_id, team_id, tracker, num_frames=3):
    for i in range(num_frames):
        print(f"\nCapturing frame {i+1}/{num_frames}...")
        photo_filename = f"temp_photo_{i}.jpg"
        photo_filename = take_photo(photo_filename, countdown_seconds=3)
        
        if photo_filename is None:
            print("Skipping frame: No valid image captured.")
            tracker.track_mood(employee_id, team_id, "neutral")
            continue
        
        emotion = detect_emotion(photo_filename)
        print(f"Frame {i+1}: Detected Emotion - {emotion}")
        
        task = recommend_task(emotion)
        print(f"Recommended Task: {task}")
        
        tracker.track_mood(employee_id, team_id, emotion)
        
        tracker.check_stress_level(employee_id)
        
        if os.path.exists(photo_filename):
            os.remove(photo_filename)
        
        time.sleep(5)
def main():
    try:
        print("Starting AI-Powered Task Optimizer...")
        tracker = MoodTracker("mood_history.csv")
        
        employees = [
            {"id": "emp001", "team_id": "teamA"},
            {"id": "emp002", "team_id": "teamB"},
            {"id": "emp003", "team_id": "teamC"},
            {"id": "emp004", "team_id": "teamD"},
            {"id": "emp005", "team_id": "teamE"}
        ]
        
        for emp in employees:
            print(f"\nProcessing employee {emp['id']} in team {emp['team_id']}...")
            anon_id = anonymize_employee_id(emp['id'])
            real_time_task_optimizer(anon_id, emp['team_id'], tracker, num_frames=3)
            
            analytics = tracker.get_team_mood_analytics(emp['team_id'])
            print(f"\nTeam Analytics for {emp['team_id']}:")
            print(json.dumps(analytics, indent=2))
        
        df = pd.DataFrame(tracker.mood_data)
        if not df.empty:
            print("\nHistorical Mood Data:")
            print(df[['employee_id', 'team_id', 'timestamp', 'mood']].to_string(index=False))
    except Exception as e:
        print(f"Error running program: {e}")

if __name__ == "__main__":
    main()