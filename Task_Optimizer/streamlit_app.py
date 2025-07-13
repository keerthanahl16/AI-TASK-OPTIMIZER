import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
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
import av # Required by streamlit-webrtc for frame handling

# --- Your existing helper functions (slightly modified or unchanged) ---

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
        # "crying" is not a direct DeepFace emotion, but good to keep if you have another way to detect it.
        # DeepFace typically maps "crying" to "sad" or "fear".
        "crying": [ # Added this to handle potential "crying" if it ever appears or if you manually set it
            "Rest and practice self-care.",
            "Reach out for support.",
            "Try meditation or stretching.",
            "Journal your feelings privately."
        ]
    }
    return random.choice(task_recommendations.get(emotion.lower(), ["No specific recommendation."]))

def anonymize_employee_id(employee_id):
    return hashlib.sha256(employee_id.encode()).hexdigest()

class MoodTracker:
    def __init__(self, csv_file="mood_history.csv"):
        self.csv_file = csv_file
        self.mood_data = []
        
        # Determine the absolute path for the CSV file
        # This makes sure it always looks in the directory where streamlit_app.py is
        script_dir = os.path.dirname(__file__) # Gets the directory of the current script
        self.csv_full_path = os.path.join(script_dir, csv_file)
        
        os.makedirs(os.path.dirname(self.csv_full_path) or '.', exist_ok=True)
        
        if os.path.exists(self.csv_full_path):
            try:
                self.mood_data = pd.read_csv(self.csv_full_path).to_dict('records')
            except pd.errors.EmptyDataError:
                self.mood_data = [] # Handle empty CSV case
        
    def track_mood(self, employee_id, team_id, mood):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.mood_data.append({
            "employee_id": employee_id,
            "team_id": team_id,
            "timestamp": timestamp,
            "mood": mood
        })
        # Save to CSV immediately after update
        pd.DataFrame(self.mood_data).to_csv(self.csv_full_path, index=False)

    def check_stress_level(self, employee_id, threshold=3, time_window_hours=24):
        df = pd.DataFrame(self.mood_data)
        if df.empty or employee_id not in df['employee_id'].values:
            return "" # Return empty string if no relevant data

        cutoff_time = pd.to_datetime(datetime.now()) - pd.Timedelta(hours=time_window_hours)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        recent_moods = df[
            (df['employee_id'] == employee_id) &
            (df['timestamp'] >= cutoff_time)
        ]
        
        negative_moods = recent_moods[recent_moods['mood'].isin(["sad", "angry", "fear", "disgust"])] # Changed "crying" to "disgust" as DeepFace outputs these.
        if len(negative_moods) >= threshold:
            return f"🚨 **Stress Alert:** Employee {employee_id[:8]}... may be experiencing stress. Recommend checking in with HR or taking a break."
        return "" # No alert

    def get_team_mood_analytics(self, team_id):
        df = pd.DataFrame(self.mood_data)
        if df.empty or team_id not in df['team_id'].values:
            return {"message": "No data for this team."}
        
        team_data = df[df['team_id'] == team_id]
        mood_counts = team_data['mood'].value_counts().to_dict()
        
        # Calculate average moods per day more robustly
        if not team_data.empty:
            team_data['date'] = pd.to_datetime(team_data['timestamp']).dt.date
            avg_moods_per_day = team_data.groupby('date')['mood'].count().mean()
        else:
            avg_moods_per_day = 0
            
        return {
            "team_id": team_id,
            "mood_distribution": mood_counts,
            "avg_moods_per_day": round(avg_moods_per_day, 2),
            "total_records": len(team_data)
        }

# --- Streamlit Video Processor Class ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self, tracker, employee_id, team_id):
        self.tracker = tracker
        self.employee_id = employee_id
        self.team_id = team_id
        self.last_analysis_time = time.time()
        self.analysis_interval = 3 # Analyze emotion every 3 seconds to reduce load

        if 'current_emotion' not in st.session_state:
            st.session_state.current_emotion = "Waiting..."
        if 'current_task' not in st.session_state:
            st.session_state.current_task = "Waiting for detection..."
        if 'stress_alert' not in st.session_state:
            st.session_state.stress_alert = ""

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        current_time = time.time()
        if current_time - self.last_analysis_time > self.analysis_interval:
            self.last_analysis_time = current_time

            try:
                results = DeepFace.analyze(
                    img_path=img,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                
                if results and len(results) > 0 and 'dominant_emotion' in results[0]:
                    emotion = results[0]['dominant_emotion']
                    st.session_state.current_emotion = emotion
                    st.session_state.current_task = recommend_task(emotion)
                    
                    self.tracker.track_mood(self.employee_id, self.team_id, emotion)
                    stress_message = self.tracker.check_stress_level(self.employee_id)
                    st.session_state.stress_alert = stress_message
                else:
                    st.session_state.current_emotion = "neutral"
                    st.session_state.current_task = recommend_task("neutral")
                    self.tracker.track_mood(self.employee_id, self.team_id, "neutral")
                    st.session_state.stress_alert = ""
                
            except Exception as e:
                st.session_state.current_emotion = "Error/Neutral"
                st.session_state.current_task = recommend_task("neutral")
                self.tracker.track_mood(self.employee_id, self.team_id, "neutral")
                st.session_state.stress_alert = ""

        if 'current_emotion' in st.session_state and st.session_state.current_emotion:
            text = f"Emotion: {st.session_state.current_emotion.upper()}"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(img, (10, 10), (10 + text_width + 10, 10 + text_height + 10), (0, 0, 0), -1)
            cv2.putText(img, text, (20, 10 + text_height + baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Main Streamlit App Layout ---
def main_streamlit_app():
    st.set_page_config(page_title="AI Mood Optimizer", layout="centered")

    st.title("💡 AI-Powered Task Optimizer")
    st.markdown(
        """
        This application uses your live webcam feed to detect your dominant emotion
        and provides a personalized task recommendation. It also logs your mood over time
        and alerts for potential stress.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        employee_id_input = st.text_input("Enter Your Employee ID:", "demo_user_123", help="This will be anonymized for privacy.")
    with col2:
        team_id_input = st.selectbox("Select Your Team:", ["Team Alpha", "Team Beta", "Team Gamma", "Team Delta"], help="Select your team for collective analytics.")

    anon_employee_id = anonymize_employee_id(employee_id_input)

    # Initialize MoodTracker (CSV will be in the same directory as the app.py)
    # The MoodTracker constructor now uses os.path.dirname(__file__) to correctly locate the CSV.
    tracker = MoodTracker("mood_history.csv") 

    st.subheader("📸 Live Emotion Detection & Task Recommendation")
    st.info("Click 'START' to activate your webcam. Ensure your face is visible for accurate detection.")

    webrtc_ctx = webrtc_streamer(
        key="webcam-emotion-detector",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: VideoProcessor(tracker, anon_employee_id, team_id_input),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )

    st.markdown(f"**Current Emotion:** <span style='font-size:24px; color:blue;'>{st.session_state.current_emotion.upper()}</span>", unsafe_allow_html=True)
    st.markdown(f"**Recommended Task:** <span style='font-size:20px; color:green;'>{st.session_state.current_task}</span>", unsafe_allow_html=True)
    
    if st.session_state.stress_alert:
        st.warning(st.session_state.stress_alert)

    st.subheader("📊 Mood History & Team Analytics")

    df_mood = pd.DataFrame(tracker.mood_data)
    if not df_mood.empty:
        user_mood_data = df_mood[df_mood['employee_id'] == anon_employee_id]
        if not user_mood_data.empty:
            st.write(f"Your Recent Mood Entries (Employee {employee_id_input[:5]}...):")
            st.dataframe(user_mood_data[['timestamp', 'mood']].tail(10))
        else:
            st.info("No mood data recorded for this employee yet. Start the webcam to log your mood!")

        st.write(f"### Team Analytics for {team_id_input}:")
        team_analytics = tracker.get_team_mood_analytics(team_id_input)
        st.json(team_analytics)

        st.write("### Overall Mood Distribution Across All Users:")
        overall_mood_counts = df_mood['mood'].value_counts().reset_index()
        overall_mood_counts.columns = ['Mood', 'Count']
        st.bar_chart(overall_mood_counts.set_index('Mood'))

    else:
        st.info("No mood data recorded yet across the system. Start the webcam to begin tracking!")

    st.markdown("---")
    st.markdown("Developed by Your Name/Team Name")

if __name__ == "__main__":
    main_streamlit_app()
