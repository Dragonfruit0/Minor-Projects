from transformers import pipeline
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Use a pipeline as a high-level helper
pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")

# Load face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def main():
    st.title("Real-Time Mood Classifier 🎮")
    st.write("This app uses your webcam to classify mood in real-time.")

    app_mode = st.sidebar.selectbox("Choose an option", ["About", "Run Mood Classifier"])

    if app_mode == "About":
        st.markdown(
            """
            ### About this App
            - This app classifies emotions from webcam input using a facial emotion detection model.
            - Powered by Hugging Face Transformers and OpenCV.
            """
        )
    elif app_mode == "Run Mood Classifier":
        run_mood_classifier()

def preprocess_face(face_rgb):
    """
    Preprocess the face for the pipeline.
    """
    try:
        # Resize and normalize the face to match pipeline requirements
        face_resized = cv2.resize(face_rgb, (224, 224))  # Adjust to model input
        face_normalized = face_resized / 255.0  # Normalize pixel values (0-1)
        face_pil = Image.fromarray((face_normalized * 255).astype(np.uint8))  # Convert back to uint8 PIL Image
        return face_pil
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None

def run_mood_classifier():
    st.write("Click the **Start Webcam** button to begin!")

    if st.button("Start Webcam", key="start_webcam"):
        cap = cv2.VideoCapture(0)  # 0 for default webcam

        frame_window = st.image([])  # Streamlit image placeholder
        stop_button_shown = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to access webcam.")
                break

            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Extract and preprocess the face
                face_rgb = frame[y:y + h, x:x + w]  # Extract RGB region
                face_pil = preprocess_face(face_rgb)

                if face_pil is None:
                    continue  # Skip if preprocessing failed

                try:
                    # Predict emotion using the Hugging Face pipeline
                    result = pipe(face_pil)
                    emotion_label = result[0]['label']
                    emotion_score = result[0]['score']
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    continue

                # Display the emotion on the frame
                cv2.putText(frame, f"{emotion_label} ({emotion_score:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Convert frame for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame)

            # Show stop button only if not already shown
            if not stop_button_shown:
                if st.button("Stop Webcam", key="stop_webcam"):
                    break
                stop_button_shown = True

        cap.release()
        cv2.destroyAllWindows()

# Run the Streamlit app
if __name__ == "__main__":
    main()
