import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import base64

DESIRED_HEIGHT = 400
DESIRED_WIDTH = 480

class GestureRecognitionTransformer(VideoTransformerBase):
    def __init__(self):
        self.base_options = python.BaseOptions(model_asset_path='model/gesture_recognizer.task')
        self.options = vision.GestureRecognizerOptions(base_options=self.base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        recognition_result = self.recognizer.recognize(mp_image)

        if recognition_result.gestures:
            top_gesture = recognition_result.gestures[0][0]
            hand_landmarks = recognition_result.hand_landmarks

            cv2.putText(img, f"Gesture: {top_gesture.category_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            for hand_landmark in hand_landmarks:
                for point in hand_landmark:
                    cv2.circle(img, (int(point.x * img.shape[1]), int(point.y * img.shape[0])), 5, (0, 255, 0), -1)

        return img


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(image_file):
    bin_str = get_base64_of_bin_file(image_file)
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
def main():
    st.set_page_config(layout="wide")
    st.logo('images/mrm-norm.png')
    # Set the background image using the set_background function
    set_background('images/background-p.png')

    input_style = """
    <style>
    input[type="text"] {
        background-color: transparent;
        color: #a19eae;  // This changes the text color inside the input box
    }
    div[data-baseweb="base-input"] {
        background-color: transparent !important;
    }
    [data-testid="stAppViewContainer"] {
        background-color: transparent !important;
    }
    </style>
    """
    st.markdown(input_style, unsafe_allow_html=True)

    col1, col2, col3= st.columns(spec = [1,2, 1])

    with col1:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.title("Real-Time Gesture Recognition")
        st.write("This webapp shows live webcam feed with gesture recognition, try thumbs up, thumbs down, point up and victory!")
                
        
    with col2:
        st.write("")
        st.write("")
        st.write("")
        webrtc_streamer(key="gesture-recognition", video_transformer_factory=GestureRecognitionTransformer)

    with col3:
        st.write("")

   
if __name__ == "__main__":
    main()
