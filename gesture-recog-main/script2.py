import cv2
import mediapipe as mp
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import streamlit as st
import base64

# Initialize MediaPipe FaceMesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

class FaceMeshTransformer(VideoTransformerBase):
    def __init__(self):
        # Face mesh options initialization
        self.drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def transform(self, frame):
        # Convert frame to RGB format
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = self.face_mesh.process(img_rgb)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Draw face mesh annotations on the image if landmarks are detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

        # Return the processed image
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
    set_background('images/background-p.png')

    input_style = """
    <style>
    input[type="text"] {
        background-color: transparent;
        color: #a19eae;
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

    col1, col2, col3 = st.columns(spec=[1, 2, 1])

    with col1:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.title("Real-Time Face Mesh Detection")
        st.write("This webapp shows live webcam feed with face mesh detection.")

    with col2:
        st.write("")
        st.write("")
        st.write("")
        webrtc_streamer(key="face-mesh-detection", video_transformer_factory=FaceMeshTransformer)

    with col3:
        st.write("")

if __name__ == "__main__":
    main()
