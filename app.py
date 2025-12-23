# Tara Tirzandina - 24060122130060
# Link GITHUB lengkap : https://github.com/CaptainRa/Strawberry-Scorch-Leaf-Detection
# Video Presentasi : https://youtu.be/WkJloJWIi8U

import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import time

model = YOLO("best.pt")
st.title("Deteksi Daun Gosong pada Daun Stroberi")

jenis_file = st.selectbox(
    'Pilih jenis file yang ingin anda deteksi',
    ('Foto', 'Video', 'Real-Time(WebCam)')
)

if jenis_file == 'Foto':
    st.write("Upload Gambar")
    uploaded = st.file_uploader("Masukkan dalam format JPG, JPEG, PNG", type=["jpg", "jpeg", "png"], accept_multiple_files=False, key=None, on_change=None, label_visibility="visible", width="stretch")
    
    if uploaded is not None:
        st.image(uploaded)

        predict = st.button("Predict", key="run")
        if predict:
            extension = uploaded.name.split('.')[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as file:
                file.write(uploaded.getbuffer())
                image = file.name

            try:
                results = model.predict(source=image, conf=0.3, save=True, project="runs/detect", name="picture", exist_ok=True)

                if len(results) > 0:
                    annotated_frame = results[0].plot()
                    if annotated_frame is not None:
                        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        st.image(annotated_frame_rgb, caption="Prediction", use_column_width=True)
                    else:
                        st.warning("Hasil deteksi kosong")
                else:
                    st.warning("Tidak ada hasil deteksi")
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                os.unlink(image)

elif jenis_file == 'Video':
    st.write("Upload Video")
    uploaded = st.file_uploader("Masukkan dalam format mp4", type=["mp4"], accept_multiple_files=False, key=None, on_change=None, label_visibility="visible", width="stretch")

    if uploaded is not None:
        st.video(uploaded)

        predict = st.button("Predict", key="run_video")
        if predict:
            extension = uploaded.name.split('.')[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as file:
                file.write(uploaded.getbuffer())
                video = file.name

            try:
                run_name = f"video_{int(time.time())}"
                results = model.predict(source=video, conf=0.3, save=True, project="runs/detect", name=run_name, exist_ok=True)

                saved_video = None
                out_dir = os.path.join("runs", "detect", run_name)
                
                if os.path.isdir(out_dir):
                    for f in os.listdir(out_dir):
                        if f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
                            video_path = os.path.join(out_dir, f)
                            
                            if f.lower().endswith('.avi'):
                                mp4_path = os.path.join(out_dir, "output.mp4")
                                cap = cv2.VideoCapture(video_path)
                                fourcc = cv2.VideoWriter_fourcc(*'H264')
                                fps = cap.get(cv2.CAP_PROP_FPS)
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                
                                out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
                                while True:
                                    ret, frame = cap.read()
                                    if not ret:
                                        break
                                    out.write(frame)
                                cap.release()
                                out.release()
                                saved_video = mp4_path
                            else:
                                saved_video = video_path
                            break

                if saved_video and os.path.isfile(saved_video):
                    with open(saved_video, "rb") as vf:
                        video_bytes = vf.read()
                    st.video(video_bytes)
                else:
                    st.warning("Hasil video tidak ditemukan.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                try:
                    os.unlink(video)
                except:
                    pass

else :
    st.write("Real-Time Detection")
    
    run_detection = st.button("Mulai Deteksi", key="start_detection")
    stop_detection = st.button("Hentikan Deteksi", key="stop_detection")
    
    if run_detection:
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model.predict(source=frame, conf=0.3, iou=0.5, verbose=False)
            
            annotated_frame = results[0].plot()
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            if len(results) > 0 and results[0].boxes is not None:
                num_detections = len(results[0].boxes)
                info_placeholder.write(f"Deteksi: {num_detections} object")
            
            frame_placeholder.image(annotated_frame_rgb, use_column_width=True)
            
            if stop_detection:
                break
        
        cap.release()
        st.write("Deteksi dihentikan")

