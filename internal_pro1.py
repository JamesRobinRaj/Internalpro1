import streamlit as st
import cv2
import pickle
import numpy as np
import os
import time
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
from datetime import timedelta, datetime
import pandas as pd
import random 
emotions = ["Sad", "Happy", "Cry", "Neutral"]

def emotions(data, name):
    emotions = ["Sad", "Happy", "Cry", "Neutral"]
    name_data = data.loc[data['Name'] == name].copy()
    name_data['Time'] = pd.to_datetime(name_data['Time'], format='%H:%M:%S')
    
    if len(name_data) == 0:
        st.write(f"No data available for {name}")
        return
    emotion_time = {}
    for emotion in emotions:
        emotion_data = name_data[name_data['Emotion'] == emotion]
        time_spent = emotion_data['Time'].diff().sum()
        emotion_time[emotion] = time_spent
    st.subheader(f"Emotion Report for {name}")
    for emotion, time_spent in emotion_time.items():
        time_spent_str = str(timedelta(seconds=time_spent.total_seconds()))
        st.write(f"Time spent on {emotion}: {time_spent_str}")
    max_emotion = max(emotion_time, key=emotion_time.get)
    st.write(f"Emotion with maximum time spent: {max_emotion}")

names_file = "names.pkl"
mugam_data_file = "mugam_data.pkl"
def register_face(name):
    video = cv2.VideoCapture(0)
    mugam_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    mugam_data = []
    i = 0
    capturing = True
    video_placeholder = st.empty() 
    while capturing:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mugam = mugam_detect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in mugam:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(mugam_data) < 100 and i % 10 == 0:
                mugam_data.append(resized_img)
            i += 1
            cv2.putText(frame, str(len(mugam_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        video_placeholder.image(frame, channels="BGR", use_column_width=True)
        k = cv2.waitKey(1)
        if k == ord('q') or len(mugam_data) == 100:
            capturing = False
    video.release()
    cv2.destroyAllWindows()
    mugam_data = np.asarray(mugam_data)
    mugam_data = mugam_data.reshape(100, -1)
    if not os.path.exists(names_file):
        names = [name] * 100
        with open(names_file, 'wb') as f:
            pickle.dump(names, f)
    else:
        with open(names_file, 'rb') as f:
            names = pickle.load(f)
        names = names + [name] * 100
        with open(names_file, 'wb') as f:
            pickle.dump(names, f)

    if not os.path.exists(mugam_data_file):
        with open(mugam_data_file, 'wb') as f:
            pickle.dump(mugam_data, f)
    else:
        with open(mugam_data_file, 'rb') as f:
            mugam = pickle.load(f)
        mugam = np.append(mugam, mugam_data, axis=0)
        with open(mugam_data_file, 'wb') as f:
            pickle.dump(mugam, f)

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def productivity(data, name):
    name_data = data.loc[data['Name'] == name].copy()
    name_data['Time'] = pd.to_datetime(name_data['Time'], format='%H:%M:%S')  
    if len(name_data) == 0:
        return None       
    total_time_duration = name_data['Time'].max() - name_data['Time'].min()
    drowsy_data = name_data[name_data['Drowsiness'] == 'Yes']
    time_diffs = drowsy_data['Time'].diff()
    total_drowsiness_time = time_diffs.sum()
    total_time_seconds = total_time_duration.total_seconds()
    total_drowsiness_seconds = total_drowsiness_time.total_seconds()
    total_hours = int(total_time_seconds // 3600)
    total_minutes = int((total_time_seconds % 3600) // 60)
    total_seconds = int(total_time_seconds % 60)
    drowsiness_hours = int(total_drowsiness_seconds // 3600)
    drowsiness_minutes = int((total_drowsiness_seconds % 3600) // 60)
    drowsiness_seconds = int(total_drowsiness_seconds % 60)
    unproductivity_percentage = 0  # Default value in case of division by zero
    if total_time_seconds != 0:
        unproductivity_percentage = (total_drowsiness_seconds / total_time_seconds) * 100
    reduction = 0
    if unproductivity_percentage <= 10:
        reduction = 0
    elif 10 < unproductivity_percentage <= 30:
        reduction = 5
    elif 30 < unproductivity_percentage <= 60:
        reduction = 10
    elif 60 < unproductivity_percentage <= 80:
        reduction = 15
    elif 80 < unproductivity_percentage <= 100:
        reduction = 20
    return {
        'total_time': f"{total_hours} hours, {total_minutes} minutes, {total_seconds} seconds",
        'total_drowsiness_time': f"{drowsiness_hours} hours, {drowsiness_minutes} minutes, {drowsiness_seconds} seconds",
        'unproductivity_percentage': f"Unproductivity percentage for {name}: {unproductivity_percentage:.2f}%",
        'reduction': f"Suggested Salary Reduction for {name}: {reduction} % from salary amount"
    }

def report(report_date):
    file = 'attendance_' + report_date + '.csv'
    path = f'C:/Users/FELIX KEVINSINGH/OneDrive/Desktop/HACKTOPIA/shape_predictor_68_face_landmarks.dat/attendance_csv/{file}'
    data = pd.read_csv(path)

    current_date = datetime.now().strftime('%Y-%m-%d')
    st.write(f"Report Date: {current_date}")
    
    name_list = data["Name"].unique()
    data = data.dropna()
    
    selected_name = st.selectbox("Select Employee", name_list)
    
    for name_to_check in name_list:
        if name_to_check == selected_name:
            result = productivity(data, name_to_check)
            if result is not None:
                st.write(f"Employee name: {name_to_check}")
                st.write(f"Total time for {name_to_check}: {result['total_time']}")
                st.write(f"Total drowsiness time for {name_to_check}: {result['total_drowsiness_time']}")
                st.write(result['unproductivity_percentage'])
                st.write(result['reduction'])
                st.write("---")
            else:
                st.warning(f"No data available for {name_to_check}")



def attendance_monitoring():
    def eye_aspect_ratio(eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    project_folder = os.path.dirname(os.path.abspath(_file_))
    names_file = os.path.join(project_folder, 'names.pkl')
    mugam_data_file = os.path.join(project_folder, 'mugam_data.pkl')
    attendance_folder = os.path.join(project_folder, 'attendance_csv')
    mugam_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    with open(names_file, 'rb') as f:
        labels = pickle.load(f)
    with open(mugam_data_file, 'rb') as f:
        MUGAM = pickle.load(f)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(MUGAM, labels)
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 48
    COUNTER = 0
    TOTAL = 0  
    step = 0
    monitoring_started = False

    emotions = ["Sad", "Happy", "Cry", "Neutral"]

    if not monitoring_started:
        if st.button("Start Monitoring"):
            monitoring_started = True

    if monitoring_started:
        if st.button("Stop Monitoring"):
            monitoring_started = False


    video_placeholder = st.empty()

    video = cv2.VideoCapture(0)

    while monitoring_started:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mugam = mugam_detect.detectMultiScale(gray, 1.3, 5)
        mugam = mugam_detect.detectMultiScale(gray, 1.3, 5)
        faces = mugam_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in mugam:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)
            face_roi = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_roi)
            emotion = emotions[3]
            if len(eyes) >= 2:
                eye_region = face_roi[eyes[0][1]:eyes[0][1]+eyes[0][3], eyes[0][0]:eyes[0][0]+eyes[0][2]]
                avg_intensity = cv2.mean(eye_region)[0]
                
                if avg_intensity < 60:  
                    emotion = emotions[0] if avg_intensity < 40 else emotions[2]
                elif avg_intensity > 100:
                    emotion = emotions[1]
            rect = dlib.rectangle(x, y, x + w, y + h)
            shape = shape_predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            left_eye = shape[42:48]
            right_eye = shape[36:42]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                COUNTER = 0

            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
            timestamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            drowsiness = "Yes" if COUNTER >= EYE_AR_CONSEC_FRAMES else "No"
            attendance_file = os.path.join(attendance_folder, 'attendance_' + date + '.csv')
            exist = os.path.isfile(attendance_file)

            if step % 60 == 0:
                with open(attendance_file, 'a') as f:
                    if not exist:
                        f.write('Name,Present,Time,Drowsiness,Emotion\n')
                    f.write(f"{output[0]},Present,{timestamp},{drowsiness},{emotion}\n")

            step += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (255, 0, 0), -1)
            cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            cv2.putText(frame, "Drowsiness: {}".format(drowsiness), (x, y-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2) 

        video_placeholder.image(frame, channels="BGR", use_column_width=True)

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


st.title("Productivity Monitoring ")

action = st.sidebar.selectbox("Select an action", ["Take Attendance", "Store Face Data", "Generate Report","Happiness Report"])

if action == "Take Attendance":
    attendance_monitoring()
elif action == "Store Face Data":
    st.title("Face Registration")
    name = st.text_input('Enter the name:')
    if name:
        start_registration = st.button("Start Registration")
        if start_registration:
            register_face(name)
            st.write("Registration completed.")
elif action == "Generate Report":
    st.title("Productivity Report Generator")
    st.sidebar.title("Employee Report Menu")
    report_date = st.sidebar.text_input("Enter Report Date (YYYY-MM-DD)")
    if st.sidebar.button("Generate Report"):
        report(report_date)
elif action == "Happiness Report":
    st.title("Productivity Report Generator")
    st.sidebar.title("Employee Happiess Report Menu")
    report_date = st.sidebar.text_input("Enter Report Date (YYYY-MM-DD)")
    file = 'attendance_' + report_date + '.csv'
    path = f'C:/Users/FELIX KEVINSINGH/OneDrive/Desktop/HACKTOPIA/shape_predictor_68_face_landmarks.dat/attendance_csv/{file}'
    data = pd.read_csv(path)
    name_list = data["Name"].unique()
    selected_name = st.selectbox("Select Employee", name_list)
    if st.button("Generate Emotion Report"):
        emotions(data, selected_name)
