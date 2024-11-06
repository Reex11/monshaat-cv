import cv2
from ultralytics import YOLO
import click
import os
import time 
from deepface import DeepFace

# Functions
def getCamera(id, width=False, height=False):
    print(f"Opening camera {id}...")
    cap = cv2.VideoCapture(id)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    if width and height:
        cap.set(3, width)
        cap.set(4, height)
    return cap

def getVideo(path):
    cap = cv2.VideoCapture(path)
    assert cap.isOpened(), "Cannot open video"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(f'output_{extract_filename(path)}.avi', cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    return cap, video_writer

def extract_filename(path):
    # extract the filename without the path and extension
    return path.split('/')[-1].split('.')[0]

def faces_annotation(faces, frame, rgb_frame):
    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']
        emotion = translate_emotion(emotion)

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Add image of the emotion from arabic folder
        emotion_img = cv2.imread(f'arabic/{emotion}.png', cv2.IMREAD_UNCHANGED)
        em_height, em_width, em_channels = emotion_img.shape

        # handle alpha channel to blend the image with the frame
        for c in range(0, 3):
            frame[y:y + em_height, x:x + em_width, c] = (1 - emotion_img[:, :, 3] / 255.0) * frame[y:y + em_height, x:x + em_width, c] + (emotion_img[:, :, c] / 255.0) * emotion_img[:, :, 3]

        # Add image to the frame
        # frame[y:y + em_height, x:x + em_width] = emotion_img

    return frame


def translate_emotion(emotion):
    # translate emotions to arabic
    if emotion == 'angry':
        return 'غاضب'
    elif emotion == 'disgust':
        return 'منزعج'
    elif emotion == 'fear':
        return 'خائف'
    elif emotion == 'happy':
        return 'سعيد'
    elif emotion == 'sad':
        return 'حزين'
    elif emotion == 'surprise':
        return 'متفاجئ'
    elif emotion == 'neutral':
        return 'محايد'
    else:
        return 'غير معروف'

# Settings
@click.command()
@click.option('--device', default='cpu', help='Device to run the model: [jetson, cpu, gpu]')
@click.option('--camera', default='0', help='Camera to use: [rtsp, 0]')
@click.option('--width', type=int, default=640, help='Width of the camera')
@click.option('--height', type=int, default=480, help='Height of the camera')
@click.option('--test', type=int, default='0', help='[1: True, 0: False]')



def main(device, camera, height, width, test):

    face_model = 'yolov8n-face-lindevs'
    # face_model = 'yolov8n-emotions'
    pose_model = 'yolo11n-pose'

    if camera == '0':
        camera = int(camera)

    if test == 1:
        print(f"Testing camera {camera}...")
    else:
        print(f"Settings:")
        print(f"-- Device: {device}")
        print(f"-- Camera: {camera}")
        print(f"-- {width}x{height} resolution\n")

    region = [(400,500),(400,0)]

    if camera == 'rtsp':
        camera = "rtsp://admin:ai123123@192.168.0.64/Streaming/Channels/101"

    # Load the YOLO model
    # face_yolo_cpu = YOLO(f'{face_model}.pt')
    pose_yolo_cpu = YOLO(f'{pose_model}.pt')
    if device == 'cpu':
        # face_yolo = face_yolo_cpu
        pose_yolo = pose_yolo_cpu
    elif device == 'jetson':
        # check if {model}.engine exists
        # if not os.path.exists(f'{face_model}.engine'):
        #     print(f"{face_model}.engine not found.")
        #     print(f"Creating {face_model}.engine...")
        #     face_yolo_cpu.export(format="engine")
        # face_model = f'{face_model}.engine'
        # face_yolo = YOLO(face_model)

        if not os.path.exists(f'{pose_model}.engine'):
            print(f"{pose_model}.engine not found.")
            print(f"Creating {pose_model}.engine...")
            pose_yolo_cpu.export(format="engine")
        pose_model = f'{pose_model}.engine'
        pose_yolo = YOLO(pose_model)

    # Load the face cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Get the video
    cap = getCamera(camera, width, height)


    if test == '1':
        if not cap.isOpened():
            print("Cannot open camera")
        else:
            print("Camera opened successfully. Press 'q' to exit")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                # Display the resulting frame
                cv2.imshow('Camera Test', frame)
                if cv2.waitKey(1) == ord('q'):
                    break
    
    else:
        # Process the video
        i = 0
        timestamp = time.time()
        while cap.isOpened():
            
            success, im_src = cap.read()
            if not success:
                break


            im0 = cv2.resize(im_src, (width, height))
            
            # track objects in the frame
            if i % 2 == 0:
                pose_results = pose_yolo.track(im0, classes=[0])

            if i % 3 == 0:
                # Convert frame to grayscale
                gray_frame = cv2.cvtColor(im_src, cv2.COLOR_BGR2GRAY)
                # Convert grayscale frame to RGB format
                rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
                # Detect faces in the frame
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))



            # merge the heatmap with the original image
            im0 = pose_results[0].plot(img=im0,boxes=False)

            # plot the results on the image
            if faces is not None:
                im1 = faces_annotation(faces, im0, rgb_frame)
            else:
                im1 = im0
            # im1 = face_results[0].plot(img=im0, conf=False, color_mode='instance', labels=False)

            # Deep face 
            # obj = DeepFace.analyze(im0, actions=['emotion','gender'])

            # for face in obj:
            #     emotion = face['dominant_emotion']
            #     gender = face['dominant_gender']
            #     if gender == 'male':
            #         gender = 'ذكر'
            #     else:
            #         gender = 'أنثى'
            #     region = face['region']
            #     im1 = cv2.putText(im0, f"{gender} {emotion(emotion)}", (region[0], region[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Innovation Lab AI Pose Detector", im1)

            if cv2.waitKey(1) == ord('q'):
                break
            i += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
