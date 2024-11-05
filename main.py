import cv2
from ultralytics import solutions, YOLO
import click
import os
import time 
from PIL import Image

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

# Load the heatmap class
from heatmapper import Heatmap
def create_heatmap(model, colormap=cv2.COLORMAP_TURBO, show=False):
    if model is None:
        raise ValueError("Model is required to create heatmap")
    return Heatmap(colormap=colormap, show=show, model=model, classes=[0])

# Settings
@click.command()
@click.option('--device', default='cpu', help='Device to run the model: [jetson, cpu, gpu]')
@click.option('--model', default='yolo11n', help='Model to use: [yolo11n, yolo11m, yolo5s, yolo5m, yolo5l, yolo5x]')
@click.option('--camera', default='0', help='Camera to use: [rtsp, 0]')
@click.option('--width', type=int, default=1440, help='Width of the camera')
@click.option('--height', type=int, default=810, help='Height of the camera')
@click.option('--test', type=int, default='0', help='[1: True, 0: False]')


def main(device, model, camera, height, width, test):
    # colored output


    
    if camera == '0':
        camera = int(camera)

    if test == 1:
        print(f"Testing camera {camera}...")
    else:
        print(f"Settings:")
        print(f"-- Device: {device}")
        print(f"-- Model: {model}")
        print(f"-- Camera: {camera} \n")
        print(f"-- {width}x{height} resolution \n")

    if camera == 'rtsp':
        live_camera = "rtsp://admin:ai123123@192.168.0.64/Streaming/Channels/101"
        pred_camera = "rtsp://admin:ai123123@192.168.0.64/Streaming/Channels/102"

    # Load the YOLO model
    yolo_cpu = YOLO(model)
    if device == 'cpu':
        yolo = yolo_cpu
    elif device == 'gpu':
        yolo = YOLO(model, device='gpu')
    elif device == 'jetson':
        # check if {model}.engine exists
        if not os.path.exists(f'{model}.engine'):
            print(f"{model}.engine not found.")
            print(f"Creating {model}.engine...")
            yolo_cpu.export(format="engine")
        model = f'{model}.engine'
        yolo = YOLO(model)

    # Create the object counter
    # counter = solutions.ObjectCounter(
    #     model=model,
    #     show=False,
    #     region=region,
    #     classes=[0]
    # )

    # Create the heatmap object
    heatmap = create_heatmap(model)

    # Get the video
    if camera == 'rtsp':
        # live_cap = getCamera(live_camera)
        cap = getCamera(live_camera, width, height)
    else:
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
        visitors_counts = []
        while cap.isOpened():

            success, im_src = cap.read()
            if not success:
                break

            im0 = cv2.resize(im_src, (width, height))

            # track objects in the frame
            if i % 7 == 0:
                heatmapImg = heatmap.generate_heatmap(im0)

            if i % 5 == 0:
                results = yolo.track(im_src, classes=[0],)

            # once per second, save the results to a text file
            if time.time() - timestamp > 1:
                timestamp = time.time()
                # write the date time to the text file and objects count
                with open("output/results.txt", "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}, {len(results[0].boxes.cls.tolist())}\n")
            
            # add no_of_visitors to the list
            visitors_counts.append(len(results[0].boxes.cls.tolist()))
            # if the len of visitors_counts is greater than 60*60 (1 hour)
            if len(visitors_counts) > 60*60:
                # remove the first element
                visitors_counts.pop(0)

            # scale up the heatmap
            heatmapImg = cv2.resize(heatmapImg, (im_src.shape[1], im_src.shape[0]))
            # merge the heatmap with the original image
            withHeatmap = cv2.addWeighted(im_src, 1, heatmapImg, 0.5, 0)

            # plot the results on the image
            im1 = results[0].plot(img=withHeatmap, line_width=2)

            # show current count of people and avg count of people in the last hour
            cv2.putText(im1, f"CURRENT VISITORS: {len(results[0].boxes.cls.tolist())}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(im1, f"AVG. VISITORS (1HR): {round(sum(visitors_counts)/len(visitors_counts))}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.namedWindow("Innovation Lab AI Analyzer", cv2.WINDOW_FREERATIO)

            cv2.imshow("Innovation Lab AI Analyzer", im1)
            
            if cv2.waitKey(1) == ord('q'):
                break
            i += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
