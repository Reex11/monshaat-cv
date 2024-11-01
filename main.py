import cv2
from ultralytics import solutions, YOLO
import click


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
@click.option('--device', prompt='Processing Device', default='cpu', help='Device to run the model: [jetson, cpu, gpu]')
@click.option('--model', prompt='Model', default='yolo11n', help='Model to use: [yolo11n, yolo11m, yolo5s, yolo5m, yolo5l, yolo5x]')
@click.option('--camera', prompt='Camera', default='0', help='Camera to use: [rtsp, 0]')
@click.option('--width', type=int, prompt='Width', default=640, help='Width of the camera')
@click.option('--height', type=int, prompt='Height', default=480, help='Height of the camera')
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
        print(f"-- {width}x{height} resolution")

    region = [(400,500),(400,0)]

    if camera == 'rtsp':
        live_camera = "rtsp://admin:ai123123@192.168.0.64/Streaming/Channels/101"
        pred_camera = "rtsp://admin:ai123123@192.168.0.64/Streaming/Channels/102"
    else:
        live_camera = camera
        pred_camera = camera

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
    counter = solutions.ObjectCounter(
        model=model,
        show=False,
        region=region,
        classes=[0]
    )

    # Create the heatmap object
    heatmap = create_heatmap(model)

    # Get the video
    if camera == 'rtsp':
        # live_cap = getCamera(live_camera)
        cap = getCamera(pred_camera, width, height)
    else:
        cap = getCamera(live_camera, width, height)


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
        while cap.isOpened():
            
            success, im0 = cap.read()
            if not success:
                break

            im0 = cv2.resize(im0, (width, height))
            # track objects in the frame
            results = yolo.track(im0)
            annotated_img = results[0].plot()

            # generate heatmap every 10 frames
            if i % 5 == 0:
                heatmapImg = heatmap.generate_heatmap(im0)
            
            annotated_img = counter.count(im0)
            # merge the heatmap with the original image
            im1 = cv2.addWeighted(annotated_img, 1, heatmapImg, 0.8, 0)
            
            cv2.imshow("Heatmap", im1)

            if cv2.waitKey(1) == ord('q'):
                break
            i += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
