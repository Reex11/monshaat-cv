import cv2
from ultralytics import solutions, YOLO

# Functions
def getCamera(id):
    cap = cv2.VideoCapture(id)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
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


# Load the YOLO model

# Load the heatmap class
from heatmapper import Heatmap
def create_heatmap(model, colormap=cv2.COLORMAP_TURBO, show=False):
    if model is None:
        raise ValueError("Model is required to create heatmap")
    return Heatmap(colormap=colormap, show=show, model=model)

# Settings
yolo = YOLO('yolo11n')
model = "yolo11n.pt"
region = [(400,500),(400,0)]
camera = 0


# Create the heatmap object
heatmap = create_heatmap(model)

# Create the object counter
counter = solutions.ObjectCounter(
    model=model,
    show=False,
    region=region,
)



# Get the video
cap = getCamera(camera)


# Process the video
i = 0
while cap.isOpened():
    
    success, im0 = cap.read()
    if not success:
        break

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