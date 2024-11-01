import click
import cv2

@click.command()
@click.option('--cam', help='Select the camera to test (0 or rtsp)')

def main(cam):
    # if numeric camera is selected
    if cam == '0':
        print(f"Testing camera {cam}...")
        camera = int(cam)
    # if rtsp camera is selected
    elif cam == 'rtsp':
        print(f"Testing camera {cam}...")
        camera = "rtsp://admin:ai123123@192.168.0.64/Streaming/Channels/101"
    else:
        print("Invalid camera option. Please select either 0 or rtsp.")

    cap = cv2.VideoCapture(camera)
    
    # change to MJPG for better performance
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # change the camera resolution
    cap.set(3, 640)
    cap.set(4, 480)

    if not cap.isOpened():
        print("Error: Camera not found.")
        return
    else:
        print("Camera found. Testing...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Frame not received.")
                break
            cv2.imshow(f'Camera Test: {cam}', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    

if __name__ == '__main__':
    main()
