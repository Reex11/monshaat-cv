# Innovation Center Booth Analyser
This is a project created by the **Innovation Center** team at **Monshaat** as a demo to the AI capabilities as one of the rising technologies the center focuses on.

## Team
- Marwan Alqisy
- Abdulmalik Almushaiqah
- Abdulaziz Abu Bakr

## WSL USBIPD Setup
For WSL users: 

- Install [USBIPD](https://github.com/dorssel/usbipd-win)

- List all USB devices connected to Windows using the command: (Powershell as Administrator)
    ```powershell
    usbipd list
    ```
- Bind the device to share it with WSL using the following command: (Powershell)
    ```powershell
    usbipd bind -b <bus_id>
    ```
    Replace <bus_id> with the bus ID of your device.
    Verify that the device is shared using the command `usbipd list`.


- Attach the device to WSL by entering the following command: (Powershell)
    ```powershell
    usbipd attach --wsl -b <bus_id>
    ```

- On the WSL terminal, list the USB devices using the command: (Bash in WSL)
    ```bash
    lsusb
    ```
    The device should be listed in the output.

- Setup WSL camera drivers:
    ```bash
    sudo apt-get install v4l2loopback-dkms
    sudo modprobe v4l2loopback
    ```

    [camera drivers](https://github.com/phuoctan4141/WSL/blob/main/Connect%20USB%20devices/USB%20Camera.md)
    
- Install libqt5gui5:
    ```bash
    sudo apt-get install libqt5gui5
    ```


- To detach the device from WSL, use the command: (Powershell)
    ```bash
    usbipd detach -b <bus_id>
    ```

## Requirements

- Python 3.8
- Install the requirements:
    ```bash
    pip install -r requirements.txt
    ```

### To run on Nvidia Jetson based devices:
Refer to the [Ultralytics Guide to run with NVIDIA Jetson](https://docs.ultralytics.com/guides/nvidia-jetson)



## Usage
To run the project:
```bash
python main.py
```

