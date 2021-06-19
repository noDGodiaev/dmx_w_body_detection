## OpenCV only version, the highest spped if run on CPU

[Source](https://wiki.loliot.net/docs/lang/python/libraries/yolov4/python-yolov4-about/)

Before install

On windows:

    python3 -m venv /path/to/new/virtual_environment
    virtual_environment\Scripts\activate.bat
    
On Linux

     python3 -m venv /path/to/new/virtual_environment
     source virtual_environment/bin/activate
       
Dependencies:

    pip install -r requirements.txt


Run 
    
    python3 detector.py "RTSP camera address" "broadcast artnet ip" lights-position frame-count
  
For middle DMX-light at -1 floor lights-position = 7
