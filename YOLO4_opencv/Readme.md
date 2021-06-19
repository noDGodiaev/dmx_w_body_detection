## OpenCV only version, yhe highest spped if run on CPU

[Source](https://gist.github.com/YashasSamaga/e2b19a6807a13046e399f4bc3cca3a49)

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
