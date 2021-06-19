import sys
from yolov4.tf import YOLOv4
import cv2
import numpy as np
from stupidArtnet.lib.StupidArtnet import StupidArtnet


def startStupidArtnet(broadcast_ip):
    target_ip = broadcast_ip
    universe = 0  # see docs
    packet_size = 100
    a = StupidArtnet(target_ip, universe, packet_size, 30, True, True)
    return a


def drawGrid(pic, n, m, line_color=(255, 255, 0), thickness=1, type_=cv2.LINE_AA):
    """
    :param pic:  captured frame
    :param n:  number of vertical lines
    :param m:  number of horizontal lines
    :param line_color:
    :param thickness:
    :param type_:
    :return: lists with node coordinates
    """
    pxstep = int(pic.shape[1] / n)
    pystep = int(pic.shape[0] / m)
    x = 0
    y = 0
    xlist = []
    ylist = []
    while x <= pic.shape[1]:
        xlist.append(x)
        x += pxstep

    while y <= pic.shape[0]:
        ylist.append(y)
        y += pystep
    return xlist, ylist


def findNearest_(xlist, ylist, center):
    """
    :param xlist: x coordinates of grid nodes
    :param ylist: y coordinates of grid nodes
    :param center: center of captured human
    :return: (x1, y1), (x2, y2) coordinates of grid rectangle to draw
            xindex, yindex index of grid part, where person was detected
    """
    xminlist = np.abs(np.subtract(xlist, center[0]))
    yminlist = np.abs(np.subtract(ylist, center[1]))
    x1 = xlist[np.argmin(xminlist)]
    y1 = ylist[np.argmin(yminlist)]
    xindex = np.argmin(xminlist)
    yindex = np.argmin(yminlist)
    return xindex, yindex


if __name__ == "__main__":
    # RTSP address of camera
    # if 0 use first found device
    rtsp_camera = sys.argv[1]
    # Broadcast ip with Artnet controller inside
    ip_artnet = sys.argv[2]
    # Number of light's position
    horizontal_split = sys.argv[3]
    # Take each frame
    frame_cut = sys.argv[4]

    class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
                   "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                   "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                   "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                   "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                   "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                   "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                   "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
                   "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                   "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    white_list = ["person"]
    if rtsp_camera == 0:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(rtsp_camera)
    # cap = cv2.VideoCapture('rtsp://172.18.191.159:554/12')
    # cap = cv2.VideoCapture('rtsp://admin:Supervisor@172.18.200.54:554/Streaming/Channels/1')

    # init artnet instance
    artnet = startStupidArtnet(ip_artnet)
    # artnet = startStupidArtnet('172.18.200.255')
    # Read model weights and config
    yolo = YOLOv4()
    yolo.config.parse_names("coco.names")
    yolo.config.parse_cfg("yolov4-tiny-tf.cfg")
    yolo.make_model()
    yolo.load_weights("yolov4-tiny.weights", weights_type="yolo")

    framecount = 0
    while cv2.waitKey(1) < 1:
        # get frame
        ret, frame = cap.read()
        if not ret:
            print('Video not captured')
            exit()
        # get nodes values list
        xlist, ylist = drawGrid(frame, int(horizontal_split), 1)
        framecount += 1
        # Take every 'frame_cut' frame
        if framecount % int(frame_cut) == 0:
            results = yolo.predict(frame, 0.2)
            for res in results:
                x, y, w, h, classid, score = res
                if class_names[int(classid)] not in white_list:
                    continue
                x = x * frame.shape[1]
                y = y * frame.shape[0]
                w = w * frame.shape[1]
                h = h * frame.shape[0]
                x1y1 = (int(x - w / 2), int(y - h / 2))
                x2y2 = (int(x + w / 2), int(y + h / 2))
                center = (int(x1y1[0] + x2y2[0] / 2), int(x1y1[1] + x2y2[1] / 2))
                xind, yind = findNearest_(xlist, ylist, center)
            try:
                # send data on artnet
                # configered for middle light at underground floor
                artnet.set_single_value(17, 170 + xind)
                artnet.set_single_value(19, 32)
                artnet.set_single_value(24, 255)
                artnet.show()
            except:
                pass
    artnet.stop()
    cap.release()
    cv2.destroyAllWindows()
