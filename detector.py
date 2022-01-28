import sys

import cv2
import numpy as np
import time
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
        cv2.line(pic, (x, 0), (x, pic.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        xlist.append(x)
        x += pxstep

    while y <= pic.shape[0]:
        cv2.line(pic, (0, y), (pic.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
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
    neigboursDiag = []
    neigboursmin = []
    if xindex > 0:
        if yindex > 0:
            neigboursDiag.append((xlist[xindex - 1], ylist[yindex - 1]))
            neigboursmin.append((xminlist[xindex - 1], yminlist[yindex - 1]))
        if yindex < len(yminlist) - 1:
            neigboursDiag.append((xlist[xindex - 1], ylist[yindex + 1]))
            neigboursmin.append((xminlist[xindex - 1], yminlist[yindex + 1]))
    if xindex < len(xminlist) - 1:
        if yindex > 0:
            neigboursDiag.append((xlist[xindex + 1], ylist[yindex - 1]))
            neigboursmin.append((xminlist[xindex + 1], yminlist[yindex - 1]))
        if yindex < len(yminlist) - 1:
            neigboursDiag.append((xlist[xindex + 1], ylist[yindex + 1]))
            neigboursmin.append((xminlist[xindex + 1], yminlist[yindex + 1]))
    neigboursDiag = np.array(neigboursDiag)
    neigboursmin = np.array(neigboursmin)
    minx = neigboursmin[0, 0]
    miny = neigboursmin[0, 1]
    minxind = 0
    minyind = 0
    for i in range(len(neigboursmin)):
        if neigboursmin[i, 0] < minx:
            minx = neigboursmin[i, 0]
            minxind = i
        if neigboursmin[i, 1] < miny:
            miny = neigboursmin[i, 1]
            minyind = i
    x2 = neigboursDiag[minxind, 0]
    y2 = neigboursDiag[minyind, 1]
    return (x1, y1), (x2, y2), xindex, yindex


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

    # init artnet instance
    artnet = startStupidArtnet(ip_artnet)
    # artnet = startStupidArtnet('172.18.200.255')
    # Read model weights and config
    net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
    # Try use gpu+cuda, if possible
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

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
            start = time.time()
            classes, scores, boxes = model.detect(frame, 0.2, 0.4)
            end = time.time()

            start_drawing = time.time()
            for (classid, score, box) in zip(classes, scores, boxes):
                if class_names[int(classid)] not in white_list:
                    continue
                label = "%s : %f" % (class_names[classid[0]], score)
                x1y1 = tuple((np.array([box[0], box[1]])).astype(np.int32))
                x2y2 = tuple((np.array([box[2] + box[0], box[3] + box[1]])).astype(np.int32))
                center = (int((x1y1[0] + x2y2[0]) / 2), int((x1y1[1] + x2y2[1]) / 2))
                x1y1, x2y2, xind, yind = findNearest_(xlist, ylist, center)
                color = (255, 0, 0)
                # Print  outputs for test
                cv2.rectangle(frame, x1y1, x2y2, color, 2)
                cv2.rectangle(frame, box, color, 2)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            end_drawing = time.time()
            try:
                # send data on artnet
                # configered for middle light at underground floor
                artnet.set_single_value(17, 170 + xind)
                artnet.set_single_value(19, 32)
                artnet.set_single_value(24, 255)
                artnet.show()
            except:
                pass

            fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (
                1 / (end - start), (end_drawing - start_drawing) * 1000)
            cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imshow("detections", frame)
    artnet.stop()
    cap.release()
    cv2.destroyAllWindows()
