## #94 Распознавание людей на сцене и их автоматическая подсветка DMX-фонарем

Инструции по развертке находятся внутри папок с вариантами программы.

Перед работой необходимо поместить конфиг-файлы и файлы весов в один каталог с программой:

[Скачать веса](https://pjreddie.com/media/files/yolov3.weights) разных версий yolo_v3

[Скачать архитектуру и веса](https://github.com/AlexeyAB/darknet) yolo_v4. В работе используется версия yolo4-tiny


Конфиг [yolov4-tiny.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg)

Веса [yolov4-tiny.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)

### Алгоритм:
- Устройство на котором размещена программа находится в одной сети с камерой и ArtNet контроллером
- Инициализируется модель нейронной сети через веса (.weights) и конфиг (.conf)
- С камеры забирается RTSP поток, покадрово. (rtsp адрес камеры и широковещательный адрес сети ArtNet контроллера передаются в качестве аргумента при запуске из командной строки)
- На обработку нейронной сети подается либо каждый кадр, либо каждый N кадр (определяется аргументом при запуске)
- Нейронная сеть идентифицирует много классов объектов - на следующий этап отделяются объекты класса 'Человек'
- Для каждого классифицированного объекта определяется зона интереса - две точки, определяющие углы прямоугольника.
- Протокол DMX предполагает передачу сигнала в дискретном наборе значений 0..255. Всё пространство сцены можно условно разделить на ячейки, которые будут однозначно определять положение фонаря. Например: от левой стены сцены до правой фонарь передвигается на 16 значений сигнала, в таком случае по горизонтальной оси будет 15 значений. 
- Для зоны интереса вокруг человека определяется её центр и ячейка решетки, после чего сигнал о перемещении фонаря в эту ячейку передается контроллеру.

Пример запуска
```commandline
python detector_wo_img.py 192.168.10.10 192.168.11.255 15 2
```
    
## Версия с OpenCV 

[Source](https://gist.github.com/YashasSamaga/e2b19a6807a13046e399f4bc3cca3a49)

windows:

    python3 -m venv /path/to/new/virtual_environment
    virtual_environment\Scripts\activate.bat
    
Linux

     python3 -m venv /path/to/new/virtual_environment
     source virtual_environment/bin/activate
       
Зависимости:

    pip install -r requirements.txt

Run 
    
    python3 detector.py "RTSP camera address" "broadcast artnet ip" lights-position frame-count
  
For middle DMX-light at -1 floor lights-position = 7