import argparse
import os
from ultralytics import YOLO

def Inference(config):
    file = os.path.join(config.path, config.filename)
    print(file)
    model = YOLO('yolov8m-seg.pt')
    results = model.predict(source=file, imgsz=config.size)
    for result in results:
        #print(result.masks)
        result.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', default='IMG_1133.jpg')
    parser.add_argument('-p', '--path', default='../../images')
    parser.add_argument('-s', '--size', default=480)
    Inference(parser.parse_args())



