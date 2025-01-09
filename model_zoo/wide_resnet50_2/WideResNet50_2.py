import argparse
import os
import torch
from torchvision import models, transforms
from PIL import Image

def Classificate(config):
    filename = os.path.join(config.path, config.filename)
    print(filename)

    model = models.wide_resnet50_2(weights = models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.486], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(Image.open(filename))
    input_batch = input_tensor.unsqueeze(0)

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
        
    with torch.no_grad():
        output = model(input_batch)

    #print(output[0])

    probabilities = torch.nn.functional.softmax(output[0], dim = 0)
    prob, cat = torch.topk(probabilities, 10)

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    for i in range(prob.size(0)):
        print(categories[cat[i]], prob[i].item())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', default='IMG_1133.jpg')
    parser.add_argument('-p', '--path', default='image')
    Classificate(parser.parse_args())
