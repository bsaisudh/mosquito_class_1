from torchvision import models
import torch

from mosquito_class_1.zoo.CNN import CNN

def generate_model(model_args, device):
    model = None
    if model_args["PRETRAINED"] == 1:
        pt = True
    else:
        pt = False

    if model_args["TYPE"] == "vgg16":
        model = models.vgg16(pretrained=pt)
        model.to(device)
        n_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(n_features, 6)

    elif model_args["TYPE"] == "cnn":
        model = CNN()
        model.to(device)

    elif model_args["TYPE"] == "resnet50":
        model = models.resnet50(pretrained=pt)
        model.to(device)

        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, 6)

    elif model_args["TYPE"] == "squeezenet":
        model = models.squeezenet1_0(pretrained=pt)
        model.to(device)

        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Conv2d(512, 6, kernel_size=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AvgPool2d(13, stride=1)
        )
        model.num_classes = 6

    elif model_args["TYPE"] == "alexnet":
        model = models.alexnet(pretrained=pt)
        model.to(device)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(256 * 6 * 6, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, 6),
        )

    else:
        print("Model ", model_args["TYPE"], " not supported.")
        exit()

    return model
