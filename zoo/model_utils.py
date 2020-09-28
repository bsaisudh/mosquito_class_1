from torchvision import models
import torch


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

    if model_args["TYPE"] == "cnn":
        pass

    return model
