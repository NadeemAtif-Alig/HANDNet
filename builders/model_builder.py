from model.handnet import Network


def build_model(model_name, num_classes):
    if model_name == 'handnet':
        return Network(num_classes=num_classes)

