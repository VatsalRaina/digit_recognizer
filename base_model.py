import torch
import torchvision.models as models

class Digitizer(torch.nn.Module):
    def __init__(self, image_height, image_width):

        super(Digitizer, self).__init__()

        self.image_width = image_width
        self.image_height = image_height

        self.resnet18 = models.resnet18()
        self.resnet18.train()

        self.final_layer = torch.nn.Linear(1000, 10)

    def forward(self, X):

        # Repeat first channel to make image three channels
        X_3 = torch.unsqueeze(X, dim=1)
        X_3_repeated = X_3.expand(-1, 3, -1, -1)

        # Pass through resnet-18
        y_1000 = self.resnet18(X_3_repeated)
        y_10 = self.final_layer(y_1000)
        y = torch.sigmoid(y_10)

        return y
