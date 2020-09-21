import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
from base_model import Digitizer
import pandas
import numpy as np
import csv

# Load up the trained model
model_path = 'baseline_trained_seed10.pt'
model = torch.load(model_path)
model.eval()

image_width = 28
image_height = 28
total_pixels = image_height * image_width

df = pandas.read_csv("../data/test.csv")
num_datapoints = len(df.index)

X = np.zeros((num_datapoints, image_width, image_height), dtype=int)
for i in range(num_datapoints):
    for pix in range(total_pixels):
        x_pos = pix % image_width
        y_pos = pix // image_width
        X[i, x_pos, y_pos] = df['pixel'+str(pix)][i]

print("Finished preprocessing the data")

X = torch.from_numpy(X)
X = X.float()

y = model.forward(X)
y = y.detach().numpy()
y_thresholded = np.argmax(y, axis=1)
print(y_thresholded.shape)

# Write results to a csv file
all_ids = np.arange(1, len(y)+1)

with open('submission.csv', 'w', newline='') as csvfile:
    fieldnames = ['ImageId', 'Label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for id in all_ids:
        writer.writerow({'ImageId': id, 'Label': y_thresholded[id-1]})

