import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
from base_model import Digitizer
import pandas
import numpy as np

seed = 10
torch.manual_seed(seed)

# Set device
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()

# Get all data from train.csv
# Split into a training set and a development set
# Extract all the labels in both cases
# Restructure the input vectors into the 28x28 pixel images

image_height = 28
image_width = 28
total_pixels = image_height * image_width

df = pandas.read_csv("../data/train.csv")
num_datapoints = len(df.index)

t_all = np.zeros(num_datapoints, dtype=int)
X_all = np.zeros((num_datapoints, image_width, image_height), dtype=int)
for i in range(num_datapoints):
    t_all[i] = df['label'][i]
    for pix in range(total_pixels):
        x_pos = pix % image_width
        y_pos = pix // image_width
        X_all[i, x_pos, y_pos] = df['pixel'+str(pix)][i]

dev_fraction = 0.01
dev_index = int(num_datapoints * dev_fraction)
X_dev = X_all[:dev_index,:,:]
X = X_all[dev_index:,:,:]
# X = X_all[-1000:,:,:]
t_dev = t_all[:dev_index]
t = t_all[dev_index:]
# t = t_all[-1000:]

print("Finished preprocessing the data")
# Generic notation: X = input data; t = targets; y = predictions

X = torch.from_numpy(X)
X = X.float()

X_dev = torch.from_numpy(X_dev)
X_dev = X_dev.float()

t = torch.from_numpy(t)
t = t.long()

t_dev = torch.from_numpy(t_dev)
t_dev = t_dev.float()

# Mini-batch size
bs = 1000
epochs = 10
lr = 1e-1

# Store all training dataset in a single wrapped tensor
train_ds = TensorDataset(X, t)

# Use DataLoader to handle minibatches easily
train_dl = DataLoader(train_ds, batch_size = bs, shuffle = True)

# Construct model
my_model = Digitizer(image_height, image_width)
my_model = my_model.float()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(my_model.parameters(), lr=lr)

for epoch in range(epochs):
    my_model.train()
    total_loss = 0
    counter = 0
    for xb, tb in train_dl:

        # Forward pass
        y = my_model.forward(xb)
        # Compute CrossEntropyLoss
        loss = criterion(y, tb)

        # Zero gradients, backward pass, update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        counter+=1
        print(counter)

    # Report results at end of epoch
    avg_loss = total_loss/counter

    # Get accuracy values on dev set
    y_dev = my_model.forward(X_dev)
    print(y_dev.size())
    y_dev = y_dev.detach().numpy()
    y_dev_thresholded = np.argmax(y_dev, axis=1)
    print(y_dev_thresholded.shape)
    num_vals = len(t_dev)
    correct = 0
    for val in range(num_vals):
        if y_dev_thresholded[val] == t_dev[val]:
            correct += 1
    dev_acc = correct / num_vals


    print("Epoch: ", epoch, "Loss: ", avg_loss, "Dev accuracy: ", dev_acc)


# Save the model to a file
file_path = 'baseline_trained_seed'+str(seed)+'.pt'
torch.save(my_model, file_path)