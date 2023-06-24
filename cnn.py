# some preprocessing steps are used from these sources:
# => https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
# => https://www.youtube.com/watch?v=j-3vuBynnOE&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN&index=2 

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
import cv2 as cv
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

# make debugging easier
def debug(name, data):
    print(f'{name} => {data}')

def calculate_center_of_mass(image):
    rows, cols = image.shape
    total_mass = np.sum(image)
    
    # calculate weighted sum of row and column indices
    row_indices = np.arange(rows).reshape((-1, 1))
    col_indices = np.arange(cols).reshape((1, -1))
    sum_row_indices = np.sum(row_indices * image)
    sum_col_indices = np.sum(col_indices * image)
    
    # calculate center of mass coordinates
    center_row = sum_row_indices / total_mass
    center_col = sum_col_indices / total_mass
    
    return center_row, center_col

def warpAffine(image, M, output_shape):
    rows, cols = image.shape
    out_rows, out_cols = output_shape

    # create output image with the specified shape
    output_image = np.zeros(output_shape, dtype=image.dtype)
    
    for out_row in range(out_rows):
        for out_col in range(out_cols):
            # calculate the corresponding input pixel coordinates using the inverse transformation matrix
            in_col, in_row, _ = np.dot(np.linalg.inv(M), np.array([out_col, out_row, 1]))

            in_col = int(round(in_col))
            in_row = int(round(in_row))

            # check if the input coordinates are within the input image bounds
            if 0 <= in_row < rows and 0 <= in_col < cols:
                # copy the pixel value from the input image to the corresponding location in the output image
                output_image[out_row, out_col] = image[in_row, in_col]
    
    return output_image

# centralize the image according to its center of mass
def get_best_shift(image):
    cy, cx = calculate_center_of_mass(image)

    rows, cols = image.shape
    shiftx = int(round((cols / 2.0) - cx))
    shifty = int(round((rows / 2.0) - cy))

    return shiftx, shifty

# shift the image in the given directions
def shift(image, shiftx, shifty):
    rows, cols = image.shape
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    M = np.vstack((M, [0, 0, 1]))

    # translation (shift the object location)
    shifted = warpAffine(image, M, (cols, rows))

    return shifted

# centralization for sudoku digit
# according to their center of mass, so that all digit images are in the SAME form
def shift_acc_center_of_mass(image):
    image = ~image

    # centralize according to center of mass
    shiftx, shifty = get_best_shift(image)
    shifted = shift(image, shiftx, shifty)
    image = shifted

    image = ~image

    return image

num_classes = 9
img_rows, img_cols = 28, 28

mnist_train = datasets.MNIST('./mnist_data', train=True, download=True)
mnist_test = datasets.MNIST('./mnist_data', train=False, download=True)

filtered_train_data = [(image, label) for image, label in zip(mnist_train.data, mnist_train.targets) if label != 0]
filtered_train_images, filtered_train_labels = zip(*filtered_train_data)
mnist_x_train = torch.stack(filtered_train_images).unsqueeze(3)
mnist_y_train = torch.tensor(filtered_train_labels)

# filter out images with labels 0 from the test dataset
filtered_test_data = [(image, label) for image, label in zip(mnist_test.data, mnist_test.targets) if label != 0]
filtered_test_images, filtered_test_labels = zip(*filtered_test_data)
mnist_x_test = torch.stack(filtered_test_images).unsqueeze(3)
mnist_y_test = torch.tensor(filtered_test_labels)

# train_split_index = int(mnist_x_train.size(0) * 0.3)
# mnist_x_train = mnist_x_train[:train_split_index, :, :, :]
# mnist_y_train = mnist_y_train[:train_split_index]

# test_split_index = int(mnist_x_test.size(0) * 0.4)
# mnist_x_test = mnist_x_test[:test_split_index, :, :, :]
# mnist_y_test = mnist_y_test[:test_split_index]

print(mnist_x_train.shape, mnist_y_train.shape)
print(mnist_x_test.shape, mnist_y_test.shape)

DATADIR = "./data"
CATEGORIES = ["1","2","3","4","5","6","7","8","9"] # used to get the corresponding data folder and for determining the class labels

# read training data (stored in folders named 1, 2, ..., 9 in ./data directory)
data = []
def create_dataset():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv.imread(os.path.join(path, img), cv.IMREAD_GRAYSCALE)
            new_array = cv.resize(img_array, (img_rows, img_cols))
            # show before
            # plt.imshow(img_array, cmap='gray')
            # plt.show()
            new_array = shift_acc_center_of_mass(new_array)
            # show after
            # plt.imshow(new_array, cmap='gray')
            # plt.show()
            data.append([new_array, class_num])

    random.shuffle(data) # mixing the data
    print(f'total training data: {len(data)}')

create_dataset()

# split the data training 80 - test 20
x_train = []
y_train = []
x_test = []
y_test = []

tot = set((x[1] for x in data))
debug("all labels", tot)

n1 = int(0.8*len(data))
x_train = np.array([d[0] for d in data[:n1]])
y_train = np.array([d[1] for d in data[:n1]])

x_test = np.array([d[0] for d in data[n1:]])
y_test = np.array([d[1] for d in data[n1:]])

# convert data to numpy array and reshape
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1) # (samples, rows, cols, channels)
x_train = x_train.astype('float32')

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1) # (samples, rows, cols, channels)
x_test = x_test.astype('float32')

ck_train_x = x_train.copy()
ck_test_x = x_test.copy()
ck_train_y = y_train.copy()
ck_test_y = y_test.copy()

ck_train_x /= 255
ck_test_x /= 255

# one hot encoding the labels
cktr_oh = np.zeros((ck_train_y.size, 9))
cktr_oh[np.arange(ck_train_y.size), ck_train_y] = 1

ckte_oh = np.zeros((ck_test_y.size, 9))
ckte_oh[np.arange(ck_test_y.size), ck_test_y] = 1

# convert to tensor
cktr_oh = torch.tensor(cktr_oh)
ckte_oh = torch.tensor(ckte_oh)

ck_train_y = torch.tensor(ck_train_y)
ck_test_y = torch.tensor(ck_test_y)

# reformat to [size, channels, rows, cols]
cktr = torch.tensor(ck_train_x).transpose(3, 1).float()
ckte = torch.tensor(ck_test_x).transpose(3, 1).float()

# combined data
all_x_train = torch.cat((torch.from_numpy(x_train), mnist_x_train), dim=0)
all_x_test = torch.cat((torch.from_numpy(x_test), mnist_x_test), dim=0)
all_y_train = torch.cat((torch.from_numpy(y_train), mnist_y_train), dim=0).long()
all_y_test = torch.cat((torch.from_numpy(y_test), mnist_y_test), dim=0).long()

debug("x train", len(all_x_train))
debug("x test", len(all_x_test))
debug("y train", len(all_y_train))
debug("y test", len(all_y_test))

input_shape = (img_rows, img_cols, 1)

# normalize the data
all_x_train /= 255
all_x_test /= 255

all_y_train -= 1
all_y_test -= 1

debug("before train labels", all_y_train[:5])
debug("before test labels", all_y_test[:5])

# one hot encoding the labels
ytr = np.zeros((all_y_train.size(0), 9))
ytr[np.arange(all_y_train.size(0)), all_y_train] = 1

yte = np.zeros((all_y_test.size(0), 9))
yte[np.arange(all_y_test.size(0)), all_y_test] = 1

debug("after train labels", ytr[:5])
debug("after test labels", yte[:5])

xtr = all_x_train.transpose(3, 1).float()
y_train = all_y_train.long() # original
ytr = torch.from_numpy(ytr).float() # one hot encoded

xte = all_x_test.transpose(3, 1).float()
y_test = all_y_test.long() # original
yte = torch.from_numpy(yte).float() # one hot encoded

# creating the convolutional network
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout2d(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

batch_size = 64
model = ConvNet(num_classes)
loss_fn = nn.CrossEntropyLoss()

def train_model(x_train, y_train_oh, y_train, x_test, y_test, epochs, lr, save_model_as):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # for plotting purposes
    training_loss = []
    training_accuracy = []
    test_accuracy = []

    for epoch in (t := trange(epochs)):
        # training mode
        model.train()
        total_loss = 0.0
        correct = 0

        for i in range(0, xtr.shape[0], batch_size):
            X = x_train[i:i+batch_size].clone()
            Y = y_train_oh[i:i+batch_size] # one hot labels
            y = y_train[i:i+batch_size] # actual labels

            # forward pass
            outs = model(X)

            # compute loss
            loss = loss_fn(outs, Y)
            total_loss += loss.item()

            # backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the training progress
            predicted = torch.argmax(outs, 1)

            correct += (predicted == y).sum().item()
    
        train_acc = correct / x_train.shape[0]
        training_accuracy.append(train_acc)
        total_loss /= x_train.shape[0]
        training_loss.append(total_loss)

        scheduler.step()

        t.set_description(f'training loss: {total_loss:.4f}, training accuracy: {train_acc:.2f}')

        # evaluation mode
        model.eval()
        with torch.no_grad():
            X = x_test
            Y = y_test
            outs = model(X)
            test_y_preds = torch.argmax(outs, 1)
            test_acc = (test_y_preds == y_test).sum().item() / y_test.shape[0]

            test_accuracy.append(test_acc)

            print(f'test accuracy: {test_acc}')

    # plotting the graphs
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_loss, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(training_accuracy, label='Training Accuracy')
    plt.plot(test_accuracy, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    # saving the model; will be later used for predicting digits captured in webcam
    torch.save(model.state_dict(), save_model_as)

train_model(cktr, cktr_oh, ck_train_y, ckte, ck_test_y, 60, 0.001, "cnn.pt") # for just chars74k dataset
# train_model(xtr, ytr, all_y_train, xte, yte, 100, 0.001, "cnn2.pt") # for combined dataset
