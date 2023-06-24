# ----- Entry point!! Run this file!! -----

# No need of running the cnn.py file to train the Convolutional Neural Network (CNN).
# The trained architecture is saved in cnn.pt

import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import sudoku_solver

# ---------- Start Video and Print Solved Image ----------

# load and set up camera
cap = cv.VideoCapture(0)

# HD camera
# cap.set(3, 1280)
# cap.set(4, 720)

# creating the convolutional network
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout2d(0.5)
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

num_classes = 9
model = ConvNet(num_classes)

# Loading model seperately to speed up prediction
# load pre-trained model (model trained in digitRecognition.py)
# saved_model = torch.load('digitRecognition.pt', map_location=torch.device('cpu'))
model.load_state_dict(torch.load('cnn.pt', map_location=torch.device('cpu')))
model.eval()

# define codec to save video
# out = cv.VideoWriter("solution_video.avi", cv.VideoWriter_fourcc('X','V','I','D'), 20.0, (640,480))

prev_frame = None

old_sudoku = None
solution_frame = None
while True:
    # read the frame
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    sudoku_frame, sudoku = sudoku_solver.recognize_and_solve(frame, model, old_sudoku, solution_frame)
    if sudoku:
        old_sudoku = sudoku[:]
        solution_frame = sudoku_frame.copy()

    if prev_frame is not None and np.array_equal(sudoku_frame, prev_frame):
        pass
    else:
        prev_frame = sudoku_frame.copy()

        # print the solved image
        new_image = np.copy(sudoku_frame)
        new_image = cv.resize(new_image, (1000, 600))
        cv.imshow("CS228 Sudoku Solver", new_image)

    # save the video
    # comment this line if you don't want to save video
    # out.write(frame)

    # waitKey(1) for continuously capturing images and quit by pressing 'q' key.
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    

# release unneeded resources
cap.release()
# out.release()
cv.destroyAllWindows()
