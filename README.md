# Real-Time-Sudoku-Solver

## Overview
- I knew almost nothing about Image Processing and Machine Learning when I started this project. Inspired by an amazing [implementation](https://www.youtube.com/watch?v=QR66rMS_ZfA) on YouTube by geaxgx1, I decided to make my own.

- This project is not perfect. The solution written on the image in real time will fluctuate a little bit. 
- You will need to keep your computer and the sudoku board very stable to see proper results.
- Along with that, the camera quality and the lighting around you will also affect the results you see.

- List of resources used to to build this project:
  * Data used to train the CNN : http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
  * OpenCV Tutorial Playlist : https://www.youtube.com/watch?v=kdLM6AOd2vc&list=PLS1QulWo1RIa7D1O6skqDQ-JZ1GGHKK-K
  * Deep Learning Playlist : https://www.youtube.com/watch?v=wQ8BIBpya2k&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN
  * Improving Digit Recognition Accuracy : https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
  * Finding Sudoku Board Corners from Contours : https://www.programcreek.com/python/example/89417/cv2.arcLength
  * Extracting Sudoku Grid : https://maker.pro/raspberry-pi/tutorial/grid-detection-with-opencv-on-raspberry-pi
  * Interesting series of articles on Sudoku Grabber : https://aishack.in/tutorials/sudoku-grabber-opencv-plot/
  * Perspective Warping : https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/
  * OpenCV Documentation : https://docs.opencv.org/2.4/doc/tutorials/tutorials.html
  * Errors and stuff : https://stackoverflow.com/

## Installations needed to run the project?
- Python 3
- OpenCV 4
- PyTorch

## How to run the project?
- Make sure you have all the required installations.
- Download all the files and run the **main.py** file.
- **Note - You don't need to train the CNN on your system. I have trained a model and stored the architecture in cnn.pt**
- **Note - If you want to try training the CNN, you will need to download the [Chars74K dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/), take images 1-9 (We only need 1-9 for the sudoku) and put them in folders named "1","2",....,"9" respectively in the same directory where all my python files are stored. 
Finally, run the cnn.py file.**
