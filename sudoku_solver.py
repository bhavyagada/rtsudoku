import math
import copy
import cv2 as cv
import numpy as np
import torch

# make debugging easier
def debug(name, data):
    print(f'{name} => {data}')

# ---------- Solving Sudoku ----------

# using Best-First search, which is an optimized version of backtracking;
# the "next cell" we choose is the cell which has the least number of possibilities
# the "number of possibilities" is calculated for each cell, by going through its corresponding row, column, and 3x3 block
# we count the number of total digits which are not chosen for the current cell
# this greedy heuristic minimizes the branching factor

# tracking "best" cell data
class Cell:
    def __init__(self, r, c, n):
        self.row = r
        self.col = c
        self.choices = n

    def set_data(self, r, c, n):
        self.row = r
        self.col = c
        self.choices = n

# check if the sudoku puzzle is solvable
def is_valid(matrix, row, col):
    # check the row
    for c in range(9):
        if matrix[row][col] != 0 and col != c and matrix[row][col] == matrix[row][c]:
            return False

    # check the column
    for r in range(9):
        if matrix[row][col] != 0 and row != r and matrix[row][col] == matrix[r][col]:
            return False

    # check the subgrids
    r = row // 3
    c = col // 3
    for i in range(r * 3, r * 3 + 3):
        for j in range(c * 3, c * 3 + 3):
            if row != i and col != j and matrix[i][j] != 0 and matrix[i][j] == matrix[row][col]:
                return False
    
    return True

# helper function for best first search
def best_first_search(matrix, flag):
    if not flag[0]: # stop
        return

    # find the entry which has the least possibilities
    best_candidate = Cell(-1, -1, 100)
    for i in range(9):
        for j in range(9):
            if matrix[i][j] == 0: # if not filled
                used = set() # count possibilities

                # check the row and column
                for k in range(9):
                    used.add(matrix[i][k])
                    used.add(matrix[k][j])

                # check the 3x3 square
                start_row = (i // 3) * 3
                start_col = (j // 3) * 3
                for x in range(start_row, start_row + 3):
                    for y in range(start_col, start_col + 3):
                        used.add(matrix[x][y])

                choices_left = 9 - len(used)
                if best_candidate.choices > choices_left:
                    best_candidate.set_data(i, j, choices_left)

    # if no choices found, it means...
    # all board cells are filled, we're done with the best-first search!
    if best_candidate.choices == 100:
        # set the flag so that the rest of the recursive calls can stop
        flag[0] = False
        return

    row = best_candidate.row
    col = best_candidate.col

    # if best candidate is found, try to fill it with 1 -> 9
    for j in range(1, 10):
        if not flag[0]: # stop
            return

        matrix[row][col] = j

        if is_valid(matrix, row, col):
            best_first_search(matrix, flag)

    if not flag[0]: # stop
        return

    matrix[row][col] = 0 # backtracking! mark the current cell empty so that the next digit can be checked

# best-first search
def solve_sudoku(matrix):
    flag = [True] # using array because it is mutable across functions

    # is it possible to have a solution?
    for i in range(9):
        for j in range(9):
            if not is_valid(matrix, i, j): # if not possible, stop
                return

    best_first_search(matrix, flag) # try to solve the puzzle

    if is_sudoku_solved(matrix):
        return True
    else:
        return False

def is_sudoku_solved(matrix):
    # check the rows
    for row in matrix:
        if set(row) != set(range(1, 10)):
            return False

    # check the columns
    for col in range(9):
        column_values = [matrix[row][col] for row in range(9)]
        if set(column_values) != set(range(1, 10)):
            return False

    # check the subgrids
    for row_offset in range(0, 9, 3):
        for col_offset in range(0, 9, 3):
            subgrid_values = []
            for row in range(row_offset, row_offset + 3):
                for col in range(col_offset, col_offset + 3):
                    subgrid_values.append(matrix[row][col])
            if set(subgrid_values) != set(range(1, 10)):
                return False

    return True

# check if the board is filled with non-zero digits; if yes, the current board state is the solution
def is_board_filled(matrix):
    for i in range(9):
        for j in range(9):
            if matrix[i][j] == 0:
                return False

    return True

# tests to see if the all functions work properly
x = [[0, 0, 4, 0, 8, 7, 0, 0, 0], 
     [0, 0, 0, 4, 3, 0, 7, 0, 1], 
     [5, 0, 0, 0, 0, 0, 0, 8, 0], 
     [0, 0, 7, 3, 0, 0, 0, 0, 0], 
     [4, 0, 0, 0, 0, 2, 8, 0, 5], 
     [2, 0, 1, 0, 0, 0, 0, 7, 6], 
     [0, 4, 2, 0, 7, 6, 5, 1, 9],
     [0, 8, 9, 5, 2, 3, 0, 4, 7],
     [0, 6, 5, 1, 0, 0, 0, 0, 0]]

x_soln = [[6, 1, 4, 2, 8, 7, 9, 5, 3], 
          [9, 2, 8, 4, 3, 5, 7, 6, 1], 
          [5, 7, 3, 6, 9, 1, 2, 8, 4], 
          [8, 5, 7, 3, 6, 4, 1, 9, 2], 
          [4, 9, 6, 7, 1, 2, 8, 3, 5], 
          [2, 3, 1, 9, 5, 8, 4, 7, 6], 
          [3, 4, 2, 8, 7, 6, 5, 1, 9], 
          [1, 8, 9, 5, 2, 3, 6, 4, 7], 
          [7, 6, 5, 1, 4, 9, 3, 2, 8]]

# assert is_board_filled(x_soln) == True, "board filled check not working!"
# print(solve_sudoku(x))
# assert x == x_soln, "sudoku solver not working!"
# assert is_sudoku_solved(x_soln) == True, "sudoku solved check not working!"

# ---------- Image Processing ----------

def calculate_sum(arr):
    total_sum = 0
    for item in arr:
        if isinstance(item, list):
            total_sum += calculate_sum(item)
        else:
            total_sum += item
    return total_sum

# write the solution on "image"
def write_soln_on_image(image, grid, user_grid):
    text_updated = False
    threshold = 0.5
    prev_bottom_left_x = prev_bottom_left_y = 0.0
    width = image.shape[1] // 9
    height = image.shape[0] // 9

    for i in range(9):
        for j in range(9):
            if user_grid[i][j] != 0: # skip if the cell is already filled
                continue

            # convert digit to string
            digit = str(grid[i][j])
            offset_x = width // 15
            offset_y = height // 15
            font = cv.FONT_HERSHEY_SIMPLEX
            (text_height, text_width), _ = cv.getTextSize(digit, font, fontScale=1, thickness=3) # determine the text size

            scale = 0.6 * min(width, height) / max(text_height, text_width)
            text_height *= scale
            text_width *= scale

            # determine the bottom left corner of the puzzle grid
            bottom_left_x = width*j + math.floor((width - text_width) / 2) + offset_x
            bottom_left_y = height*(i+1) - math.floor((height - text_height) / 2) + offset_y

            if (abs(bottom_left_x - prev_bottom_left_x) > threshold * width or abs(bottom_left_y - prev_bottom_left_y) > threshold * height):
                text_updated = True

            # LINE_AA minimizes distortion in the drawn text
            image = cv.putText(image, digit, (bottom_left_x, bottom_left_y), font, scale, (0, 255, 0), thickness=3, lineType=cv.LINE_AA)

            prev_bottom_left_x = bottom_left_x
            prev_bottom_left_y = bottom_left_y
    
    if text_updated:
        return image
    else:
        return None

# compare 2 matrices (old frame and current frame) and return if all corresponding entries are equal
def is_mat_equal(matrix_1, matrix_2, row, col):
    for i in range(row):
        for j in range(col):
            if matrix_1[i][j] != matrix_2[i][j]:
                return False

    return True

# criteria 1 for detecting contours (outline/border) on the sudoku board,
# length of sides CANNOT be too different in size; because the sudoku puzzle has a square board
def are_side_lengths_too_different(points, eps_scale):
    AB = math.sqrt((points[0][0] - points[1][0])**2 + (points[0][1] - points[1][1])**2)  # side 1
    AD = math.sqrt((points[0][0] - points[3][0])**2 + (points[0][1] - points[3][1])**2)  # side 2
    BC = math.sqrt((points[1][0] - points[2][0])**2 + (points[1][1] - points[2][1])**2)  # side 3
    CD = math.sqrt((points[2][0] - points[3][0])**2 + (points[2][1] - points[3][1])**2)  # side 4

    shortest = min(AB, AD, BC, CD)  # shortest side
    largest = max(AB, AD, BC, CD)  # longest side

    return largest > eps_scale * shortest


# criteria 2 for detecting contours (outline/border) on the sudoku board,
# all corner angles have to be 90 degrees approximately with epsilon tolerance (roundoff)
def is_approx_90_deg(angle, epsilon):
    return abs(angle - 90) < epsilon

# dividing the sudoku board into 9x9 small square images (cells)
# each cell will be a "image_crop"
# separate digit from noise in "image_crop"
def max_component(image):
    image = image.astype('uint8')

    # get the labelled image and statistics for each image
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, cv.CC_STAT_AREA]

    if len(sizes) <= 1:
        empty_image = np.zeros(image.shape)
        empty_image.fill(255)
        return empty_image
    
    # we start from component 1 (instead of 0) because we want to leave out the background
    maximum_label, maximum_size = 1, sizes[1]

    for i in range(2, num_labels):
        if sizes[i] > maximum_size:
            maximum_label = i
            maximum_size = sizes[i]
    
    image2 = np.zeros(labels.shape)
    image2.fill(255)
    image2[labels == maximum_label] = 0
    return image2

# finding angle between two vectors to check if corners are 90 degrees
def get_angle(v1, v2):
    # linalg is used for linear algebra
    # norm function returns the length of vector
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)

    return angle * 57.2958 # convert to degree

def warp_affine(image, M, output_shape):
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

# calculating how to centralize using center of mass of image
def best_shift(image):
    # cy, cx = calculate_center_of_mass(image)
    rows, cols = image.shape

    # calculating the center of mass of the image
    total_mass = np.sum(image)
    
    # calculate weighted sum of row and column indices
    row_indices = np.arange(rows).reshape((-1, 1))
    col_indices = np.arange(cols).reshape((1, -1))
    sum_row_indices = np.sum(row_indices * image)
    sum_col_indices = np.sum(col_indices * image)
    
    # calculate center of mass coordinates
    center_row = sum_row_indices / total_mass
    center_col = sum_col_indices / total_mass

    shiftx = np.round((cols / 2.0) - center_row).astype(int)
    shifty = np.round((rows / 2.0) - center_col).astype(int)

    return shiftx, shifty

# shift image based on best shift values
def shift(image, sx, sy):
    rows, cols = image.shape
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    M = np.vstack((M, [0, 0, 1]))

    shifted = warp_affine(image, M, (cols, rows))

    return shifted

# getting the contours (outline/border) of the sudoku board
def get_contour_corners(contours, total_corners=4, total_iterations=200):
    coeff = 1
    while total_iterations > 0 and coeff >= 0:
        total_iterations = total_iterations - 1
        # maximum distance from contour to approximated contour
        epsilon = coeff * cv.arcLength(contours, True)
        
        # approximation of contour shape (True means curve is closed)
        approx = cv.approxPolyDP(contours, epsilon, True)
        
        # check curves for convexity defects
        hull = cv.convexHull(approx)
        if len(hull) == total_corners:
            return hull
        else:
            if len(hull) > total_corners:
                coeff += .01
            else:
                coeff -= .01

    return None

# make the digit prediction
def predict(model, input_data):
    # convert to tensor
    input_tensor = torch.tensor(input_data)

    # reformat to [size, channels, rows, cols]
    input_tensor = input_tensor.transpose(3, 1).float()
    # debug("input image shape", input_tensor.shape)

    # predict the cell digit
    out = model(input_tensor)
    prediction = torch.argmax(out).item()
    # debug("prediction", prediction)

    return prediction

# taking a webcam image, finding the sudoku board, recognizing digits and solving the puzzle,
# finally, print the result on image and return that image
def recognize_and_solve(image, model, old_sudoku, solution_warp):
    # convert image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # make the image smooth using gaussian blur
    # kernel size is 5, 0 is for auto-completion of sigma value
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    
    # adaptive thresholding
    adaptive_threshold = cv.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    
    # finding all contours (outline/border)
    # contours is a list of all contours in image (numpy array of (x, y) coordinates of boundary points)
    contours, _ = cv.findContours(adaptive_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    biggest = None
    for c in contours:
        area = cv.contourArea(c)
        if area > max_area:
            max_area = area
            biggest = c
    
    # if there is no sudoku in the image!
    if biggest is None:
        return (image, old_sudoku)
    
    # getting the 4 corners of the biggest contour
    corners = get_contour_corners(biggest, 4)

    # if there is no sudoku in the image!
    if corners is None:
        return (image, old_sudoku)

    # since we have 4 corners; find the top left, top right, bottom left, bottom right corners
    square = np.zeros((4, 2), dtype="float32")
    corners = corners.reshape(4, 2)

    # top left (smallest sum of coordinates)
    sum = 10000
    index = 0
    for i in range(4):
        if (corners[i][0] + corners[i][1]) < sum:
            sum = corners[i][0] + corners[i][1]
            index = i

    square[0] = corners[index]
    corners = np.delete(corners, index, 0)

    # bottom right (largest sum of coordinates)
    sum = 0
    for i in range(3):
        if (corners[i][0] + corners[i][1]) > sum:
            sum = corners[i][0] + corners[i][1]
            index = i
    
    square[2] = corners[index]
    corners = np.delete(corners, index, 0)

    # top right and bottom left
    if corners[0][0] > corners[1][0]:
        square[1] = corners[0]
        square[3] = corners[1]
    else:
        square[1] = corners[1]
        square[3] = corners[0]
    
    square = square.reshape(4, 2)

    # we have found the 4 corners now, check if ABCD is approximately a square;
    A = square[0]
    B = square[1]
    C = square[2]
    D = square[3]

    # if all four angles are not approximately 90 degrees, stop!
    # vectors - AB, AD, BC, CD
    AB = B - A
    AD = D - A
    BC = B - C
    CD = C - D
    eps_angle = 20

    if not (is_approx_90_deg(get_angle(AB, AD), eps_angle) and is_approx_90_deg(get_angle(AB, BC), eps_angle) and is_approx_90_deg(get_angle(BC, CD), eps_angle) and is_approx_90_deg(get_angle(CD, AD), eps_angle)):
        return (image, old_sudoku)
    
    # the lenghts of AB, AD, BC, CD have to be approximately equal
    # i.e largest and shortest sides have to be approximately equal
    # largest cannot be longer than eps_scale * shortest
    eps_scale = 1.2
    if are_side_lengths_too_different(square, eps_scale):
        return (image, old_sudoku)
    
    # At this point we are sure that ABCD correspond to the 4 corners of the sudoku board

    # width of the sudoku board
    (tl, tr, br, bl) = square
    width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # height of the sudoku board
    height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # the maximum of the width and height values to reach the final dimensions
    max_width = max(int(width_A), int(width_B))
    max_height = max(int(height_A), int(height_B))

    # contructing destination points to get a top-down [birds eye] view
    destination = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype = "float32")
    
    # calculating perspective transform matrix and warp it to grab the screen
    # perspective transform function is used to implement the top-down transform
    perspective_transformed_matrix = cv.getPerspectiveTransform(square, destination)
    warp = cv.warpPerspective(image, perspective_transformed_matrix, (max_width, max_height))
    original_warp = np.copy(warp)

    # now, the warp only contains the chopped sudoku board
    # do some image processing for recognizing digits
    warp = cv.cvtColor(warp, cv.COLOR_BGR2GRAY)
    warp = cv.GaussianBlur(warp, (5, 5), 0)
    warp = cv.adaptiveThreshold(warp, 255, 1, 1, 11, 2)
    warp = ~warp
    _, warp = cv.threshold(warp, 150, 255, cv.THRESH_BINARY)

    # initializing grid to store the sudoku digits
    SIZE = 9

    grid = []
    for i in range(SIZE):
        row = []
        for j in range(SIZE):
            row.append(0)
        grid.append(row)
    
    height = warp.shape[0] // 9 
    width = warp.shape[1] // 9

    # offset used to get rid of the boundaries
    offset_width = math.floor(width / 10)
    offset_height = math.floor(height / 10)

    # dividing the sudoku board into 9x9 square
    for i in range(SIZE):
        for j in range(SIZE):
            # crop with offset to remove the boundaries
            image_crop = warp[height*i+offset_height:height*(i+1)-offset_height, width*j+offset_width:width*(j+1)-offset_width]

            # we are still left with the boundary lines
            # we remove all black lines near edges if 60% pixels are black
            # stop as soon as we reach a non black line
            ratio = 0.6

            # top
            while calculate_sum(image_crop[0]) <= (1 - ratio) * image_crop.shape[1] * 255:
                image_crop = image_crop[1:]

            # bottom
            while calculate_sum(image_crop[:, -1]) <= (1 - ratio) * image_crop.shape[1] * 255:
                image_crop = np.delete(image_crop, -1, 1)
            
            # left
            while calculate_sum(image_crop[:, 0]) <= (1 - ratio) * image_crop.shape[0] * 255:
                image_crop = np.delete(image_crop, 0, 1)
            
            # right
            while calculate_sum(image_crop[-1]) <= (1 - ratio) * image_crop.shape[0] * 255:
                image_crop = image_crop[:-1]
            
            # take the largestConnectedComponent (digit) and remove noises
            image_crop = ~image_crop
            image_crop = max_component(image_crop)

            # resize the crop image
            image_size = 28
            image_crop = cv.resize(image_crop, (image_size, image_size))

            ## if the crop image is a white cell, set grid[i][j] to 0 and continue to next crop image
            
            # if the crop image has very few black pixels
            if image_crop.sum() >= image_size**2 * 255 - image_size * 1 * 255:
                grid[i][j] = 0
                continue

            # if the crop image has a huge white area in the center
            center_width = image_crop.shape[1] // 2
            center_height = image_crop.shape[0] // 2
            x_start = center_height // 2
            x_end = center_height // 2 + center_height
            y_start = center_width // 2
            y_end = center_width // 2 + center_width
            center_region = image_crop[x_start:x_end, y_start:y_end]

            if center_region.sum() >= center_width * center_height * 255 - 255:
                grid[i][j] = 0
                continue

            # now, we are certain that the crop image contains a digit

            # applying binary threshold to make the digit more clear
            _, image_crop = cv.threshold(image_crop, 200, 255, cv.THRESH_BINARY)
            image_crop = image_crop.astype(np.uint8)

            # centralizing the image according to center of mass
            image_crop = ~image_crop
            shift_x, shift_y = best_shift(image_crop)
            shifted = shift(image_crop, shift_x, shift_y)
            image_crop = shifted

            image_crop = ~image_crop

            # converting to proper format for recognition (using our digitRecognition model)
            new_array = image_crop.reshape(-1, 28, 28, 1)
            new_array = new_array.astype('float32')
            new_array = new_array / 255
            image_crop = new_array

            # recognize digits using the digitRecognition model
            # model is trained by digitRecognition.py
            prediction = predict(model, image_crop)
            # debug("prediction", prediction)

            # predictions start from 0, so add 1 (1 -> 9)
            grid[i][j] = prediction + 1
            # debug("grid row and column", (i, j))

    user_grid = copy.deepcopy(grid)

    ## solving the sudoku after recognizing each digit of the board:

    # if same board is found as it was in the last camera frame;
    # no need to solve again; print the same solution.
    
    solve_sudoku(grid) # solve the sudoku

    # if its a new board
    if not old_sudoku:
        # debug("grid", grid)
        if is_board_filled(grid):
            updated_image = write_soln_on_image(original_warp, grid, user_grid)
            if updated_image is not None:
                original_warp = updated_image
                old_sudoku = grid[:]
    # if its the same board, do not change the frame
    elif old_sudoku and is_mat_equal(old_sudoku, grid, 9, 9):
        # debug("grid", grid)
        if is_board_filled(grid):
            updated_image = write_soln_on_image(original_warp, old_sudoku, user_grid)
            if updated_image is not None:
                original_warp = updated_image
    # later, if a new board is found, update the result
    else:
        # debug("grid", grid)
        if is_board_filled(grid):
            updated_image = write_soln_on_image(original_warp, grid, user_grid)
            if updated_image is not None:
                original_warp = updated_image
                old_sudoku = grid[:]

    # applying inverse perspective transform and pasting the solution on top of original image
    result_sudoku = cv.warpPerspective(original_warp, perspective_transformed_matrix, (image.shape[1], image.shape[0]), flags=cv.WARP_INVERSE_MAP)
    result = np.where(result_sudoku.sum(axis=-1,keepdims=True) != 0, result_sudoku, image)

    return (result, old_sudoku)
