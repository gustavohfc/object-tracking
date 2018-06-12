import cv2
import sys
import math
import os
import numpy as np

# Possible methods: 'BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN'
METHOD = 'MIL'

def setValue(value, bound):
    if value >= bound:
        return bound - 1
    
    if value < 0:
        return 0
    return value


def setGt(gt, shape):
    return (setValue(gt[0], shape[1]),
            setValue(gt[1], shape[0]),
            setValue(gt[2], shape[1]),
            setValue(gt[3], shape[0]))


def read_ground_truth(ground_truth_file):
    ground_truth_list = []

    with open(ground_truth_file) as fp:  
        for line in fp:
            numbers = list(map(float, line.rstrip().split(',')))

            # (x0, y0, width, height )
            ground_truth_list.append((numbers[0], numbers[1], numbers[2] - numbers[0], numbers[3] - numbers[1]))

    return ground_truth_list


def read_frames(frames_folder):
    frames_list = []

    for filename in sorted(os.listdir(frames_folder)):
        if filename == ".DS_Store":
            continue

        frames_list.append(cv2.imread(frames_folder + filename, cv2.IMREAD_COLOR))

    return frames_list


def calculate_jaccard(obj, gt, shape):
    gt = setGt(gt, shape)

    imgGt = np.zeros(shape, dtype='uint8')
    imgGt[int(gt[1]) : int(gt[1]) + int(gt[3]), int(gt[0]) : int(gt[0]) + int(gt[2])] = 255

    imgCalc = np.zeros(shape, dtype='uint8')
    imgCalc[int(obj[1]) : int(obj[1]) + int(obj[3]), int(obj[0]) : int(obj[0]) + int(obj[2])] = 255
    
    # Bitwise operations between GT and object found.
    andImg = cv2.bitwise_and(imgGt, imgCalc)
    orImg = cv2.bitwise_or(imgGt, imgCalc)

    # Count of True occurances in both matrix.
    andCount = np.count_nonzero(andImg)
    orCount = np.count_nonzero(orImg)

    # Calculating j
    return andCount / orCount


def main(frames_folder, ground_truth_file):
    frameCount = 0
    fault = 0
    jaccard_list = []

    frames_list = read_frames(frames_folder)
    ground_truth_list = read_ground_truth(ground_truth_file)

    if len(ground_truth_list) != len(frames_list):
        raise Exception("O número de ground truths deve ser igual ao número de frames")

    restart_tracker = True
    tracker = cv2.Tracker_create(METHOD)

    for frame, gt in zip(frames_list, ground_truth_list):
        frameCount += 1

        if math.isnan(gt[0]):
            restart_tracker = True
            continue

        if restart_tracker:
            tracker = cv2.Tracker_create(METHOD)
            tracker.init(frame, ((int(gt[0])), int(gt[1]), int(gt[2]), int(gt[3])))
            restart_tracker = False

        # Update tracker
        _, obj = tracker.update(frame)

        jaccard_list.append(calculate_jaccard(obj, gt, (frame.shape[0], frame.shape[1])))

        if jaccard_list[-1] == 0:
            fault += 1
            restart_tracker = True

    jaccard_mean = np.mean(jaccard_list, axis=0)
    print("Jaccard = " + str(jaccard_mean))

    Rs = math.exp(-30*(fault / frameCount))
    print("Robustez = " + str(Rs))


if __name__ == '__main__' :
    print("Car1:")
    main("PD8-files/car1/", "PD8-files/gtcar1.txt")

    print("\n\nCar2:")
    main("PD8-files/car2/", "PD8-files/gtcar2.txt")
    