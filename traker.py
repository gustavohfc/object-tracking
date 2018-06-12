import cv2
import sys
import math
import os

METHOD = 'KCF'


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


def create_tracker(frame, gt):
    tracker = cv2.Tracker_create(METHOD)
    tracker.init(frame, gt)

    return tracker


def main(frames_folder, ground_truth_file):

    ground_truth_list = read_ground_truth(ground_truth_file)
    frames_list = read_frames(frames_folder)

    if len(ground_truth_list) != len(frames_list):
        raise Exception("O número de ground truths deve ser igual ao número de frames")

    restart_tracker = True
 
    for frame, gt in zip(frames_list, ground_truth_list):

        if math.isnan(gt[0]):
            restart_tracker = True
            continue

        if restart_tracker:
            tracker = create_tracker(frame, gt)
            restart_tracker = False
 
        # Update tracker
        _, obj = tracker.update(frame)

        print(obj)
 



             
if __name__ == '__main__' :
    main("PD8-files/car1/", "PD8-files/gtcar1.txt")