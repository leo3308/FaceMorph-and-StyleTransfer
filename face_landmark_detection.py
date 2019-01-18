#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   The face detector we use is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset (see
#   https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):  
#      C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
#      300 faces In-the-wild challenge: Database and results. 
#      Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
#   You can get the trained model file from:
#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.
#   Note that the license for the iBUG 300-W dataset excludes commercial use.
#   So you should contact Imperial College London to find out if it's OK for
#   you to use this model file in a commercial product.
#
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy

import sys
import os
import dlib
import glob
import numpy as np
from PIL import Image

#if len(sys.argv) != 3:
#    print(
#        "Give the path to the trained shape predictor model as the first "
#        "argument and then the directory containing the facial images.\n"
#        "For example, if you are in the python_examples folder then "
#        "execute this program by running:\n"
#        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
#        "You can download a trained facial shape predictor from:\n"
#        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
#    exit()
def face_landmark_detection(filename):

    predictor_path = "shape_predictor_68_face_landmarks.dat"
    faces_folder_path = filename

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    file = open(faces_folder_path+'.txt','w+')

    #win = dlib.image_window()
    array=[]
    border=[]
    for f in glob.glob(os.path.join(faces_folder_path)):
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)

        #win.clear_overlay()
        #win.set_image(img)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            """
            border.append(str(d.left()))
            border.append(str(d.top()))
            border.append(str(d.right()))
            border.append(str(d.bottom()))
            
            file.write(border[0])
            file.write(" ")
            file.write(border[1])
            file.write(" ")

            file.write(border[0])
            file.write(" ")
            file.write(border[3])
            file.write(" ")

            file.write(border[2])
            file.write(" ")
            file.write(border[1])
            file.write(" ")

            file.write(border[2])
            file.write(" ")
            file.write(border[3])
            file.write(" ")
            """
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            #print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
            #                                          shape.part(1)))
            # Draw the face landmarks on the screen.
            #win.add_overlay(shape)
        img = Image.open(filename)
        img = np.array(img)
        width = img.shape[1]-1
        heigth = img.shape[0]-1
        half_width = img.shape[1] // 2
        half_height = img.shape[0] // 2
        border_points = [(0,0),(width,0),(0,heigth),(width,heigth),(0,half_height),(width,half_height),(half_width,0),(half_width,heigth)]

        for i in range(68):
            array.append(shape.part(i))
            ca = str(array[i])
            c=(ca.split(","))
            if i!=0:
                file.write("\n")
            for j in c[0]:
                if j!='(':
                        file.write(j)
            for j in c[1]:
                if j!=')':
                    file.write(j)
        file.write("\n")
        for i in range(len(border_points)):
            file.write(str(border_points[i][0]) + " " + str(border_points[i][1]) + "\n")
        file.close()

    #win.add_overlay(dets)
    
