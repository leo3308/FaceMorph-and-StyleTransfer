#!/usr/bin/env python

import numpy as np
import cv2
import sys
from PIL import Image,ImageDraw
from matplotlib import pyplot as plt
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import dlib
from face_landmark_detection import face_landmark_detection


def delaunay(filename):
    
    # Define window names
    win_delaunay = "Delaunay Triangulation"
    win_voronoi = "Voronoi Diagram"
    
    # Turn on animation while drawing triangles
    animate = False
    
    # Define colors for drawing.
    delaunay_color = (255,255,255)
    points_color = (0, 0, 255)
    
    # Read in the image.
    #img = cv2.imread("obama.jpg");
    #img = cv2.imread(sys.argv[1])
    img = cv2.imread(filename)
    
    # Keep a copy around
    img_orig = img.copy();
    
    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])
    
    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect);
    
    # Create an array of points.
    points = [];
    
    # Read in the points from a text file
    with open(filename+".txt") as file :
        for line in file :
            #x, y = line.split()
            c = line.split()
            #for i in range():
            #    points.append((int(x), int(y)))
            for i in range(len(c)):
                if i%2==0:
                    points.append((int(c[i]),int(c[i+1])))

    # Insert points into subdiv
    for p in points :
        subdiv.insert(p)
    
    # Show animation
    #        if animate :
    #            img_copy = img_orig.copy()
    #            # Draw delaunay triangles
    #            draw_delaunay( img_copy, subdiv, (255, 255, 255) );
    #            #cv2.imshow(win_delaunay, img_copy)
    #            cv2.waitKey(100)
    
    # Draw delaunay triangles
    #draw_delaunay( img, subdiv, (255, 255, 255) );
    
    # Draw points
    #    for p in points :
    #        draw_point(img, p, (0,0,255))
    
    # Allocate space for voronoi Diagram
    #img_voronoi = np.zeros(img.shape, dtype = img.dtype)
    
    # Draw voronoi diagram
    #draw_voronoi(img_voronoi,subdiv)
    
    tri_points = []
    triangleList = subdiv.getTriangleList()
    for t in triangleList:
        #print (t[0],t[1],t[2],t[3],t[4],t[5])
        for i in range(len(points)):
            if (points[i][0]==int(t[0]) and points[i][1]==int(t[1])):
                tri_points.append(i)
            elif (points[i][0]==int(t[1]) and points[i][1]==int(t[0])):
                tri_points.append(i)
            elif (points[i][0]==int(t[2]) and points[i][1]==int(t[3])):
                tri_points.append(i)
            elif (points[i][0]==int(t[3]) and points[i][1]==int(t[2])):
                tri_points.append(i)
            elif (points[i][0]==int(t[4]) and points[i][1]==int(t[5])):
                tri_points.append(i)
            elif (points[i][0]==int(t[5]) and points[i][1]==int(t[4])):
                tri_points.append(i)

    outfile = open("triangle.txt",'w')
    for i in range(len(tri_points)):
        if i!=0 and i%3 == 0:
            outfile.write("\n")
        outfile.write(str(tri_points[i])+" ")
    outfile.close()

# Read points from text file
def readPoints(path) :
    # Create an array of points.
    points = [];
    # Read points
    with open(path) as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))

    return points

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))


    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []


    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask


def MorphImage(alpha, filename1, filename2):

#    print(filename1, filename2)
#    filename1 = 'ted_cruz.jpg'
#    filename2 = 'donald_trump.jpg'
    #alpha = 0.5
    #face_landmark_detection(filename1)
    #face_landmark_detection(filename2)
    delaunay(filename1)
    
    # Read images
    img1 = cv2.imread(filename1);
    img2 = cv2.imread(filename2);

    
    # Convert Mat to float data type
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    # Read array of corresponding points
    points1 = readPoints(filename1 + '.txt')
    points2 = readPoints(filename2 + '.txt')
    points = [];

    # Compute weighted average point coordinates
    for i in range(0, len(points1)):
        x = ( 1 - alpha ) * points1[i][0] + alpha * points2[i][0]
        y = ( 1 - alpha ) * points1[i][1] + alpha * points2[i][1]
        points.append((x,y))


    # Allocate space for final output
    imgMorph = np.zeros(img1.shape)

    # Read triangles from tri.txt
    with open("triangle.txt") as file :
        for line in file :
            x,y,z = line.split()
            
            x = int(x)
            y = int(y)
            z = int(z)
            
            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [ points[x], points[y], points[z] ]

            # Morph one triangle at a time.
            morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)

    return imgMorph
    # Display Result
    # imgMorph2 = imgMorph[:,:,[2,1,0]]
#    imgMorph = np.uint8(imgMorph)
#    imgMorph2 = cv2.cvtColor(imgMorph,cv2.COLOR_BGR2RGB)
#    height, width, channels= imgMorph2.shape
#    bytesPerLine = channels * width
#    app = QApplication(sys.argv)
#    qImg = QImage(imgMorph2.data, width, height, bytesPerLine, QImage.Format_RGB888)
#    pixmap01 = QPixmap.fromImage(qImg)
#    pixmap_image = QPixmap(pixmap01)
#    label_imageDisplay = QLabel()
#    label_imageDisplay.setPixmap(pixmap_image)
#    widget = QWidget()
#    layout = QVBoxLayout()
#    layout.addWidget(label_imageDisplay)
#    widget.setLayout(layout)
#    widget.show()
#    app.exec_()
#    plt.imshow(np.uint8(imgMorph2))
#    plt.show()
    #cv2.imshow("Morphed Face", np.uint8(imgMorph))
    #cv2.waitKey(0)
#images = []
#for i in range(10):
#    alpha = i/10 + 0.1
#    img_array = morphing(alpha)
#    img_array = img_array[:,:,[2,1,0]]
#    img = Image.fromarray(img_array.astype("uint8"), "RGB")
#    images.append(img)
#images[0].save('test.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
#img_array = morphing(0.5)
#img = Image.fromarray(img_array.astype("uint8"), "RGB")
#img.show()









