import os
import re
import sys
import math
import time
from shutil import copy
from pathlib import Path, PureWindowsPath
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import skimage
import skimage.io
import skimage.filters as skf
import skimage.morphology as skm
from skimage.external.tifffile import TiffFile
from skimage.external.tifffile import TiffWriter
from scipy.ndimage.morphology import binary_fill_holes as fillHoles
from cellpose.models import Cellpose
import timeit
from tqdm.notebook import tqdm



#This function creates each single "square", used to crop the area around each aggregate.
#It takes the coordinates of the centroid of an object in each microwell and extends around it
#of a number of pixels equal to "radius" (this is half of the edge of the square)
#It then creates the array of coordinates of the square around
#the centroid:
def returnSquare(coords : tuple, radius : int, refImage : np.ndarray)-> np.ndarray:
    y = np.uint16(np.round(coords[0]))
    x = np.uint16(np.round(coords[1]))
    cols = np.array([pntX for pntX in range(x-radius, x+radius+1,1) if ((pntX >= 0) and (pntX <= refImage.shape[1]-1))])
    rows = np.array([pntY for pntY in range(y-radius, y+radius+1,1) if ((pntY >= 0) and (pntY <= refImage.shape[0]-1))])
    return np.array(np.meshgrid(rows, cols))


###############################################################################################################
####                                  MICROWELLS EXTRACTION FUNCTIONS                                      ####
###############################################################################################################



#THE MAIN FUNCTION FOR MICRWELL EXTRACTION (MICROWELL CROP):
#The arguments are:
#1) the organoid type (mISC, hSI, ...)
#1) the folder where the files to process are located (string format of the absolute path);
#2) the "radius" (in pixels) of the square used to crop the wells (edge of the square = 2*radius + 1);
#3) the minsize (= minimum area size in pixels) of the objects to consider as valid;
#4) the maxsize (= maximum area size in pixels) of the objects to consider as valid;
def returnCoord(folder : str = "", radius : int = 330, minsize : int = 20000, maxsize : int = 400000):
    #dictionary returned by the function. Its keys are the file names
    #and its values are the squares list for the corresponding file
    dic_squares = {}
    dic_centroids = {}
    dic_outfolder = {}
    folderPath = Path(folder) #Creates a "pathlib.PosixPath" object from the string format.
    #Regular expression (RE) to identify only ".tif" or “.tiff“ files in "folderPath":
    pattern = re.compile(r"^.*\.tif+$")
    #Create a list with all the ".tif/.tiff" files contained in "folderPath" in string format:
    fileList = [file.as_posix() for file in folderPath.iterdir() if file.is_file() and re.search(pattern, file.as_posix())]
    #Better to sort the files of the list in alphabetical order.
    fileList.sort()
    #print('the file list in folder ', folder, ' is \n', fileList)

    #Test code:
    #print(fileList

    #Here is the "for loop" iterating over all the "file paths" (= all the files) in the
    #"fileList" list:
    #Before I will define here everything that is common for every cycle of the "for loop"
    #RE used to isolate different only the path to the file without extension:
    pattern = re.compile(r".tif+$")
    #General "counter" used to give a progressive number name to each created file:
    imageCount = 0
    #"for loop" on "fileList":
    for file in tqdm(fileList):
        startImageTime = timeit.default_timer() #To time the processing of every original image/file.
        #Create the full path of the directory where the single "cropped area files" of
        #this specific "original file/well file" will be saved:
        patt2 = re.compile("\\\\") #""\\"" is the path-separator character used by the current OS.
        patt3 = re.compile(r".tif+$")
        moFullPath = re.split(patt2, file)
        outFolder = ""
        for part in moFullPath[1: -1]:
            outFolder = outFolder + part + "\\"
        finalPart = moFullPath[-1].replace(" ", "_")
        finalPart = re.split(patt3, finalPart)[0]
        outFolder = outFolder + finalPart
        #Test code:

        dic_outfolder[file] = outFolder
        #print(outFolder)

        #Create the actual folder. The argument "exist_ok" of this method states that if
        #the folder already exists a "FileExistsError" exception will be raised.
        #This exception is catched in the code below and it will return a sentence
        #and terminate the program execution:

        #Open the current single file (one file per "for loop" cycle):
        with TiffFile(file) as tif:
            imageInitial = tif.asarray() #return the image file as an "numpy.ndArray".

        #Filtering with gaussian filter:
        stackFiltered = skf.gaussian(imageInitial[:, :], sigma = 10,multichannel=False)

        #Find ideal threshold value
        #Threshold with "Li threshold method" (this takes a bit of time):
        threshold = skf.threshold_li(stackFiltered)

        #Create a mask (boolean array of values that are smaller than the threshold):
        stackThreshold = stackFiltered < threshold

        #Binary processing of mask -> Binary Dilation (using a disk of 7 pixels of diameter):
        circle = skm.disk(3)
        stackProcessed = skm.binary_dilation(stackThreshold, selem = circle)

        #Object/particle identification + labeling of objects/particles identified:
        stackLabels = skimage.measure.label(stackProcessed)

        #Create the collection of all the properties of all the identified objects/particles:
        objProperties = skimage.measure.regionprops(stackLabels)

        #FOR SOME AGGREGATES MORE THAN ONE PARTICLE/OBJECT COULD BE IDENTIFIED.
        #THIS MEANS THAT I WILL OBTAIN MORE THAN ONE CENTROID FOR SOME AGGREGATES...
        #THIS BRINGS THE RISK OF DUPLICATING SOME AGGREGATES OR TO CUT SOME OF THEM
        #"IN HALF".
        #TO SOLVE THIS ISSUE I HAVE TO GROUP TOGETHER CENTROIDS THAT EVENTUALLY BELONG TO
        #THE SAME PARTICLE/OBJCET.
        #THIS IS MY STRATEGY:
        #1) KEEP ONLY THE PARTICLES THAT HAVE AN AREA BETWEEN THE "minsize" AND "maxsize" ARGUMENTS
        #   OF THE "processFiles" FUNCTION.

        #2) CREATE A LIST OF ALL THE CENTROIDS (LIST OF TUPLES OF INTS)
        #I will filter by Area (particles between minArea and maxArea values, in pixel number), and also by Roundness.
        #Roundness is defined as (4 * Area) / (pi * Major_Axis^2).
        #By some tests I performed on test-shapes I observed that this parameter is mostly sensitive to
        #elongation, but still it is a bit sensitive to the roundenss of the shape when tested on objects that
        #have the same level of "elongation". Since I think we are mostly interested in excluding objects that are
        #too much elongated and do not care about perfect roundness, I decided to keep the "particles" with a
        #Roundness >= 0.4:
        originalCentroids = [obj.centroid for obj in objProperties if obj.area > minsize and obj.area < maxsize and
                             (4*obj.area)/(math.pi*(obj.major_axis_length)**2) >= 0.4]


        #3) CALCULATE THE DISTANCE BETWEEN EACH OF THE CENTROIDS (USING PITAGORA'S THEOREM) AGAINST ALL THE OTHERS
        #   (AVOID REPEATING COMPARISONS = EACH POINT WILL BE COMPARED ONLY WITH THE ONES THAT LIE DOWNSTREAM TO IT)
        #4)  IF POINTS THAT HAVE A DISTANCE LESS THAN "radius" FROM THE CURRENTLY ANALYZED POINT
        #    ARE FOUND, TWO ASSOCIATED TYPES OF LISTS MUST BE CREATED (GROUPED INSIDE A DICTIONARY):
        #     a) A LIST WITH THE COORDINATES OF THE ANALYZED POINT AND THE COORDINATES OF ALL THE POINTS THAT HAVE
        #       BEEN FOUND CLOSE TO IT.
        #     b) A LIST WITH THE INDEX OF EACH OF THESE POINTS IN THE ORIGINAL LIST OF CENTROIDS.
        listOfCentroidsDicts = list() #THIS IS THE LIST THAT YOU HAVE TO USE IN THE NEXT STEP!!
        for ind in range(0, len(originalCentroids)-1, 1):
            dictCentroids = {"Points" : [], "Indeces" : []}
            indexPnt = ind
            pntY = originalCentroids[ind][0]
            pntX = originalCentroids[ind][1]

            for otherInd in range(indexPnt+1, len(originalCentroids), 1):
                distance = math.sqrt((pntX - originalCentroids[otherInd][1])**2 + (pntY - originalCentroids[otherInd][0])**2)
                if distance < int(round(radius)):
                    dictCentroids["Points"].append(originalCentroids[otherInd])
                    dictCentroids["Indeces"].append(otherInd)
            #Only if centroids close to the currently analyzed centroid have been found
            #add also the coordinates and index of the currently analyzed centroid to the
            #lists in the "dictCentroids" dictionary:
            if len(dictCentroids["Indeces"]) > 0:
                dictCentroids["Points"].append(originalCentroids[indexPnt])
                dictCentroids["Indeces"].append(indexPnt)
            #Append the dictionary for the currently analyzed centroid to the list
            #"listOfCentroidsDicts":
            listOfCentroidsDicts.append(dictCentroids)

        #5) NOW A CHECK MUST BE PERFORMED. INDEED ONE POINT COULD BE CLOSE TO MORE THAN ONE POINT...
        #   THIS CREATES ANOTHER PROBLEM IN THE PROCEDURE, BECAUSE SOME POINTS WILL BE REPEATED IN THE DICTIONARIES' LISTS.
        #   THIS MUST BE AVOIDED.
        #   THE TRICK IS TO CHECK ALL THE OBTAINED "Indeces" LISTS INSIDE EACH DICTIONARY AND SEE IF THEY SHARE AN INDEX.
        #   IF THIS IS THE CASE THE LISTS (BOTH COORDINATES AND INDECES) MUST BE COMBINED. THE REPEATED CENTROIDS'
        #   COORDINATES AND INDECES MUST BE COMBINED. THEN BY TRANSFORMING THE LISTS IN SETS AND THEN
        #   BACK TO LISTS IT IS POSSIBLE TO GET RID OF THE DUPLICATED ELEMENTS:
        newlistOfCentroidsDicts = []
        indecesElemsToRemove = []
        #Execute this code only if there are centroid dicts in "listOfCentroidsDicts"...
        if len(listOfCentroidsDicts) > 0:
            for ind in range(0, len(listOfCentroidsDicts)-1, 1):
                tempDict = listOfCentroidsDicts[ind]
                for incr in range(1, len(listOfCentroidsDicts)-ind, 1):
                    for el in listOfCentroidsDicts[ind]["Indeces"]:
                        #This "if" statement checks if the element in "Indeces" of the
                        #currently analyzed dictionary is present in "Indeces" of one or more
                        #of the downstream dictionaries. It tests this for all the Indeces in
                        #the currently analyzed dictionary:
                        if el in listOfCentroidsDicts[ind+incr]["Indeces"]:
                            #If this condition is True, all the Indeces and all the corresponding coordinates
                            #(= Points) of the compared dictiobary are added to "Indeces" and "Points" of the
                            #currently analyzed dictionary = they are combined in a unique group!!
                            tempDict["Points"].extend(listOfCentroidsDicts[ind+incr]["Points"])
                            tempDict["Indeces"].extend(listOfCentroidsDicts[ind+incr]["Indeces"])
                            #When you find some elements in common between the currently
                            #analyzed dictionary and the compared dictionary, the indeces of both (as they are in
                            #"listOfCentroidsDicts") are recorded in "indecesElemsToRemove":
                            indecesElemsToRemove.append(ind+incr)
                            indecesElemsToRemove.append(ind)
                #If the newly created dictionary is different from the original currently analyzed
                #dictionary (= some points in common with other dictionaries have been found), the
                #proceed with the removal of duplicated "Points" and "Indeces":
                if tempDict != listOfCentroidsDicts[ind]:
                    tempDict["Points"] = set(tempDict["Points"])
                    tempDict["Points"] = list(tempDict["Points"])
                    tempDict["Indeces"] = set(tempDict["Indeces"])
                    tempDict["Indeces"] = list(tempDict["Indeces"])
                    #Finally append this new Dictionary to the "newlistOfCentroidsDicts", which will contain
                    #all the dictionaries modified/grouped:
                    newlistOfCentroidsDicts.append(tempDict)

        #Set to 0 all the elements to remove from listOfCentroidsDicts.
        #These are the dictionaries that have been combined together in "newlistOfCentroidsDicts".
        #These elements are the ones occupying the index positions collected in
        #"indecesElemsToRemove":
        if len(indecesElemsToRemove) > 0:
            for ind in indecesElemsToRemove:
                listOfCentroidsDicts[ind] = 0

        #Remove every element in the list that is equal to 0:
        while(True):
            try:
                listOfCentroidsDicts.remove(0)
            except ValueError: #When there are no ore "0" elements exit from the "while loop"
                break
        #Append the groups of points of the newly combined
        listOfCentroidsDicts.extend(newlistOfCentroidsDicts)
        #OK!! FINALLY "listOfCentroidsDicts" IS THE LIST WITH ALL THE GROUPS OF POINTS THAT
        #ARE CLOSE ONE TO THE OTHER!


        #6) THE FINAL STEP IS TO USE THE COORDINATES IN EACH COORDINATES LIST OF EACH DICTIONARY TO CALCULATE AN "AVEARGE" CENTROID = CALCULATE
        #   THE MEAN OF THE X AND Y COORDINATES FOR EACH POINT IN THE LIST. THIS WILL GIVE THE NEW CENTROID.
        #   APPEND THIS NEW CENTROID TO THE ORIGINAL CENTROIDS LIST.
        #   DELETE ALL THE ORIGINAL CENTROIDS USED TO CALCULATE THE "AVERAGE CENTROIDS" FROM THE ORIGINAL CENTROIDS LIST.
        #   TO DO THIS USE THE LIST WITH THE INDECES IN EACH DICTIONARY (= DELETE ELEMENTS OCCUPYING THOSE POSITIOINS IN THE ORIGINAL LIST).

        #I start with removing the centroids that have been included in groups of "close-enough centroids":
        for d in listOfCentroidsDicts:
            for ind in d["Indeces"]:
                originalCentroids[ind] = 0

        finalCentroids = set(originalCentroids)#convert to "set" to avoid duplicated elements.
        finalCentroids = list(finalCentroids)#convert back to "list".

        #This piece of code removes all the "0" elements and when it does not find anymore "0"
        #elements it returns a ValueError exits from the "while loop":
        while(True):
            try:
                finalCentroids.remove(0)
            except ValueError:
                break

        #Here I calculate the new "average" centroid for each "group of centroids" in "listOfCentroidsDicts"
        #and I append the new centroids to "finalCentroids"
        for d in listOfCentroidsDicts:
            sumX = 0
            sumY = 0
            pointsNumber = len(d["Points"])
            if pointsNumber == 0:
                continue
            for pnt in d["Points"]:
                sumX += pnt[1]
                sumY += pnt[0]
            newX = int(round(sumX/pointsNumber))
            newY = int(round(sumY/pointsNumber))
            newPnt = (newY, newX)
            finalCentroids.append(newPnt)

        #NOW YOU CAN FINALLY USE THIS MODIFIED VERSION OF THE LIST AS THE FIRST ARGUMENT OF THE
        #"returnSquare" FUNCTION!



        #Create a list with all the squares centered on the centroids of all the selected objects (= aggregates).
        #Make use of the function "returnSquare" defined at the beginning of this section.
        #The function takes radius, minsize and maxsize directly from the arguments of the "processFiles" function!
        squaresList = [returnSquare(centroid, radius, imageInitial) for centroid in finalCentroids]
        dic_squares[file] = squaresList
        dic_centroids[file] = finalCentroids


    return dic_squares, dic_centroids, dic_outfolder

# give folder with initial images in it
def crop_original_img(dic_squares, dic_centroids, dic_outfolder, folder=''):
    imageCount = 0
    folderPath = Path(folder) #Creates a "pathlib.PosixPath" object from the string format.
    #Regular expression (RE) to identify only ".tif" or “.tiff“ files in "folderPath":
    pattern = re.compile(r"^.*\.tif+$")
    #Create a list with all the ".tif/.tiff" files contained in "folderPath" in string format:
    fileList = [file.as_posix() for file in folderPath.iterdir() if file.is_file() and re.search(pattern, file.as_posix())]
    #Better to sort the files of the list in alphabetical order.
    print('the file list in folder ', folder, ' is \n', fileList)
    fileList.sort()

    for file in fileList:
        startImageTime = timeit.default_timer()
        with TiffFile(file) as tif:
            imageInitial = tif.asarray() #return the image file as an "numpy.ndArray".

        #This part of code returns an overlay of the original image with all the cropeed areas
        #on top of it, to check how the aggregate identification went!
        #Here I will draw the coordinates of all the Squares on an image of the same size of the original one.
        #Then I will put it as a second layer in transparency on top of the original image.
        croppedAreasImage = np.ones((imageInitial.shape[-2], imageInitial.shape[-1], 4))

        #Create a matplotilb.Figure and associated Axes:
        figCrop, axCrop = plt.subplots(1)
        #Create a random color arrangement for the squares and the text on top of them.
        randomValueSquare = []
        randomValueText = []

        keys = list(dic_squares.keys())
        # index of the file in the dictionary containing the squares for the file in question
        idx = [i for i, e in enumerate(keys) if (file.rpartition('/')[-1] in e)][0]


        squaresList = list(dic_squares.values())[idx]
        finalCentroids = list(dic_centroids.values())[idx]
        outFolder = list(dic_outfolder.values())[idx]

        for ind in range(0, len(squaresList), 1):
            #Attribute a random value from 0-255 to the pixels of each square. This will result in different colors,
            #taken from the colormap "nipy_spectral":
            randomValue = np.random.randint(0, 256, size = 1, dtype = np.uint8)[0]
            randomValueSquare.append(randomValue)
            #This should make possible to have a color for text on the squares that is different enough from the color
            #of the background square:
            randomValueText.append(np.uint8(255 - randomValue))

        #This is the colormap I will use. I need to instantiate it because I want to get the color values (rgba 0-1)
        #for the text (starting from the list of random values of "randomValueText")
        nipy_spectral_cm = matplotlib.cm.get_cmap("nipy_spectral")
        #Create the squares in the array and place the text objects on the image:
        for ind, square in enumerate(squaresList, 0):
            croppedAreasImage[square[0], square[1], :] = nipy_spectral_cm(randomValueSquare[ind])
            text = ""
            if int(ind+1) < 10:
                text = "0" + str(ind+1)
            else:
                text = str(ind+1)
            axCrop.text(x = finalCentroids[ind][1], y = finalCentroids[ind][0], s = text, fontsize = 3,
                        color = nipy_spectral_cm(randomValueText[ind]), horizontalalignment = "center",
                        verticalalignment = "center")

        #Draw the original image and on top the array with all the squares. The text objects have been already
        #inserted by the lines above:
        axCrop.imshow(imageInitial, cmap = "gray", vmin = 0, vmax = 255)
        axCrop.imshow(croppedAreasImage, alpha = 0.45)
        axCrop.set_axis_off()
        figCrop.savefig(outFolder + "\\" + "Cropped_Areas.pdf", dpi = 200, format = "pdf", bbox_inches = "tight")
        plt.show(figCrop);


        #Save the wells as separated images:
        counter = 0
        for square in squaresList:
            counter = counter + 1
            filename = ""
            if counter < 10:
                filename = "0" + str(counter) + ".tif"
            else:
                filename = str(counter) + ".tif"
            #print('outfolder', outFolder)
            #print('filename', filename)
            path = outFolder.rpartition('/')[0] + '/output/' + outFolder.rpartition('/')[2] + '\\' + filename
            #outFolder + "\\" + filename
            #Crop the single areas around aggregates and save them as separated files:
            image = imageInitial[square[0].min():square[0].max(), square[1].min():square[1].max()]
            with TiffWriter(path, bigtiff = False) as writer:
                writer.save(image)
                writer.close()
        finalImageTime = timeit.default_timer()
        imageCount = imageCount + 1
        print("Time for file ", imageCount, ": ", finalImageTime-startImageTime, " s\n")


###############################################################################################################
####                                  AGGREGATES SEGMENTATION FUNCTIONS                                    ####
###############################################################################################################
def segmentation(mainFolder : str = ""):
    #Here all the folders with the images to segment are collected in the "foldersList" and are then returned one by one
    #to the code below (inside the "for loop"):


    #The folder containing the files to segment/to use for mask creation (use absolute path in string format):
    inputFolder = mainFolder
    #The channel to use for mask creation. Use the BF channel.
    #It must be the same channel number (= in the same position) for all the files in the folder!!
    #In the case of SUN Bioscience the images must be only BF with one Z-Stack (one Channel, one Z-Stack)
    channelForSegm = 0

    #Timing the whole process for this folder:
    generalBeginning = timeit.default_timer()


    #Setting the deep-learning model of Cellpose to use for segmentation.
    #These are the suggested settings:
    #myModel = Cellpose(gpu = False, model_type = "cyto", net_avg = True)
    #To shorten the processing time the "net_avg" argument can be set to False, but the segmentation will be less precise...
    myModel = Cellpose(gpu = False, model_type = "cyto", net_avg = True)


    #Create the output folders:
    #Output folder for masks:
    maskOutput = inputFolder + "/Masks"
    os.mkdir(maskOutput)
    #Output folder for "Original Image + Mask Overlay"
    overlayOutput = inputFolder + "/Mask_Overlay"
    os.mkdir(overlayOutput)


    #Extraction of .tif and/or .tiff files paths from the inputFolder and creation of a fileList.
    #To convert these "file Path object" into a string you have to call the .as_posix() method on them.
    pattern = re.compile(r"^.*\.tif+$")
    folderPath = Path(inputFolder)
    fileListOne = []
    fileList = []
    fileListOne = [file for file in folderPath.iterdir() if file.is_file() and re.search(pattern, file.as_posix())]
    fileListOne.sort()

    for file in fileListOne:
        newPath = file.parts[0]
        for el in file.parts[1:]:
            newPath = newPath + "/" + el
        fileList.append(newPath)

    for el in fileList:
            print(el)


    #From here starts the "for loop" that will process each file in the inputFolder:
    for fileIndex, f in enumerate(fileList):
        #Timing for each single file:
        beginning = timeit.default_timer()

        #Here starts the hardcore part of the code!!!
        originalImage = None
        image = None
        with TiffFile(f) as tif:
            originalImage = tif.asarray()
        image = originalImage.copy()  #ARE ALL THESE COPIES NECESSARY??
        temp = image[:, :].copy()     #ARE ALL THESE COPIES NECESSARY?? THIS COULD BE DELETED IN CASE!!



        #Set the paths (in string format) of the output files (it will create sub-folders containing files with the names of the original files):
        pattern = re.compile("/")
        mo = re.split(pattern, f)
        fileName = mo[-1]
        print(inputFolder)
        print("\n", fileName)

        outputMask = inputFolder + "/Masks/" + fileName
        outputMaskOverlay = inputFolder + "/Mask_Overlay/" + fileName


        #This part will get the "measured radius" of the aggregate (in reality of the particle identified on BF).
        #The "measured radius" will be used to calculate a Segmentation Radius to use as argument
        #in the eval() method. One specific Segmentation Radius for each image!
        SegmentationRadius = None
        image0 = image[:, :]


        #Filtering with gaussian filter:
        image0 = skf.gaussian(image0, sigma = 10)
        #Find ideal threshold value:
        #Threshold with "Li threshold method" (this takes a bit of time):
        threshold = skf.threshold_li(image0)
        #Create a mask (boolean array of values that are smaller than the threshold):
        image0 = image0 < threshold
        #Binary processing of mask -> Binary Dilation (using a disk of 7 pixels of diameter)
        #(Maybe this Dilation step is not necessary, but if you remove this step I think you will have to
        #re-adjust the ratio between "measuredRadius" and "SegmentationRadius", bringing it closer to 1.0):
        image0 = skm.binary_dilation(image0, selem = skm.disk(3))
        #Object/particle identification + labeling of object/particles identified:
        particles = skimage.measure.label(image0)
        #Create the collection of all the properties of all the identified objects/particles:
        particlesProps = skimage.measure.regionprops(particles)

        area = None
        for el in particlesProps:
            if el.area > 20000:
                area = el.area

        #This code converts the size of the "measuredRadius" (calculated here starting from "area") to
        #the size of the "SegmentationRadius". This is made by multiplying "measuredRadius" 1,3 times.
        #1.3 is a value that I found empirically, checking which was the minimum best value for "
        #SegmentationRadius" of different particles of known "measuredRadius":
        if area != None:
            measuredRadius = math.sqrt(area/math.pi)
        else: #If a particle with area < 20000 pixels has not been identified, skip this file, go to the next one!
            continue
        SegmentationRadius = round((measuredRadius*1.3))

        #Here I run the Cellpose model "myModel" on the image using as "diameter" the value of "SegmentationRadius":
        segmentation = myModel.eval(image, channels = [0, 0], diameter = SegmentationRadius)

        #Extract the coordinates of the segmented object, what I call "mask", from the "segmentation" object:
        mask = segmentation[0]
        #Convert to boolean mask:
        mask = mask > 0
        #Convert the boolean "mask" to an np.uint8 "mask" (False = 0, True = 1):
        mask = mask.astype(np.uint8)


        #Unluckily Cellpose's segmentation identifies sometimes two or more "particles": one which contains the aggregate
        #located at the center of the image and sometimes some others located at the edges of the image.
        #I want to get rid of these eventual particles located at the edges of the images and keep only the
        #central particle.
        #I can use the centroid of each paticle to do that.
        #Let's say that if one of the coordinates of the centroid of a particle are +/- 150 pixels from the center
        #of the image, they are the one of the central particle and this must be kept. The other particles are not
        #considered, thus they are removed!
        imageXDim = image.shape[-1]
        imageYDim = image.shape[-2]
        imageCenterXY = {"x" : round(imageXDim/2), "y" : round(imageYDim/2)}
        #Transform "image" in a boolean mask where all the values different from 0 are set to True (= 1).
        imageThresholded = mask > 0
        particles = skimage.measure.label(imageThresholded)
        try:
            particlesProps = skimage.measure.regionprops(particles)
        except TypeError:
            print("A weird exception occured!\nI will skip the segmentation of this microwell!")
            continue
        #Creation of the "mask" with only the coordinates to keep.
        #Keep only the coordinates of the "central-correct" particle:
        coordsToKeep = None

        for el in particlesProps:
            y, x = el.centroid
            if (x < imageCenterXY["x"] + 150 and x > imageCenterXY["x"] - 150) or (y < imageCenterXY["y"] + 150 and y > imageCenterXY["y"] - 150):
                coordsToKeep = el.coords

        #Sometimes Cellpose dose not manage to identify the cell aggregate... and it just gives some random things
        #at the edges of the image... This makes it impossible to identify a "central particle"...
        #Thus, "coordsToKeep" remains None and when you iterate over it, it causes a blocking Exception.
        #For this reason it is better to catch this expcetion and say that if it appears the program should
        #skip (continue) the cycle for this file:

        mask = np.zeros(image.shape)
        try:
            for pnt in coordsToKeep:
                mask[pnt[0], pnt[1]] = 1
        except(TypeError):
            print(f"Cellpose failed for file: {f}. \n\n")
            continue

        #Mask the image with "mask" to keep only the correct particle in "image", the one containing the
        #aggregate:
        image = image * mask


        #From here I will apply to this "BF only image segmented by Cellpose" ("image") a series of
        #filters and size-based selection of particles to clean up further the segmented aggregate
        #from not-aggregated/shed cells deposited around it.

        #Apply an Hessian filter to "image". It is a filter that somehow show a sort of local variance.
        #It manages to somehow highlight a kind of border for the aggregates, more or less...
        filteredImage = skf.hessian(image, (2,2))
        #Invert the image (what is 0 will become 255 and viceversa):
        filteredImage = skimage.util.invert(filteredImage)
        #This step is required to cut out the border that the Hessian filter creates around the Cellpose segmented
        #area. This border is not good at all:
        filteredImage = filteredImage * mask
        #Boolean transformation of the "filteredImage" (every value which is greater than 0 it is set to True, meaning 1):
        filteredImage = filteredImage > 0
        #Erode "filteredImage" with a disk of 3 pixels of diameter:
        filteredImage = skm.binary_erosion(filteredImage, selem = skm.selem.disk(1))
        #Connect neighoboring particles using the "skm.closing" function (with default arguments):
        filteredImage = skm.closing(filteredImage)
        #Identify the particles created by the Hessian filter and all the other operations,
        #and create a collection of their properties:
        particles = skimage.measure.label(filteredImage)
        particlesProps = skimage.measure.regionprops(particles)
        #Keep only "big connected pieces" of the "filteredImage". In this way many of the marginal small pieces that
        #stay around the aggregate are removed:
        minArea = 300   #This defines the minimum area (in pixels' number) for a particle to be kept in the image.
        selectedCoords = []
        for el in particlesProps:
            if el.area > minArea:
                for pnt in el.coords:
                    selectedCoords.append(pnt)

        #Combination of all particles with area greater than "minArea" in the "finalMask":
        finalMask = np.zeros(image.shape)
        for pnt in selectedCoords:
            finalMask[pnt[0], pnt[1]] = 1

        #Closing using as selem a disk(25):"
        finalMask = skm.closing(finalMask, selem = skm.selem.disk(25))

        #Fill the holes (with a scipy function, check the "import section"):
        finalMask = fillHoles(finalMask) #THIS IS THE finalMask OBTAINED AFTER ALL THESE STEPS!!!

        #Mask with value of "0" or "1" for each pixel:
        segMaskImage = finalMask.copy()
        segMaskImage = segMaskImage.astype(np.uint8)


        #Here the saving of all the files in ".tif" format:
        #Save the Mask:
        with TiffWriter(outputMask, bigtiff = False, imagej = True) as writer:
            writer.save(segMaskImage)
            writer.close()

        #Save BF from "originalImage" with "finalMask" overlaid:
        fig, ax = plt.subplots(1)
        ax.imshow(originalImage, cmap = "gray")
        ax.imshow(finalMask, cmap = "inferno", alpha = 0.2)
        fig.savefig(outputMaskOverlay, format = "png", dpi = 200)
        plt.close(fig)

        #Timing for each single file:
        end = timeit.default_timer()
        print(f"Time for processing the file: {end-beginning} s . \n\n")


    #Timing of the whole process for the current folder:
    generalEnd = timeit.default_timer()
    print(f"Processing all the {len(fileList)} files took: {generalEnd-generalBeginning} s .")
