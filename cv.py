#!/usr/bin/python3

import cv2
import os
import numpy as np
import shutil
import random
import signal
import math
import sys
import matplotlib.pyplot as plt

threshold = 42.0

#datasetpath = "./datasets/dataset2/480x600/"
#gallerypath = "./datasets/dataset2/480x600probes/"
#impostorspath = "./datasets/dataset2/480x600impostors/"
datasetpath = './datasets/dataset1/180x200/'
gallerypath = './datasets/dataset1/180x200probes/'
impostorspath = './datasets/dataset1/180x200impostors/'

haarcascade = './haarcascades/haarcascade_frontalface_alt.xml'
datasetfaces = []
datasetlabels = []
log = False
interactive = False
scalefactor = 1.0
minneighbors = 3
#load OpenCV face detector, I am using LBP which is fast
face_cascade = cv2.CascadeClassifier(haarcascade)
facerecognizer = cv2.face.LBPHFaceRecognizer_create()
totdbimg = 0
totimpostors = 0
totgalleryimg = 4
totimpostorsimg = 4
dataset = []
gallery = []

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        print()

def totimages(path):
    totdbimg = 0
    for root, dirs, files in os.walk(path):
        totdbimg+=len(files)
    return totdbimg


'''function to detect face using OpenCV'''
def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
    #let's detect multiscale (some images may be closer to camera than others) images
    #result is a list of faces
    face = face_cascade.detectMultiScale(gray, scaleFactor=scalefactor, minNeighbors=minneighbors, minSize=(0,0), maxSize=(4096,4096));
    
    #if no faces are detected then return original img
    if not np.array(face).any():
        #return None, None
        return None
            
    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = face[0]
            
    #return only the face part of the image
    #return gray[y:y+w, x:x+h], faces[0]
    return gray[y:y+w, x:x+h]

def train(dataset, facerecognizer, loadingprefix='Training'):
    global datsetfaces
    global datasetlabels
    datasetfaces = []
    datasetlabels = []
    
    noface = []

    count = 0
    label = 0
    for root, dirs, files in os.walk(datasetpath):
        if files:
            for img in files:
                dataset.append(root+'/'+img)
                face = cv2.imread(root+'/'+img)
                #print(str(count)+')', root+'/'+img)
                #print(str(count)+')', '['+str(label)+"] "+img+" SIZE:",len(face[0]), len(face), root+'/'+img)
                face = detect_face(face)
                if face is not None:
                    datasetfaces.append(face)
                    datasetlabels.append(label)
                else:
                    noface.append(root+'/'+img)
                printProgressBar(count+1 , totdbimg-totgalleryimg, prefix = loadingprefix, suffix = 'Complete', length = 70)
                count+=1
                label+=1
    facerecognizer.train(datasetfaces, np.array(datasetlabels))
    if noface and log:
        print(bcolors.WARNING+'The following images may have no face:')
        for face in noface:
            print(bcolors.WARNING+'\t'+face+bcolors.ENDC)
    return len(noface)

#do i need to save/load the trained data?
#facerecognizer.save("tained.yml")

#prediction
def topMatch(probe, facerecognizer):
    testimg = cv2.imread(probe).copy()
    face = detect_face(testimg)
    if face is None:
        if log:
            print(bcolors.FAIL+'[topMatch()] '+probe+' has no face!'+bcolors.ENDC)
        return (None, None)
    label, confidence = facerecognizer.predict(face)
    return (label, confidence)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def resetgallery():
    print(bcolors.WARNING+'Resetting probes images...'+bcolors.ENDC)
    g = os.listdir(gallerypath)
    if not g:
        print(bcolors.OKBLUE+'Probes are clean.'+bcolors.ENDC)
    for root, dirs, files in os.walk(gallerypath):
        for fname in files:
            dst = ''
            src = ''
            if fname.startswith('IMP_'):
                dst = impostorspath+fname.split('.')[0]+'/'+fname
                src = root+fname
            else:
                dst = datasetpath+fname.split('.')[0]+'/'+fname
                src = root+fname
            if log:
                print(bcolors.WARNING+'\t'+src+' ==> '+dst+bcolors.ENDC)
            shutil.move(src, dst)

def creategallery(path):
    if path == datasetpath:
        print(bcolors.OKBLUE+'Creating genuine probes...'+bcolors.ENDC)
    else:
        print(bcolors.OKBLUE+'Creating impostors'+bcolors.ENDC)
    dirs = os.listdir(path)
    num = set()
    i = 0
    totimg = 0
    if path == datasetpath:
        totimg = totgalleryimg
    if path == impostorspath:
        totimg = totimpostorsimg
    while i < totimg:
        drnd = random.randint(0, len(dirs)-1)
        if drnd not in num:
            num.add(drnd)
            files = os.listdir(path+dirs[drnd])
            frnd = random.randint(0, len(files)-1)
            src = path+dirs[drnd]+'/'+files[frnd]
            dst = gallerypath+files[frnd]
            if log:
                print(bcolors.OKBLUE+'\t'+src+' ==> '+dst+bcolors.ENDC)
            shutil.move(src, dst)
            i+=1


def check(show, groundtruth):
    global gallery
    match = []
    nonmatch = []
    ret = []
    gallery = [f for f in os.listdir(gallerypath) if os.path.isfile(os.path.join(gallerypath, f))]
    for probe in gallery:
        label,confidence = topMatch(gallerypath+probe, facerecognizer)
        ret.append((probe, confidence))
        if label is not None:
            if groundtruth:
                name = probe.split('.')[0].lower() 
                if name in dataset[label].lower():
                    match.append(probe+' '+dataset[label]+' '+str(label)+' '+str(confidence))
                else:
                    nonmatch.append(probe+' '+dataset[label]+' '+str(label)+' '+str(confidence))
            else:
                if confidence <= threshold:
                    match.append(probe+' '+dataset[label]+' '+str(label)+' '+str(confidence))
                else:
                    nonmatch.append(probe+' '+dataset[label]+' '+str(label)+' '+str(confidence))
                
    if log or show:
        print('Ground truth: '+str(groundtruth))
        if match:
            print(bcolors.OKGREEN+'Matched images:'+bcolors.ENDC)
            for img in match:
                print(bcolors.OKGREEN+'\t'+img+bcolors.ENDC)
        if nonmatch:
            print(bcolors.FAIL+'Non matched images:'+bcolors.ENDC)
            for img in nonmatch:
                print(bcolors.FAIL+'\t'+img+bcolors.ENDC)
    return (len(match), len(nonmatch), ret)

def printconfig():
    print(bcolors.OKBLUE+'Acual configuratoins:')
    print('\tThresold: '+str(threshold))
    print('\tDataset path: '+datasetpath)
    print('\tGallery path: '+gallerypath)
    print('\tImpostors Path: '+impostorspath)
    print('\tHaarcascade: '+haarcascade)
    print('\tTotal database images: '+str(totdbimg))
    print('\tTotal impostors images: '+str(totimpostors))
    print('\tTotal impostors probes: '+str(totimpostorsimg))
    print('\tTotal genuine probes: '+str(totgalleryimg))
    print('\tScale factor: '+str(format(scalefactor, '.1f')))
    print('\tMinimum neighbors: '+str(minneighbors)+' (set manually)')
    print('\tTotal impostors: '+str(totimpostorsimg))
    if log:
        print('\tLog: Enabled')
    else:
        print('\tLog: Disabled')
    print('\tInteractive mode: '+str(interactive))
    print(bcolors.ENDC)

def plotcms(prob, ratio, k):
    xaxxis = []
    for i in range(k):
        xaxxis.append(i+1)
    plt.plot(xaxxis, prob, label='prob')
    plt.plot(xaxxis, ratio, label='ratio')
    plt.legend()
    plt.show()


def cms(facerecognizer):
    print(bcolors.OKBLUE+'Calculating CMS (Cumulative Match Score), may take several minutes.'+bcolors.ENDC)
    cms = []
    cmsnoface = []
    plotvalprob = []
    plotvalratio = []
    k = 4 
    matrix = []
    c = 0
    #matric probes x gallery
    for root, dirs, files in os.walk(gallerypath):
        for f in files: #for each probes
            confidences = []
            probe = ''
            for ro, di, fi in os.walk(datasetpath):
                minconf = 140
                probe = ''
                for ff in fi: #for each img in dataset
                    probe = ff
                    face = cv2.imread(ro+'/'+ff)
                    face = detect_face(face)
                    printProgressBar(c+1 , (totdbimg-totgalleryimg)*totgalleryimg, prefix = 'CMS:', suffix = 'Complete', length = 70)
                    c+=1
                    if face is not None:
                        #print('>>>', root, f)
                        facerecognizer.train([face, face], np.array([1, 2]))
                        label, confidence = topMatch(root+'/'+f,facerecognizer)
                        if confidence < minconf:
                            minconf = confidence
                    else:
                        cmsnoface.append(ro+'/'+ff)
                confidences.append((probe, minconf))
            confidences.sort(key=lambda tup:tup[1])
            matrix.append((f, confidences))
    for i in range(k):
        prob = 0
        for j in matrix:
            if j[0].split('.')[0] in j[1][i][0]:
                prob+=1
        plotvalprob.append(prob/totgalleryimg)
        if log:
            print('prob(k='+str(i)+') =',prob/totgalleryimg)

    for x in range(k):
        ratio = 0
        for i in range(x+1):
            for j in matrix:
                if j[0].split('.')[0] in j[1][i][0]:
                    ratio+=1
        plotvalratio.append(ratio/totgalleryimg)
        if log:
             print('Ratio 1 -', x+1, ':',ratio/totgalleryimg)
    plotcms(plotvalprob, plotvalratio, k)
    if log and cmsnoface:
        print(bcolors.FAIL+'Not recognized faces:'+bcolors.ENDC)
        for i in cmsnoface:
            print(bcolors.FAIL+'\t'+i+bcolors.ENDC) 
                    
def createopensetgallery():
    global totimpostors
    totimpostors =  len([f for f in os.listdir(impostorspath) if os.path.isdir(os.path.join(impostorspath, f))])
    if totimpostorsimg > totimpostors:
        print(bcolors.FAIL+'Too many impostors set, Max possible value: '+str(totimpostors)+' set value: '+totimpostorsimg+bcolors.ENDC)
        exit(0)
    print(totimpostorsimg, totimpostors)
    creategalery(datasetpath)
    creategallery(impostorspath)
    
def detectdatasetimgsize():
    print(bcolors.OKBLUE+'Checking dataset images size...'+bcolors.ENDC)
    imgsizes = dict()
    for root, dirs, files in os.walk(datasetpath):
        for f in files:
            x, y, z = cv2.imread(root+'/'+f).shape         
            key = str(x)+'x'+str(y)
            if key in imgsizes:
                imgsizes[key] = imgsizes[key] + 1
            else:
                imgsizes[key] = 1
    if len(imgsizes.keys()) > 1:
        print(bcolors.WARNING+'\tDifferent types of image sizes has been found! it\'s suggested to have images of same size.'+bcolors.ENDC)
        key = ''
        maxx = 0
        for k, v in imgsizes.items():
            if log:
                print(bcolors.OKBLUE+'\t'+k+': '+str(v)+' images'+bcolors.ENDC)
            if v > maxx:
                maxx = v
                key = k
        print(bcolors.WARNING+'\tSystem might be optimized for images of size '+key+bcolors.ENDC)
    else:
        print(bcolors.OKBLUE+'Dataset images are same size.'+bcolors.ENDC)


def cmdlinearguments(cmd):
    global log
    global interactive
    if cmd == '-l':
        log = True
    if cmd == '-i':
        interactive = True
           

def plotgraph(vfrr, vfar):
    xassis = []
    yfrr = []
    xfrr = []
    yfar = []
    xfar = [] 
    for i in vfar:
        xfar.append(float(i[1]))
        yfar.append(float(i[0]))
    for i in vfrr:
        xfrr.append(float(i[1]))
        yfrr.append(float(i[0]))
    
    plt.plot(xfrr, yfrr, label = 'frr')
    plt.plot(xfar, yfar, label = 'far')
    plt.legend()
    plt.show()

def frrfar():
    global threshold
    print('Searching the best threshold...')
    if totgalleryimg != totimpostorsimg:
        print(bcolors.FAIL+'Genuine and impostors have to be the same number!'+bcolors.ENDC)
        exit(0)
    maxt = 140
    mint = 0
    maxtry = 10
    frr = 0
    far = 0
    fase = 1
    vfrr = set()
    vfar = set()
    #m, nm, val = check(True)
    while frr != far or  maxtry > 0:
        resetgallery()
        creategallery(datasetpath)
        creategallery(impostorspath)
        train(dataset, facerecognizer, 'Fase '+str(fase)+'/10')
        m, nm, val = check(False, True)
        threshold = math.ceil((maxt+mint)/2)
        for i in val:
            if i[1] is None:
                continue # sometimes a confidence is None have no idea why
            if i[1] > threshold and not i[0].startswith('IMP'):
                frr+=1
            if i[0].startswith('IMP') and i[1] < threshold:
                far+=1
        frr/=totgalleryimg
        far/=totimpostorsimg
        if frr > far:
            mint = threshold
        elif far > frr:
            maxt = threshold
        fase+=1
        maxtry-=1
        vfrr.add((frr, threshold))
        vfar.add((far, threshold))
        if log:
            print('Threshold:', threshold, 'FRR:', frr, 'FAR:', far, 'Equal?:', frr==far)
        frr = 0
        far = 0
    print(bcolors.OKBLUE+'Threshold set to: '+str(threshold)+bcolors.ENDC)
    
    vfrr = list(vfrr)
    vfar = list(vfar)
    vfrr.sort(key=lambda tup:tup[1])
    vfar.sort(key=lambda tup:tup[1])
    plotgraph(vfar, vfrr)
    return frr == far


if  __name__ == '__main__': 
    
    for cmd in sys.argv:
        cmdlinearguments(cmd)
    print('===== Face recognition V2 =====')
    print(bcolors.HEADER+'DESCLAIMER: Photos of people for the test are taken from a free and open database:\nhttps://cswww.essex.ac.uk/mv/allfaces/index.html'+bcolors.ENDC)
    print('Initialinzing...')
    
    if interactive:
        input('Check dataset image size >')
    detectdatasetimgsize()
    for _, dirs, _ in  os.walk(datasetpath):
        if totgalleryimg > len(dirs):
            print(bcolors.FAIL+'Trying to get too many images of different people, Max value possible: '+str(len(dirs))+', value set: '+str(totgalleryimg)+bcolors.ENDC)     
            exit(0)
        break
    resetgallery()
    totdbimg = totimages(datasetpath)
    totimpostors = totimages(impostorspath)
    errfc = totdbimg
    mtch = 0
    nnmtch = totgalleryimg
    #----------------------------------------------------------------
    # add more code here if necessary
    # ...
    #----------------------------------------------------------------
    if interactive:
        input('Calculate scale factor >')
    creategallery(datasetpath)
    for i in range(10):
        scalefactor += float(format(0.1, '.1f'))
        errorfaces = train(dataset, facerecognizer)
        match, nonmatch, val = check(False, True)

        if match > mtch:
            mtch = match
        if errorfaces < errfc:
            errfc = errorfaces
        if nonmatch < nnmtch:
            nnmtch = nonmatch
        if match < mtch or errorfaces > errfc:
            scalefactor-= float(format(0.1, '.1f'))
            break
    print('Best configuration for Scale Factor: '+str(format(scalefactor, '.1f')))
    print('Not recognizer dataset image: '+str(errfc)+'/'+str(totdbimg - totgalleryimg))
    print('Match: '+str(mtch)+'/'+str(totgalleryimg))
    print('Error: '+str(nnmtch)+'/'+str(totgalleryimg))
    
    printconfig()
    if interactive:
        input('Calculate CMS >')
    cms(facerecognizer)

    if interactive:
        input('Calculate threshold >')
    frrfar()
    opt = -1
    while True:
        count = 0
        print('1 - Create Closed Set Gallery')
        print('2 - Create Open Set Gallery')
        print('3 - Train')
        print('4 - Predict')
        print('5 - Reset Probes')
        print('6 - Complete Test (Closed Test) (5+1+3+4+5)')
        print('7 - Complete Test (Open Set) (5+2+3+4+5) ')
        if log:
            print('8 - Enable/Disable Logs ('+bcolors.OKBLUE+str(log)+bcolors.ENDC+')')
        else:
            print('8 - Enable/Disable Logs ('+bcolors.HEADER+str(log)+bcolors.ENDC+')')
        print('9 - Print configurations')
        print('0 - Exit')
        while True:
            try:
                opt = int(input('Select option: '))
                break
            except:
                pass
        if opt == 1:
            resetgallery()
            creategallery(datasetpath)
        elif opt ==2:
            resetgallery()
            creategallery(datasetpath)
            creategallery(impostorspath)
        elif opt == 3:
            train(dataset, facerecognizer)
        elif opt == 4:
            check() 
        elif opt == 5:
            resetgallery()
        elif opt == 6:
            resetgallery()
            creategallery(datasetpath)
            train(dataset, facerecognizer)
            check(True, False)
            resetgallery()
        elif opt == 7:
            resetgallery()
            creategallery(datasetpath)
            creategallery(impostorspath)
            train(dataset, facerecognizer)
            check(True, False)
            resetgallery()
        elif opt == 8:
            if log:
                log = False
            else:
                log = True
        elif opt == 9:
            printconfig()
        elif opt == 0:
            exit(0)
        else:
            pass
        
