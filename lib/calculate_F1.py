"""

    Calculation of F1 value based on a KD-Tree

    Marc Aubreville, Pattern Recognition Lab, FAU Erlangen-Nürnberg
    
    To reduce complexity of calculation, the F1 score is here derived by querying a KD Tree.
    
    This is possible, since all objects are round and have the same diameter. 

"""


import bz2
import pickle
import numpy as np
from sklearn.neighbors import KDTree

from SlideRunner.dataAccess.database import Database
from lib.nms_WSI import nms

def _F1_core(centers_DB : np.ndarray, boxes : np.ndarray, score : np.ndarray, det_thres:float):

        to_keep = score>det_thres
        boxes = boxes[to_keep]


        if boxes.shape[-1]==6:
            # 4-coordinates -> x1,y1,x2,y2 --> calculate centers
            center_x = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
            center_y = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2
        else:
            center_x = boxes[:, 0] 
            center_y = boxes[:, 1] 

        isDet = np.zeros(boxes.shape[0]+centers_DB.shape[0])
        isDet[0:boxes.shape[0]]=1 # mark as detection, rest ist GT

        if (centers_DB.shape[0]>0):
            center_x = np.hstack((center_x, centers_DB[:,0]))
            center_y = np.hstack((center_y, centers_DB[:,1]))
        radius=25
        

        # set up kdtree 
        X = np.dstack((center_x, center_y))[0]

        if (X.shape[0]==0):
            return 0,0,0,0

        
        try:
            tree = KDTree(X)
        except:
            print('Shapes of X: ',X.shape)
            raise Error()

        ind = tree.query_radius(X, r=radius)

        annotationWasDetected = {x: 0 for x in np.where(isDet==0)[0]}
        DetectionMatchesAnnotation = {x: 0 for x in np.where(isDet==1)[0]}

        # check: already used results
        alreadyused=[]
        for i in ind:
            if len(i)==0:
                continue
            if np.any(isDet[i]) and np.any(isDet[i]==0):
                # at least 1 detection and 1 non-detection --> count all as hits
                for j in range(len(i)):
                    if not isDet[i][j]: # is annotation, that was detected
                        if i[j] not in annotationWasDetected:
                            print('Missing key ',j, 'in annotationWasDetected')
                            raise ValueError('Ijks')
                        annotationWasDetected[i[j]] = 1
                    else:
                        if i[j] not in DetectionMatchesAnnotation:
                            print('Missing key ',j, 'in DetectionMatchesAnnotation')
                            raise ValueError('Ijks')

                        DetectionMatchesAnnotation[i[j]] = 1

        TP = np.sum([annotationWasDetected[x]==1 for x in annotationWasDetected.keys()])
        FN = np.sum([annotationWasDetected[x]==0 for x in annotationWasDetected.keys()])

        FP = np.sum([DetectionMatchesAnnotation[x]==0 for x in DetectionMatchesAnnotation.keys()])
        F1 = 2*TP/(2*TP + FP + FN)

        return F1, TP, FP, FN



def calculate_F1(databasefile, result_boxes=None, resfile=None, det_thres=0.5, hotclass=2):

    
    DB = Database()
    DB = DB.open(databasefile)

    if (result_boxes is None):
        if resfile is None:
            raise ValueError('At least one of resfile/result_boxes must be given')
    
    if (resfile[-3:] == 'bz2'):
        f = bz2.BZ2File(resfile, 'rb')
    else:
        f = open(resfile,'rb')

    result_boxes = pickle.load(f)

    sTP, sFN, sFP = 0,0,0
    F1dict = dict()
    
    result_boxes = nms(result_boxes, det_thres)

    print('Calculating F1 for test set of %d files' % len(result_boxes))
    
    for resfile in result_boxes:
        boxes = np.array(result_boxes[resfile])
        

        TP, FP, FN = 0,0,0
        if boxes.shape[0]>0:
            score = boxes[:,-1]

            DB.loadIntoMemory(DB.findSlideWithFilename(resfile,''))

            # perform NMS on detections

            annoList=[]
            for annoI in DB.annotations:
                anno = DB.annotations[annoI]
                if anno.agreedClass==hotclass:
                    annoList.append([anno.x1,anno.y1])

            centers_DB = np.array(annoList)


            F1,TP,FP,FN = _F1_core(centers_DB, boxes, score,det_thres)
        



        sTP+=TP
        sFP+=FP
        sFN+=FN
        F1dict[resfile]=F1
        
    print('Overall: ')
    sF1 = 2*sTP/(2*sTP + sFP + sFN)
    print('TP:', sTP, 'FP:', sFP,'FN: ',sFN,'F1:',sF1)

    return sF1, F1dict



def optimize_threshold(databasefile, result_boxes=None, resfile=None, hotclass=2, minthres=0.5):

    
    DB = Database()
    DB = DB.open(databasefile)

    if (result_boxes is None):
        if resfile is None:
            raise ValueError('At least one of resfile/result_boxes must be given')

    if (resfile[-3:] == 'bz2'):
        f = bz2.BZ2File(resfile, 'rb')
    else:
        f = open(resfile,'rb')

    result_boxes = pickle.load(f)


    sTP, sFN, sFP = 0,0,0
    F1dict = dict()
    
    MIN_THR = minthres

    result_boxes = nms(result_boxes, MIN_THR)
    TPd, FPd, FNd, F1d = dict(), dict(), dict(), dict()
    thresholds = np.arange(MIN_THR,0.99,0.01)
    
    print('Optimizing threshold for validation set of %d files: '%len(result_boxes.keys()))#, ','.join(list(result_boxes.keys())))

    for resfile in result_boxes:
        boxes = np.array(result_boxes[resfile])

        TP, FP, FN = 0,0,0
        TPd[resfile] = list()
        FPd[resfile] = list()
        FNd[resfile] = list()
        F1d[resfile] = list()

        if (boxes.shape[0]>0):
            score = boxes[:,-1]

            DB.loadIntoMemory(DB.findSlideWithFilename(resfile,''))
        
            # perform NMS on detections

            annoList=[]
            for annoI in DB.annotations:
                anno = DB.annotations[annoI]
                if anno.agreedClass==hotclass:
                    annoList.append([anno.x1,anno.y1])

            centers_DB = np.array(annoList)



            for det_thres in thresholds:
                F1,TP,FP,FN = _F1_core(centers_DB, boxes, score,det_thres)
                TPd[resfile] += [TP]
                FPd[resfile] += [FP]
                FNd[resfile] += [FN]
                F1d[resfile] += [F1]
        else:
            for det_thres in thresholds:
                TPd[resfile] += [0]
                FPd[resfile] += [0]
                FNd[resfile] += [0]
                F1d[resfile] += [0]
            F1 = 0
            

        F1dict[resfile]=F1

    allTP = np.zeros(len(thresholds))
    allFP = np.zeros(len(thresholds))
    allFN = np.zeros(len(thresholds))
    allF1 = np.zeros(len(thresholds))
    allF1M = np.zeros(len(thresholds))



    for k in range(len(thresholds)):
        allTP[k] = np.sum([TPd[x][k] for x in result_boxes])
        allFP[k] = np.sum([FPd[x][k] for x in result_boxes])
        allFN[k] = np.sum([FNd[x][k] for x in result_boxes])
        allF1[k] = 2*allTP[k] / (2*allTP[k] + allFP[k] + allFN[k])
        allF1M[k] = np.mean([F1d[x][k] for x in result_boxes])

    print('Best threshold: F1=', np.max(allF1), 'Threshold=',thresholds[np.argmax(allF1)])
        
    return thresholds[np.argmax(allF1)], allF1, thresholds


