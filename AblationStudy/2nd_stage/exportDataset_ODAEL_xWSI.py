"""

        Export 2nd-stage dataset (patches of 128px) to file system
        
        -- ONLY EXPORT a certain number of WSI from ODAEL data set variant --
        
        Requirement to be run: 
            Whole slide images need to be in WSI/ folder (please get from figshare)
        
        Syntax:
            exportDataset_ODAEL_xWSI.py [3|6|12]
            
        
        Marc Aubreville, Pattern Recognition Lab, FAU Erlangen-Nürnberg, 2019

"""


import numpy as np 
import SlideRunner.general.dependencies
from SlideRunner.dataAccess.database import Database
import os
import openslide
import sqlite3
import cv2

import sys

if (len(sys.argv)<2) or (sys.argv[1] not in ['3','6','12']):
    print('syntax: exportDataset_ODAEL_xWSI.py x')
    print('   where x is the number of WSI to export (3,6,12)')
    exit()

DB = Database()


basepath='../../WSI/'
patchSize=128

os.system('mkdir -p DataODAEL_%sWSI' % sys.argv[1])

dirs = ['Mitosis', 'Mitosislike', 'Tumorcells', 'Granulocytes']
for k in dirs:
    os.system('mkdir -p DataODAEL_%sWSI/train/%s' % (sys.argv[1],k))
    os.system('mkdir -p DataODAEL_%sWSI/test/%s' % (sys.argv[1],k))

def listOfSlides(DB):
    DB.execute('SELECT uid,filename from Slides')
    return DB.fetchall()

test_slide_filenames = ['3369_07_B_1_MCT Mitose 2017.svs',
 '3786_09 A MCT Mitose 2017.svs',
 '1659_08_1_MCT Mitose 2017.svs',
 '28_08_A_1_MCT Mitose 2017.svs',
 '3806_09_B_1_MCT Mitose 2017.svs',
 '2253_06_A_1_MCT Mitose 2017.svs',
 '1410_08_A_1_MCT Mitose 2017.svs',
 '1490_08_1_MCT Mitose 2017.svs',
 '2281_14_A_1_MCT Mitose 2017.svs',
 '221_08 MCT Mitose 2017.svs',
 '5187_11 B MCT Mitose 2017.svs']

DB.open('../databases/MITOS_WSI_CMCT_ODAEL_%sWSI.sqlite' % sys.argv[1])#Slides_final_cleaned_checked.sqlite')

for slide,filename in listOfSlides(DB):
    DB.loadIntoMemory(slide)
    
    
    slide=openslide.open_slide(basepath+filename)

    for k in DB.annotations.keys():

        anno = DB.annotations[k]

        coord_x = anno.x1
        coord_y = anno.y1

        lu_x = int(coord_x - int(patchSize/2))
        lu_y = int(coord_y - int(patchSize/2))
        img = np.array(slide.read_region(location=(lu_x, lu_y), level=0, size=(patchSize, patchSize)))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        istest = 'train/' if filename not in test_slide_filenames else 'test/'
        if (anno.agreedClass==2):
            fname = ('DataODAEL_%sWSI/' % sys.argv[1])+istest+'Mitosis/%d.png' % (k)
            if not cv2.imwrite(fname, img):
                  print('Write failed: ',fname)

        if (anno.agreedClass==7):
            cv2.imwrite(('DataODAEL_%sWSI/' % sys.argv[1])+istest+'Mitosislike/%d.png' % k, img)

        if (anno.agreedClass==3):
            cv2.imwrite(('DataODAEL_%sWSI/' % sys.argv[1])+istest+'Tumorcells/%d.png' %k, img)

        if (anno.agreedClass==1):
            cv2.imwrite(('DataODAEL_%sWSI/' % sys.argv[1])+istest+'Granulocytes/%d.png' %k, img) 



