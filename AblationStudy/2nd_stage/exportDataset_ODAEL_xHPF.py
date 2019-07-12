"""

        Export 2nd-stage dataset (patches of 128px) to file system
        
        -- ONLY EXPORT a certain area from ODAEL data set variant --
        
        Requirement to be run: 
            Reduced whole slide images need to be in images/xHPF/ folder (please get from figshare)
            
        
        Syntax:
            exportDataset_ODAEL_xHPF.py [5|10|50]
            
        
        Marc Aubreville, Pattern Recognition Lab, FAU Erlangen-Nürnberg, 2019

"""
import sys

if (len(sys.argv)<2) or (sys.argv[1] not in ['5','10','50']):
    print('syntax: exportDataset_ODAEL_xHPF.py x')
    print('   where x is the HPF area of the WSI subset to export (5,10,50)')
    exit()

    
import numpy as np 
import SlideRunner.general.dependencies
from SlideRunner.dataAccess.database import Database
import os
import openslide
import sqlite3
import cv2

DB = Database()

basepath='../images/%sHPF/' % sys.argv[1]

patchSize=128

def listOfSlides(DB):
    DB.execute('SELECT uid,filename from Slides')
    return DB.fetchall()

os.system('mkdir -p DataODAEL_%sHPF' % sys.argv[1])

dirs = ['Mitosis', 'Mitosislike', 'Tumorcells', 'Granulocytes']
for k in dirs:
    os.system('mkdir -p DataODAEL_%sHPF/train/%s' % (sys.argv[1],k))
    os.system('mkdir -p DataODAEL_%sHPF/test/%s' % (sys.argv[1],k))


test_slide_filenames = ['3369_07_B_1_MCT Mitose 2017.tiff',
 '3786_09 A MCT Mitose 2017.tiff',
 '1659_08_1_MCT Mitose 2017.tiff',
 '28_08_A_1_MCT Mitose 2017.tiff',
 '3806_09_B_1_MCT Mitose 2017.tiff',
 '2253_06_A_1_MCT Mitose 2017.tiff',
 '1410_08_A_1_MCT Mitose 2017.tiff',
 '1490_08_1_MCT Mitose 2017.tiff',
 '2281_14_A_1_MCT Mitose 2017.tiff',
 '221_08 MCT Mitose 2017.tiff',
 '5187_11 B MCT Mitose 2017.tiff']

DB.open('../databases/MITOS_WSI_CMCT_ODAEL_%sHPF.sqlite' % sys.argv[1])#Slides_final_cleaned_checked.sqlite')

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
            if not cv2.imwrite(('DataODAEL_%sHPF/' % sys.argv[1])+istest+'Mitosis/%d.png' % (k), img):
                  print('Write failed: '+('DataODAEL_%sHPF/' % sys.argv[1])+istest+'Mitosis/%d.png' % (k))

        if (anno.agreedClass==7):
            cv2.imwrite(('DataODAEL_%sHPF/' % sys.argv[1])+istest+'Mitosislike/%d.png' % k, img)

        if (anno.agreedClass==3):
            cv2.imwrite(('DataODAEL_%sHPF/' % sys.argv[1])+istest+'Tumorcells/%d.png' %k, img)

        if (anno.agreedClass==1):
            cv2.imwrite(('DataODAEL_%sHPF/' % sys.argv[1])+istest+'Granulocytes/%d.png' %k, img) 



