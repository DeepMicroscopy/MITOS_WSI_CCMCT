"""

        Process 2nd stage of approach 
        
        Requires:
            Results of first stage (pickle format: .p)
            2nd stage model (.pth)

        Outputs:
            Will output a file with same name as first results, but with a prefix of "2ndstage_".
        
        Syntax:
            classify_2ndStage.py results.p model.pth [imagePath]
            
        
        Marc Aubreville, Pattern Recognition Lab, FAU Erlangen-Nürnberg, 2019

"""
import sys

sys.path.append('../')
from lib.processSecondStageClassification import *
import pickle
import sys,os
if len(sys.argv)<2:
    print('Syntax: classify_2ndStage.py results.p model.pth')
    exit()
fname = sys.argv[1]
p = pickle.load(open(fname, 'rb'))
path,fname = os.path.split(fname)
if len(path)>0:
    path+='/'
basepath='../../WSI/' if len(sys.argv)<=3 else sys.argv[3]
modelpath='CellClassifier_128px.pth' if len(sys.argv)<=2 else sys.argv[2]
out_2ndstage = processSecondStageClassifier(p,  intermediate=path+'2ndstage_'+fname, modelpath=modelpath, basepath=basepath)
pickle.dump(out_2ndstage, open(path+'2ndstage_'+fname,'wb'))
