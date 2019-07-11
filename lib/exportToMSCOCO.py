from SlideRunner.dataAccess.database import Database
import sys
import json

import openslide
class exportableDataset(Database):
    def exportToMSCOCO(self, targetFilename:str):
        """
           Export database to MS COCO format
        """
        db = dict()
        db['info'] = {
            "description": "Large-Scale Mitotic Figure Dataset for Canine Cutaneous Mast Cell Tumors (MITOS_WSI_CMCT)",
            "version": "1.0",
            "year": 2019,
            "contributor": "Christof A. Bertram, Marc Aubreville, Christian Marzahl, Andreas Maier, Robert Klopfleisch",
            "date_created": "2019/07/05"
        }
        db['licenses'] = [
            {
                "url": "http://creativecommons.org/licenses/by-nc-nd/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-NoDerivs License"
            },
        ]
        
        db['images'] = list()
        print('Collecting image info')
        for id,fname in DB.listOfSlides():
            img = {'license':1, 'file_name':fname, 'id':id}
            
            
            sl = openslide.open_slide('../WSI/'+fname)
            
            img['width'], img['height']  = sl.dimensions
            db['images'].append(img)
            
            print(img)
        db['categories'] = list()
        for id, name in DB.execute('SELECT uid, name from Classes').fetchall():
            db['categories'].append({'id':id, 'name':name})
            
        print('Now reading all objects..')
        db['annotations'] = list()
        for annoid,x,y,agreedClass,slide in DB.execute('SELECT annoId, coordinateX, coordinateY, agreedClass, Annotations.slide FROM Annotations_coordinates LEFT JOIN Annotations on Annotations.uid == Annotations_coordinates.annoId').fetchall():
            db['annotations'].append({'bbox': [x-25,y-25,x+25,y+25],
                                      'category_id' : agreedClass,
                                      'image_id' : slide,
                                      'id' : annoid})
            
            
        with open(targetFilename,'w') as f:
            f.write(json.dumps(db))
        
        
if (__name__ == '__main__'):
    if len(sys.argv)<3:
        print('Syntax: exportToMSCOCO databasefile.sqlite targetFilename.json')
    
    DB = exportableDataset()
    DB.open(sys.argv[1])
    DB.exportToMSCOCO(sys.argv[2])