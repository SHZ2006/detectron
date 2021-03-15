import json
#import numpy as np
#import time
import cv2
# Detectron2
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

# Import Mask RCNN
#from mrcnn.config import Config
#from mrcnn import model as modellib, utils
#from mrcnn import visualize



#===========================================================================================

class Inference_object:
    
    def __init__(self, weights_path):

        train_conf = "conf_cube.json"
        with open(train_conf) as f:
            train_conf = json.load(f)
        self.dataset_classes = train_conf["dataset"]["dataset_classes"]

        self.metadata = MetadataCatalog.get("_train").set(thing_classes=self.dataset_classes)
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.DATASETS.TEST = ( "_val",)
        self.cfg.MODEL.WEIGHTS = weights_path   #os.path.join(cfg.OUTPUT_DIR, "weights/4objects_1600.pth")
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.dataset_classes)
        self.predictor = DefaultPredictor(self.cfg)

        pass


    def detect_objects2(self, rgb_image=None , realtime=False):
        #object_name=[]
        #cropped_image=[]

        if not rgb_image is None:
            #predictor = DefaultPredictor(self.cfg)
            rgb_image = cv2.convertScaleAbs(rgb_image)
            #start_time = time.time()
            outputs = self.predictor(rgb_image)
            #print("Prediction time :", time.time() - start_time)
            pred= outputs["instances"].to("cpu")
            obj_number = len(pred.pred_boxes)

            v = Visualizer(rgb_image,
                           metadata=self.metadata,
                           scale=1
                           )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            segmented =  v.get_image()[:, :, ::-1]
            
            if realtime == True:
                v = Visualizer(rgb_image,
                               metadata=self.metadata,
                               scale=0.9
                               )

                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                cv2.imshow("predictions", v.get_image()[:, :, ::-1])
                if cv2.waitKey()== 27:
                    cv2.destroyAllWindows()

            """
            for i in range (obj_number):
                object_name.append(self.dataset_classes[pred.pred_classes[i]])
                y1, x1, y2, x2 = pred.pred_boxes.tensor.numpy()[i]
                y1=int(y1)
                x1=int(x1)
                y2=int(y2)
                x2=int(x2)
                crop_image = np.ones(rgb_image.shape, dtype="uint8") * 255
                crop_image[ x1:x2 , y1:y2]= rgb_image[x1:x2 , y1:y2]
                cropped_image.append(crop_image)
            return object_name, cropped_image, obj_number, segmented
            """
            return obj_number, segmented



#if __name__ == '__main__':
    #inference = Inference('../weights/mask_rcnn_bag_0029.h5')
    #inference.detect_and_color_splash(image_path='../realbag.jpg')
