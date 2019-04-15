"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"  # specify which GPU(s) to be used
#os.environ["CUDA_VISIBLE_DEVICES"]="1,3"

import sys
import json
import datetime
import numpy as np
import skimage.draw

# test ..
import scipy
from scipy import misc
#import matplotlib.pyplot as plt

import cv2

#from imgaug import augmenters as iaa


# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class BoltsConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "voronoi"
    
    # Can override..? yes. make sure for batch_size!
    GPU_COUNT = 1

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + bolt & corrosion)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 80% confidence
    DETECTION_MIN_CONFIDENCE = 0.6
    
    MAX_GT_INSTANCES = 250

    

############################################################
#  Dataset
############################################################

class BoltsDataset(utils.Dataset):

    def load_bolts(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        self.add_class("voronoi", 1, "textRegion")                 
        #self.add_class("voronoi", 2, "imageRegion")
        """
        self.add_class("voronoi", 3, "lineDrawingRegion")
        self.add_class("voronoi", 4, "graphicRegion")
        self.add_class("voronoi", 5, "tableRegion")
        self.add_class("voronoi", 6, "chartRegion")
        self.add_class("voronoi", 7, "separatorRegion")
        self.add_class("voronoi", 8, "mathsRegion")
        self.add_class("voronoi", 9, "noiseRegion")
        self.add_class("voronoi", 10, "frameRegion")
        self.add_class("voronoi", 11, "unknownRegion")        
        """
        
        # Train or validation dataset?
        assert subset in ["train", "val"]  ## Can I make a split function?
        dataset_dir = os.path.join(dataset_dir, subset)
        
        
        '''
        For LableBox Labeling Tool
        '''
        # Load annotations
        annotations = json.load(open(os.path.join(dataset_dir, "mike.json")))

        #Skip unannotated images.
        annotations = [a for a in annotations if type(a['Label'])==dict] 
        
        # Add images
        for a in annotations:
            dic = a['Label']
            polygon_class1 = []
            polygon_class2 = []
            polygon_class3 = []
            polygon_class4 = []
            polygon_class5 = []
            polygon_class6 = []
            polygon_class7 = []
            polygon_class8 = []
            polygon_class9 = []
            polygon_class10 = []            
            polygon_class11 = []            

            if 'textRegion' in dic.keys():
                for r in dic.get('textRegion'):
                    for j in r.values():
                        polygon_class1.append(j)
            """
            if 'imageRegion' in dic.keys():
                for r in dic.get('imageRegion'):
                    for j in r.values():
                        polygon_class2.append(j)
            """
            """
            if 'lineDrawingRegion' in dic.keys():
                for r in dic.get('lineDrawingRegion'):
                    for j in r.values():
                        polygon_class3.append(j)        
            if 'graphicRegion' in dic.keys():
                for r in dic.get('graphicRegion'):
                    for j in r.values():
                        polygon_class4.append(j)        
            if 'tableRegion' in dic.keys():
                for r in dic.get('tableRegion'):
                    for j in r.values():
                        polygon_class5.append(j)        
            if 'chartRegion' in dic.keys():
                for r in dic.get('chartRegion'):
                    for j in r.values():
                        polygon_class6.append(j)                                
            if 'separatorRegion' in dic.keys():
                for r in dic.get('separatorRegion'):
                    for j in r.values():
                        polygon_class7.append(j)        
            if 'mathsRegion' in dic.keys():
                for r in dic.get('mathsRegion'):
                    for j in r.values():
                        polygon_class8.append(j)                                
            if 'noiseRegion' in dic.keys():
                for r in dic.get('noiseRegion'):
                    for j in r.values():
                        polygon_class9.append(j)                                
            if 'frameRegion' in dic.keys():
                for r in dic.get('frameRegion'):
                    for j in r.values():
                        polygon_class10.append(j)                                
            if 'unknownRegion' in dic.keys():
                for r in dic.get('unknownRegion'):
                    for j in r.values():
                        polygon_class11.append(j)                                
            """          
            image_path = os.path.join(dataset_dir, a['ID']+'.jpg')          
            image = skimage.io.imread(image_path, plugin='matplotlib') # not working with up_data
            h, w = image.shape[:2]
            
            self.add_image(
                "voronoi",
                image_id=a['ID'],
                path=image_path,
                width=w, height=h,
                class_num=1,
                polygons1=polygon_class1,
                polygons2=polygon_class2)
            '''
                            polygons3=polygon_class3,
                polygons4=polygon_class4,
                polygons5=polygon_class5,
                polygons6=polygon_class6,
                polygons7=polygon_class7,
                polygons8=polygon_class8,
                polygons9=polygon_class9,                
                polygons10=polygon_class10,
                polygons11=polygon_class11
            '''
            
            print("image {} is added".format(a['ID']+'.jpg'))
            
        
    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        info = self.image_info[image_id]
        if info["source"] != "voronoi":
            return super(self.__class__, self).load_mask(image_id)
        

        # Convert polygons to a bitmap mask of shape
        voronois = []
        #count = len(info["polygons1"])+len(info["polygons2"])+len(info["polygons3"])+len(info["polygons4"])+len(info["polygons5"])+len(info["polygons6"])+len(info["polygons7"])+len(info["polygons8"])+len(info["polygons9"])+len(info["polygons10"])+len(info["polygons11"])
        count = len(info["polygons1"])#+len(info["polygons2"])
        #print("number of instances: {}".format(count))
        mask = np.zeros([info["height"], info["width"], count], dtype=np.uint8)
        #print("size of mask: {}".format(mask.shape))
        mask_idx = 0
        for i, p in enumerate(info["polygons1"]):
            row = []
            col = []
            for r in p:
                row.append(r['y'])
                col.append(r['x'])
            rr, cc = skimage.draw.polygon(row, col)
            mask[rr, cc, mask_idx] = 1            
            textRegion_tuple = ('textRegion', tuple(row), tuple(col))
            voronois.append(textRegion_tuple)
            mask_idx+=1
        """
        for i, p in enumerate(info["polygons2"]):
            row = []
            col = []
            for r in p:
                row.append(r['y'])
                col.append(r['x'])
            rr, cc = skimage.draw.polygon(row, col)
            mask[rr, cc, mask_idx] = 2     
            imageRegion_tuple = ('imageRegion', tuple(row), tuple(col))
            voronois.append(imageRegion_tuple)
            mask_idx+=1
        """
        """    
        for i, p in enumerate(info["polygons3"]):
            row = []
            col = []
            for r in p:
                row.append(r['y'])
                col.append(r['x'])
            rr, cc = skimage.draw.polygon(row, col)
            mask[rr, cc, i+start_idx] = 3
            start_idx = start_idx+1            
            lineDrawingRegion_tuple = ('lineDrawingRegion', tuple(row), tuple(col))
            voronois.append(lineDrawingRegion_tuple)

        for i, p in enumerate(info["polygons4"]):
            row = []
            col = []
            for r in p:
                row.append(r['y'])
                col.append(r['x'])
            rr, cc = skimage.draw.polygon(row, col)
            mask[rr, cc, i+start_idx] = 4
            start_idx = start_idx+1            
            graphicRegion_tuple = ('graphicRegion', tuple(row), tuple(col))
            voronois.append(graphicRegion_tuple)

        for i, p in enumerate(info["polygons5"]):
            row = []
            col = []
            for r in p:
                row.append(r['y'])
                col.append(r['x'])
            rr, cc = skimage.draw.polygon(row, col)
            mask[rr, cc, i+start_idx] = 5
            start_idx = start_idx+1            
            tableRegion_tuple = ('tableRegion', tuple(row), tuple(col))
            voronois.append(tableRegion_tuple)

        for i, p in enumerate(info["polygons6"]):
            row = []
            col = []
            for r in p:
                row.append(r['y'])
                col.append(r['x'])
            rr, cc = skimage.draw.polygon(row, col)
            mask[rr, cc, i+start_idx] = 6
            start_idx = start_idx+1            
            chartRegion_tuple = ('chartRegion', tuple(row), tuple(col))
            voronois.append(chartRegion_tuple)

        for i, p in enumerate(info["polygons7"]):
            row = []
            col = []
            for r in p:
                row.append(r['y'])
                col.append(r['x'])
            rr, cc = skimage.draw.polygon(row, col)
            mask[rr, cc, i+start_idx] = 7
            start_idx = start_idx+1            
            separatorRegion_tuple = ('separatorRegion', tuple(row), tuple(col))
            voronois.append(separatorRegion_tuple)

        for i, p in enumerate(info["polygons8"]):
            row = []
            col = []
            for r in p:
                row.append(r['y'])
                col.append(r['x'])
            rr, cc = skimage.draw.polygon(row, col)
            mask[rr, cc, i+start_idx] = 8
            start_idx = start_idx+1            
            mathsRegion_tuple = ('mathsRegion', tuple(row), tuple(col))
            voronois.append(mathsRegion_tuple)

        for i, p in enumerate(info["polygons9"]):
            row = []
            col = []
            for r in p:
                row.append(r['y'])
                col.append(r['x'])
            rr, cc = skimage.draw.polygon(row, col)
            mask[rr, cc, i+start_idx] = 9
            start_idx = start_idx+1            
            noiseRegion_tuple = ('noiseRegion', tuple(row), tuple(col))
            voronois.append(noiseRegion_tuple)

        for i, p in enumerate(info["polygons10"]):
            row = []
            col = []
            for r in p:
                row.append(r['y'])
                col.append(r['x'])
            rr, cc = skimage.draw.polygon(row, col)
            mask[rr, cc, i+start_idx] = 10
            start_idx = start_idx+1            
            frameRegion_tuple = ('frameRegion', tuple(row), tuple(col))
            voronois.append(frameRegion_tuple)

        for i, p in enumerate(info["polygons11"]):
            row = []
            col = []
            for r in p:
                row.append(r['y'])
                col.append(r['x'])
            rr, cc = skimage.draw.polygon(row, col)
            mask[rr, cc, i+start_idx] = 11
            start_idx = start_idx+1            
            unknownRegion_tuple = ('unknownRegion', tuple(row), tuple(col))
            voronois.append(unknownRegion_tuple)
        """

        # Map class names to class IDs.
        #class_ids = np.array([self.class_names.index(v[0]) for v in voronois])
        #return mask, class_ids.astype(np.int32)
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "voronoi":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
    '''
    MODIFYING NOW -----------------
    '''
    def augment_image(self, idx, image_name, num_sample):
        num_sample = num_sample
        w_sample = 1500
        h_sample = 1000
        
        # info : class, id, path, w, h, polygons
        info = self.image_info[idx]
        
        # read image from the index
        image = skimage.io.imread(info["path"], plugin='matplotlib')
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        new_mask = np.zeros([h_sample, w_sample, len(info["polygons"])], dtype=np.uint8)
        
        # Do crop 200*300
        # find top-left points
        x1 = np.random.choice(3200-h_sample, num_sample) # random choice x1: 0~(3200-1000)
        y1 = np.random.choice(4800-w_sample, num_sample) # random choice y1: 0~(4800-1500)
        x2 = x1+h_sample
        y2 = y1+w_sample
        
        print("{}, {}, {}, {}".format(x1[0], y1[0], x2[0], y2[0]))
        print("{}, {}, {}, {}".format(x1[1], y1[1], x2[1], y2[1]))
        print("{}, {}, {}, {}".format(x1[2], y1[2], x2[2], y2[2]))

        
        # draw original mask
        num_polygon = 0
        new_masks = []
        for i, p in enumerate(info["polygons"]):
            x = p['x']
            y = p['y']
            w = p['width']
            h = p['height']
            col = np.array([x, x+w, x+w, x])
            row = np.array([y, y, y+h, y+h])
            rr, cc = skimage.draw.polygon(row, col) 
            mask[rr, cc, i] = 1
            for j in range(num_sample):
                if abs(x-x1[j])<w or abs(y-y1[j]<h):
                    new_masks.append([mask[x1[j]:x2[j],y1[j]:y2[j],i]])
            num_polygon = num_polygon+1
           
        print("New masks len {}".format(len(new_masks)))
        
        # Update image_info
        #   remove original image from the dataset
        self.delete_image(idx)    
        
        #   add cropped image & masks
        for j in range(num_sample):
            cropped = image[x1[j]:x2[j],y1[j]:y2[j]]
            print("{}th: {}".format(j, '{}_{}'.format(str(j),image_name)))
            
            #plt.imshow(cropped)
            #plt.show
            
            # save image to path
            new_name = '{}_{}'.format(str(j),image_name)
            new_path='{}{}'.format(info["path"][:-12], new_name)
            skimage.io.imsave(fname=new_path, arr=cropped)
            new_polygons=[]
            #print("{}: {} polygons in image. type = {},{}".format(image_name, num_polygon, type(image), type(cropped)))
            #for k in range(1,num_polygon):
            #    new_polygons.append(mask[x1[j]:x2[j],y1[j]:y2[j],k].tolist())
                
            self.add_image(
                "voronoi",
                image_id=new_name,
                path=new_path,
                width=w_sample, height=h_sample,
                masks=np.array(new_masks[j]))
            

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = BoltsDataset()
    dataset_train.load_bolts(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BoltsDataset()
    dataset_val.load_bolts(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30, #default = 30
                layers='all') #default = 'heads'


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to segment document image.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        default='./../dataset/mike',
                        metavar="/path/to/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=False,
                        default='coco',
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BoltsConfig()
    else:
        class InferenceConfig(BoltsConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Image augmentation
    """
    augmentation = iaa.Sometimes(0.9, [
        iaa.CoarseDropout(0.01, size_percent=0.5),
        iaa.Affine(shear=(-3,3)),
        iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])
    """
    
    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
