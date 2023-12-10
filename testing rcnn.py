import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo
import torch
print(torch.cuda.is_available())
# Replace with the path to your trained weights file
MODEL_WEIGHTS_PATH = "model_final.pth"

# Replace with the path to the image you want to perform inference on
IMAGE_PATH = "car4.jpg"

# Load the configuration for inference
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))

# Set the number of classes based on your training configuration
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Adjust according to your dataset classes

# Set the model weights
cfg.MODEL.WEIGHTS = MODEL_WEIGHTS_PATH

# Create a predictor
predictor = DefaultPredictor(cfg)

# Read the input image
image = cv2.imread(IMAGE_PATH)

# Perform inference
outputs = predictor(image)

# Visualization
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Inference Result", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()