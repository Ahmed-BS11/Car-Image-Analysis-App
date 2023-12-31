import os
import tensorflow as tf
import cv2
from tensorflow import keras
import numpy as np
from ultralytics import YOLO
import  json
import torch
from matplotlib import pyplot as plt
from PIL import Image
################
import logging
from collections import OrderedDict
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.utils.events import EventStorage
from detectron2.modeling import build_model
import detectron2.utils.comm as comm
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch

################

register_coco_instances("coco_dataset", {},
                            "/content/CarDD_release/CarDD_COCO/annotations/instances_val2017.json",
                            "/content/CarDD_release/CarDD_COCO/val2017")

##############


class damage_mask:
  def __init__(self,model_path,number_of_classes):
    # Assuming your dataset is in COCO format


    # Getting the metadata
    self.car_damage_metadata = MetadataCatalog.get("coco_dataset")
    self.weights_path=model_path
    self.number_of_classes=number_of_classes
  def setup_cfg(self):
      """
      Set up and return the Detectron2 config.
      """
      cfg = get_cfg()
      cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
      cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.number_of_classes # Change as per your model's classes
      cfg.MODEL.WEIGHTS = self.weights_path
      cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the detection threshold
      cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
      return cfg

  def load_model(self):
      """
      Load the Detectron2 model with the specified weights.
      """
      cfg = self.setup_cfg()
      return DefaultPredictor(cfg)


  def detect_damage(self,image):
      """
      Perform car damage detection and return the image with visualized predictions.
      """
      model=self.load_model()
      outputs = model(image)
      v = Visualizer(image[:, :, ::-1],
                    metadata=self.car_damage_metadata,
                    scale=1,)
      out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
      print(outputs["instances"].to("cpu"))
      return outputs["instances"].to("cpu") , Image.fromarray(out.get_image()[:, :, ::-1])