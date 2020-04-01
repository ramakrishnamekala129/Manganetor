import torch, torchvision
import detectron2

# import some common libraries
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
import os
import json
from detectron2.structures import BoxMode
import pdb, argparse
from dataset import get_balloon_dicts
torch.__version__




def get_opt():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", default = "data")
	parser.add_argument("--learn_rate", type=int, default = 0.00025,help='set learning rate')
	parser.add_argument('--n_iter', type=int, default=10000, help='number of iteration')
	opt = parser.parse_args()
	return opt


opt = get_opt()

for d in ["train", "valid"]:
    DatasetCatalog.register("manga_" + d, lambda d=d: get_balloon_dicts(opt.dataset + d))
    MetadataCatalog.get("manga_" + d).set(thing_classes=["manga"])
balloon_metadata = MetadataCatalog.get("manga_train")

dataset_dicts = get_balloon_dicts(opt.dataset+'train')
# for d in random.sample(dataset_dicts, 3):
#     img = plt.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     plt.figure(dpi=180)
#     plt.imshow(vis.get_image()[:, :, ::-1])
    
    
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("manga_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = opt.learn_rate  #0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = opt.n_iter    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()