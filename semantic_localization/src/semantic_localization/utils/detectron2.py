from typing import List, Tuple

import cv2
from datetime import datetime
import numpy as np
import os
import torch

from detectron2.config.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, VisImage, Visualizer, _PanopticPrediction

import rospy
from sensor_msgs.msg import Image
from tue_msgs.msg import MaskedImage
from tue_msgs.srv import GetMaskedImage, GetMaskedImageRequest, GetMaskedImageResponse

from semantic_localization.utils.conversions import color_imgmsg_to_bgr_np_ndarray, uint_np_ndarray_to_mono_imgmsg


class D2Annotator:
    def __init__(
        self,
        cfg: CfgNode,
        instance_mode: ColorMode = ColorMode.IMAGE,
        save_masked_image: bool = False,
    ):
        """
        :param cfg:
        :param instance_mode
        :param save_masked_image: Save original and masked image
        """
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor = DefaultPredictor(cfg)

        self.save_masked_image = save_masked_image

    def run_on_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Run model on provided image.

        :param image: an image of shape (H, W, C) (in BGR order). This is the format used by OpenCV.
        :return: Labeled image and labels
        """
        predictions = self.predictor(image)

        if self.save_masked_image:
            self.save_images(image, predictions)

        return self.predictions_to_masked_image(image, predictions)

    def predictions_to_masked_image(self, image, predictions: dict) -> Tuple[np.ndarray, List[str]]:
        """
        Convert the predictions to a uint masked image

        :param image: original image
        :param predictions: predictions dictionary, can support both 'panoptic_seg' or 'sem_seg' and 'instances'
        :return:
        """
        output_image = np.full(image.shape[:2], 255, dtype=np.dtype("uint8"))
        output_labels = []

        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            pred = _PanopticPrediction(panoptic_seg.to(self.cpu_device), segments_info, self.metadata)

            for mask, info in pred.semantic_masks():
                np.place(output_image, mask, len(output_labels))
                output_labels.append(self.metadata.stuff_classes[info["category_id"]])

            for mask, info in pred.instance_masks():
                np.place(output_image, mask, len(output_labels))
                label = self.metadata.thing_classes[info["category_id"]]
                instance_counter = sum(bool(f"{label}^" in item) for item in output_labels)
                output_labels.append(f"{label}^{instance_counter}")
        else:
            if "sem_seg" in predictions:
                sem_seg = predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                if isinstance(sem_seg, torch.Tensor):
                    sem_seg = sem_seg.numpy()
                labels, areas = np.unique(sem_seg, return_counts=True)
                sorted_idxs = np.argsort(-areas).tolist()
                labels = labels[sorted_idxs]

                for label_id in labels:
                    mask = (sem_seg == label_id).astype(np.bool)
                    label = self.metadata.stuff_classes[label_id]
                    np.place(output_image, mask, len(output_labels))
                    output_labels.append(label)

            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                if instances.has("pred_masks") and instances.has("pred_classes"):
                    masks = np.asarray(instances.pred_masks)
                    labels = (self.metadata.thing_classes[x] for x in instances.pred_classes.tolist())
                    for mask, label in zip(masks, labels):
                        np.place(output_image, mask, len(output_labels))
                        instance_counter = sum(bool(f"{label}^" in item) for item in output_labels)
                        output_labels.append(f"{label}^{instance_counter}")

        return output_image, output_labels

    def save_images(self, image, predictions: dict) -> None:
        image_name = f"detectron2_{datetime.now().strftime('%y-%m-%d_%H:%M:%S')}"
        file_dirname = os.path.expanduser("~/ros/data/semantic_localization")
        filename_base = os.path.join(file_dirname, image_name)
        if not os.path.isdir(file_dirname):
            os.makedirs(file_dirname)

        filename_original = f"{filename_base}_original.jpg"
        rospy.logdebug(f"Saving original image to {filename_original}")
        # Convert image from BGR format to RGB format.
        cv2.imwrite(filename_original, image[:, :, ::-1])

        vis_output = None
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg(panoptic_seg.to(self.cpu_device), segments_info)
        else:
            if "sem_seg" in predictions:
                sem_seg = predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                vis_output = visualizer.draw_sem_seg(sem_seg)

            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        filename_masked = f"{filename_base}_masked.jpg"
        rospy.logdebug(f"Saving masked image to {filename_masked}")
        if isinstance(vis_output, VisImage):
            vis_output.save(filename_masked)
        else:
            rospy.logwarn("Could not save masked image")


class D2AnnotatorNode(D2Annotator):
    def __init__(
        self,
        cfg: CfgNode,
        service_name: str = None,
        input_topic: str = None,
        output_topic: str = None,
        instance_mode: ColorMode = ColorMode.IMAGE,
        save_masked_image: bool = False,
    ):
        super().__init__(cfg, instance_mode, save_masked_image)

        if service_name is not None:
            self.srv = rospy.Service(service_name, GetMaskedImage, self.srv_callback)

        if input_topic is not None and output_topic is not None:
            self.pub = rospy.Publisher(output_topic, GetMaskedImageResponse, queue_size=1)
            self.sub = rospy.Subscriber(input_topic, Image, self.topic_callback)

    def callback(self, input_image: Image) -> MaskedImage:
        img = color_imgmsg_to_bgr_np_ndarray(input_image)
        output_image, output_labels = self.run_on_image(img)

        return MaskedImage(image=uint_np_ndarray_to_mono_imgmsg(output_image), labels=output_labels)

    def srv_callback(self, req: GetMaskedImageRequest) -> MaskedImage:
        return GetMaskedImageResponse(output_image=self.callback(req.input_image))

    def topic_callback(self, msg: Image) -> None:
        res = self.callback(msg)
        self.pub.publish(res)
