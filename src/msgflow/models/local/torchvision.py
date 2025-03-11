from typing import Literal, Optional
try:
    import torch
    from torchvision.models import get_model, get_model_weights
except:
    raise ImportError("`torchvision` not detected, please install"
                      "using `pip install msgflow[torchvision]`")
from msgflow.models.local.base import (
    BaseImageClassifier,
    BaseVideoClassifier,
    BaseVision,     
)    
from msgflow.models.response import Response
from msgflow.models.base import BaseClient
from msgflow.models.types import (
    ImageClassifierModel, 
    ImageSegmenterModel, 
    ObjectDetectorModel,
    VideoClassifierModel
)
from msgflow.utils.torch import TORCH_DTYPE_MAP
from msgflow.telemetry.events import EventsTiming


class _BaseTorchVision(BaseClient, BaseVision):

    provider: str = "torchvision"

    def __init__(
        self,
        *,
        model_id: str,
        device: Optional[Literal["cpu", "cuda"]] = "cpu",
        dtype: Optional[Literal["float32", "float16", "bfloat16"]] = "float32",
        compile: Optional[bool] = False,
        return_score: Optional[bool] = False,
        box_score_thresh: Optional[float] = 0.2,
    ):
        super().__init__()
        self.model_id = model_id
        self.device = device
        self.dtype = TORCH_DTYPE_MAP[dtype]
        self.compile = compile
        self.return_score = return_score
        self.sampling_params = {"name": model_id}        
        if self.model_type == "object_detector":
            self.sampling_params["box_score_thresh"] = box_score_thresh
        self._initialize_client()
    
    def _initialize_client(self):
        weight_enum = get_model_weights(**self.sampling_params)
        weights = weight_enum.DEFAULT        
        model = get_model(**self.sampling_params, weights=weights)
        self.model = model.eval().to(self.dtype).to(self.device)
        if self.compile:
            self.model = torch.compile(self.model)        
        self.processor = weights.transforms()
        self.id2label = weights.meta["categories"]
    
class TorchVisionImageClassifier(_BaseTorchVision, 
                                 BaseImageClassifier,
                                 ImageClassifierModel):
    """ TorchVision Image Classifier """

class TorchVisionImageSegmenter(_BaseTorchVision, ImageSegmenterModel):
    ...
    # TODO    
    # https://pytorch.org/vision/main/auto_examples/others/plot_visualization_utils.html#instance-seg-output

class TorchVisionObjectDetector(_BaseTorchVision, BaseVision, ObjectDetectorModel):
    """ TorchVision Object Detector """

    def _process_det_output(self, model_output):
        batch_predictions = []
        
        for o in model_output:
            # TODO: score filter
            # Filter predictions by confidence threshold
            # mask = o["scores"] >= self.threshold
            
            #filtered_boxes = o["boxes"][mask]
            #filtered_labels = o["labels"][mask]
            #filtered_scores = o["scores"][mask]
            
            # Process each detection in the image
            predictions = []
            for box, label_id, score in zip(o["boxes"], o["labels"], o["scores"]):

                score = score.item()
                label = self.id2label[label_id.item()]
                box = [round(coord.item(), 2) for coord in box]
                
                result = {
                    "label": label,
                    "box": box
                }
                
                if self.return_score:
                    result["score"] = round(score, 4)
                    
                predictions.append(result)
                
            batch_predictions.append(predictions)
            
        return batch_predictions

    def _generate(self, data):
        response = Response()
        metadata = {}
        events_timing = EventsTiming()

        events_timing.start("model_execution")
        events_timing.start("model_generation")                
        model_output = self._execute_model(data)
        events_timing.end("model_generation")

        predictions = self._process_det_output(model_output)
        events_timing.end("model_execution")

        metadata["timing"] = events_timing.get_events()
        metadata["model_info"] = self.get_model_info()
                
        response.set_response_type("object_detection")
        response.add(predictions)
        response.set_metadata(metadata)

        return response    

# TODO: key point detection
# https://github.com/pytorch/vision/blob/main/torchvision/models/detection/keypoint_rcnn.py

class TorchVisionVideoClassifier(_BaseTorchVision, 
                                 BaseVideoClassifier, 
                                 VideoClassifierModel):
    """ TODO doc """