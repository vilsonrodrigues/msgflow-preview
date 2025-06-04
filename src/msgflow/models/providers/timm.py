from typing import Dict, Literal, Optional
try:
    import timm
    import torch
except:
    raise ImportError("`timm` is not detected, please install"
                      "using `pip install msgflow[timm]`")

from msgflow.models.providers.base import BaseVision, BaseVisionClassifier
from msgflow.models.response import ModelResponse
from msgflow.models.base import BaseModel
from msgflow.models.types import ImageClassifierModel, ImageEmbedderModel
from msgflow.utils.torch import TORCH_DTYPE_MAP


class _BaseTimm(BaseModel, BaseVision):

    provider: str = "timm"

    def __init__(
        self,
        *,
        model_id: Optional[str] = "vit_so400m_patch14_siglip_384",
        device: Optional[Literal["cpu", "cuda"]] = "cpu",
        dtype: Optional[Literal["float32", "float16", "bfloat16"]] = "float32",
        compile: Optional[bool] = False,
        return_score: Optional[bool] = False,
        id2label: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self.device = device
        self.model_id = model_id
        self.sampling_params = {
            "model_name": model_id,
            "pretrained": True,
        }
        if self.model_type == "image_embedder":
            self.sampling_params["num_classes"] = 0                        
        self.dtype = TORCH_DTYPE_MAP[dtype]
        self.compile = compile
        self.return_score = return_score
        self.id2label = id2label
        self._initialize()

    def _initialize(self):
        model = timm.create_model(**self.sampling_params)
        self.model = model.eval().to(self.dtype).to(self.device)
        if self.compile:
            self.model = torch.compile(self.model)        
        data_config = timm.data.resolve_model_data_config(self.model)
        self.processor = timm.data.create_transform(**data_config, is_training=False)

class TimmImageEmbedder(_BaseTimm, ImageEmbedderModel):

    def _generate(self, images):       
        response = ModelResponse()
        model_output = self._execute_model(images)
        model_output_list = model_output.tolist()
        response.set_response_type("image_embedding")
        response.add(model_output_list)        
        return response

class TimmImageClassifier(_BaseTimm, ImageClassifierModel, BaseVisionClassifier):  
    """ TIMM Image Classifier"""