from typing import List, Literal, Optional, Union
import numpy as np
try:
    import torch
    from torchaudio import pipelines
except:
    raise ImportError("`torchaudio` not detected, please install"
                      "using `pip install msgflow[torchaudio]`")
from msgflow.models.response import Response
from msgflow.models.base import BaseClient
from msgflow.models.types import AudioEmbedderModel
from msgflow.utils.pooling import apply_pooling
from msgflow.utils.torch import TORCH_DTYPE_MAP


class _BaseTorchAudio(BaseClient):

    provider: str = "torchaudio"

    def __init__(
        self,
        *,
        model_id: str,
        device: Optional[Literal["cpu", "cuda"]] = "cpu",
        dtype: Optional[Literal["float32", "float16", "bfloat16"]] = "float32",
        compile: Optional[bool] = False,
        pooling_strategy: Optional[Literal["mean", "max", "cls"]] = "mean",
    ):
        super().__init__()
        self.model_id = model_id
        self.device = device
        self.sampling_params = {"model_id": model_id.upper()}                      
        self.dtype = TORCH_DTYPE_MAP[dtype]
        self.compile = compile
        self.pooling_strategy = pooling_strategy
    
    def _initialize_client(self):
        bundle = pipelines.__dict__[self.sampling_params.get("model_id")]
        model = bundle.get_model()
        self.model = model.eval().to(self.dtype).to(self.device)
        if self.compile:
            self.model = torch.compile(self.model)        

    def __call__(
        self,
        data: Union[
            np.ndarray, List[np.ndarray]
        ],
    ) -> Response:
        if not isinstance(data, list):
            data = [data]
        response = self._generate(data=data)
        return response


class TorchAudioAudioEmbedder(_BaseTorchAudio, AudioEmbedderModel):

    @torch.inference_mode()
    def _execute_model(self, data):
        inputs = torch.from_numpy(data)
        inputs = inputs.to(self.dtype).to(self.device)
        model_output, _ = self.model.extract_features(inputs)
        model_output = model_output.cpu()
        return model_output
    
    def _generate(self, data):
        response = Response()
        model_output = self._execute_model(data)
        last_hidden_state = model_output[-1]
        embeddings = apply_pooling(last_hidden_state.numpy(), self.pooling_strategy)
        embeddings_list = embeddings.tolist()        
        response.set_response_type("audio_embeddings")
        response.add(embeddings_list)
        return response    