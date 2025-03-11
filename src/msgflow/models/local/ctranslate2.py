from typing import List, Literal, Optional, Union
import subprocess
import numpy as np
try:
    import ctranslate2
    import torch
    from transformers import AutoModelForSequenceClassification, AutoProcessor
except:
    raise ImportError("`ctranslate2` not detected, please install"
                      "using `pip install msgflow[ctranslate2]`")    
from msgflow.models.response import Response
from msgflow.models.base import BaseClient
from msgflow.models.types import (
    TextClassifierModel,
    TextEmbedderModel,
)
from msgflow.utils.pooling import apply_pooling
from msgflow.telemetry.events import EventsTiming


def _ct2_transformers_converter(model_id: str, output_dir: str):
    cvtr = f"ct2-transformers-converter --model {model_id} --output_dir {output_dir}"
    subprocess.run(cvtr.split(" "))
    

CT2_DTYPE = Literal["int8", "int8_float32", "int8_float16", "int8_bfloat16", 
                    "int16", "float16", "bfloat16", "float32"]

class _BaseCTranslate2(BaseClient):

    provider: str = "ctranslate2"

    def __init__(
        self, 
        model_id: str, 
        device: Optional[Literal["cpu", "cuda"]] = "cpu",
        dtype: Optional[CT2_DTYPE] = "float32",
        processor_id: Optional[str] = None,
        pooling_strategy: Optional[Literal["mean", "max", "cls"]] = "mean",        
        trust_remote_code: Optional[bool] = False,
        flash_attn: Optional[bool] = False,
        tensor_parallel: Optional[bool] = False,
    ):
        super().__init__()
        self.processor_params = {"pretrained_model_name_or_path": processor_id or model_id,
                                 "trust_remote_code": trust_remote_code}
        self.processor_run_params = {"return_tensors": "pt"}
        self.sampling_params = {
            "device": device,            
            "compute_type": dtype,
            "flash_attention": flash_attn,
            "tensor_parallel": tensor_parallel,
            "trust_remote_code": trust_remote_code,
        }
        self.model_id = model_id
        self.pooling_strategy = pooling_strategy
        self._initialize_client()

    def _initialize_client(self):
        self._init_processor()
        self._convert_to_ct2()
        self._init_model()
        if self.model_type == "text_classifier":
            self._init_head_model()     

    def _convert_to_ct2(self):        
        self.ct2_model_id = f"{self.model_id.split("/")[1]}-ct2"        
        _ct2_transformers_converter(self.model_id, self.ct2_model_id)

    def _init_processor(self):
        self.processor = AutoProcessor.from_pretrained(**self.processor_params)

    @torch.inference_mode()
    def _execute_model(self, data):
        inputs = self.processor(data, **self.processor_run_params)
        tokens = inputs.input_ids        
        model_outputs = self.model.forward_batch(tokens)
        return model_outputs

    def __call__(self, data: Union[str, List[str]]) -> Response:
        if not isinstance(list):
            data = [data]
        return self._generate(data)


class CTranslate2TextEmbedder(_BaseCTranslate2, TextEmbedderModel):

    def _init_model(self):
        self.model = ctranslate2.Encoder(self.ct2_model_id, **self.sampling_params)        

    def _get_embeddings(self, last_hidden_state):
        if self.sampling_params.get("device") == "cuda":
            last_hidden_state = torch.as_tensor(
                last_hidden_state, 
                device=self.sampling_params.get("device")
            ).cpu().numpy()
        else:
            last_hidden_state = np.array(last_hidden_state)
        embeddings = apply_pooling(last_hidden_state, self.pooling_strategy)
        return embeddings

    def _generate(self, data):
        response = Response()
        metadata = {}        
        events_timing = EventsTiming()
        
        events_timing.start("model_execution")
        events_timing.start("model_generation")        
        model_output = self._execute_model(data)
        events_timing.end("model_generation")

        last_hidden_state = model_output.last_hidden_state
        embeddings = self._get_embeddings(last_hidden_state)
        embeddings_list = embeddings.tolist()

        events_timing.end("model_execution")

        metadata["timing"] = events_timing.get_events()
        metadata["model_info"] = self.get_model_info()

        response.set_response_type("text_embedding")
        response.add(embeddings_list)
        response.set_metadata(metadata)

        return response
    
class CTranslate2TextClassifier(_BaseCTranslate2, TextClassifierModel):

    def _init_head_model(self):
        model = AutoModelForSequenceClassification(self.model_id)
        self.id2label = model.config.id2label
        self.head_model = model.classifier.eval().to(self.sampling_params.get("device"))        

    def _execute_head_model(self, pooler_output) -> List[str]:
        if self.sampling_params.get("device") == "cuda":
            pooler_output = torch.as_tensor(
                pooler_output, 
                device=self.sampling_params.get("device"))
        else:
            pooler_output = np.array(pooler_output)
            pooler_output = torch.as_tensor(pooler_output)
        logits = self.head_model(pooler_output)
        predicted_class_ids = logits.argmax(1)
        labels = [self.id2label[id] for id in predicted_class_ids]
        return labels

    def _generate(self, data):
        response = Response()                
        metadata = {}        
        events_timing = EventsTiming()
        
        events_timing.start("model_execution")
        events_timing.start("model_generation")
        model_output = self._execute_model(data)        
        events_timing.end("model_generation")
        
        pooler_output = model_output.pooler_output    
        
        events_timing.start("model_head_generation")          
        
        labels = self._execute_head_model(pooler_output)
        
        events_timing.end("model_head_generation")
        events_timing.end("model_execution")
        
        metadata["timing"] = events_timing.get_events()
        metadata["model_info"] = self.get_model_info()
        
        response.set_response_type("text_classification")        
        response.add(labels)
        response.set_metadata(metadata)
        
        return response
