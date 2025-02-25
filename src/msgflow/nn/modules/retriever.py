from typing import Any, Dict, List, Optional, Union
from msgflow.message import Message
from msgflow.data.databases.types import VectorDB
from msgflow.models.types import (
    AudioEmbedderModel,
    ImageEmbedderModel,
    TextEmbedderModel,
)
from msgflow.data.retrievers.types import (
    HybridRetriever,
    LexicalRetriever,
    SemanticRetriever,
)
from msgflow.models.router import ModelRouter
from msgflow.nn.modules.module import Module
from msgflow.utils import to_io_object


class Retriever(Module):
    """
    dict_key

    """

    def __init__(
        self,
        *,
        name: str,
        retriever: Union[
            HybridRetriever, LexicalRetriever, SemanticRetriever, VectorDB
        ],
        model: Optional[
            Union[
                AudioEmbedderModel, ImageEmbedderModel, TextEmbedderModel, ModelRouter
            ]
        ] = None,
        task_inputs: Union[str, Dict[str, str]] = None,
        task_multimodal_inputs: Optional[Dict[str, List[str]]] = None,
        response_mode: Optional[str] = "plain_response",
        response_template: Optional[str] = None,
        top_k: Optional[int] = 4,
        threshold: Optional[float] = 0.0,
        return_score: Optional[bool] = False,
        dict_key: Optional[str] = None,
    ):
        super().__init__()

        if task_inputs is None and task_multimodal_inputs is None:
            raise ValueError(
                "`nn.Retriver requires `task_inputs` " "or `task_multimodal_inputs`"
            )

        self.set_name(name)
        self._set_retriever(retriever)        
        self._set_model(model)
        self._set_task_inputs(task_inputs)
        self._set_task_multimodal_inputs(task_multimodal_inputs)
        self._set_response_mode(response_mode)
        self._set_response_template(response_template)
        self._set_top_k(top_k)
        self._set_threshold(threshold)
        self._set_return_score(return_score)
        self._set_dict_key(dict_key)

    # TODO: allow imgs
    def forward(self, message: Union[str, List[str], List[Dict[str, Any]], Message]):
        queries = self._prepare_task(message)
        retriever_response = self._execute_retriever(queries)
        response = self._prepare_response(retriever_response, message)
        return response

    def _execute_retriever(self, queries) -> List[Dict[str, Any]]:
        if self.model:
            model_response = self.model(queries)
            queries = model_response.consume()

        retriever_response = self.retriever(
            queries=queries,
            top_k=self.top_k,
            threshold=self.threshold,
            return_score=self.return_score,
        )

        results = []

        for query, query_results in zip(queries, retriever_response):
            formatted_result = {
                "query": query,
                "results": [
                    {"data": item.get("data"), "score": item.get("score")}
                    for item in query_results
                ],
            }
            results.append(formatted_result)

        return results

    def _prepare_response(self, retriever_response, message):
        if self.response_template:
            response = self._format_iterative_response_template(retriever_response)
        else:
            response = retriever_response

        # TODO: mover para module como uma função?
        if self.response_mode == "plain_response":
            return response
        elif isinstance(message, Message):
            if self.response_mode.startswith(("context", "outputs", "response")):
                message.set(f"{self.response_mode}.{self.name}", response)
            return message
        else:
            raise ValueError(
                "For `response_mode` other than `plain_response` "
                "the message object must be of type Message"
            )

    def _prepare_task(
        self, message: Union[str, List[str], List[Dict[str, Any]], Message]
    ) -> List[str]:
        if isinstance(message, str):
            queries = [message]
        elif isinstance(message, list):
            if isinstance(message[0], dict):
                queries = self._process_list_dict_task(message)
            else:
                queries = message
        elif isinstance(message, Message):
            queries = self._process_message_task(message)
        else:
            raise ValueError("Unsupported message type")
        return queries

    def _process_list_dict_task(self, message: List[Dict[str, Any]]) -> List[str]:
        """Useful to generation schemas
        [{'name': 'vilsin'}]
        dict_key='name'
        """
        if isinstance(self.dict_key, str):
            queries = [data[self.dict_key] for data in message]
            return queries
        else:
            raise ValueError(
                "message that contain List[Dict[str, Any]] "
                "require a `dict_key` to select the key for retrieval"
            )

    def _process_message_task(self, message: Message) -> List[str]:
        if self.task_inputs:
            content = self._process_text_inputs(message)
        elif self.task_multimodal_inputs:
            content = self._process_multimodal_inputs(message)
        # Recurssion
        queries = self._prepare_task(content)
        return queries

    def _process_text_inputs(self, message):        
        if isinstance(self.task_inputs, tuple): # OR inputs
            content = self._get_content_from_or_input(self.task_inputs, message)
        else:
            content = message.get(self.task_inputs)

        if content is None:
            raise ValueError(f"No content found in paths: {self.task_inputs}")

        return content

    def _process_multimodal_inputs(self, message: Message) -> List[Dict[str, Any]]:
        content = []
        for image_path in self.task_multimodal_inputs.get("images", []):
            if isinstance(image_path, tuple):
                image_data = self._get_content_from_or_input(image_path, message)
            else:
                image_data = message.get(image_path)
            if image_data:
                image_bytes_io = to_io_object(image_data)
                content.append(image_bytes_io)
        # TODO: add raise se len(content) == 0?
        # another multimodal inputs is not supported yet
        return content

    def _set_retriever(
        self,
        retriever: Union[
            HybridRetriever, LexicalRetriever, SemanticRetriever, VectorDB
        ],
    ):
        if isinstance(
            retriever, (HybridRetriever, LexicalRetriever, SemanticRetriever, VectorDB)
        ):
            self.register_buffer("retriever", retriever)
        else:
            raise TypeError(
                "`retriever` requires `HybridRetriever`, `LexicalRetriever`, "
                f"`SemanticRetriever` or `VectorDB` instance given `{type(retriever)}`"
            )        

    def _set_model(
        self,
        model: Union[
            AudioEmbedderModel, ImageEmbedderModel, TextEmbedderModel, ModelRouter
        ],
    ):
        if (
            isinstance(
                model,
                (
                    AudioEmbedderModel,
                    ImageEmbedderModel,
                    TextEmbedderModel,
                    ModelRouter,
                ),
            )
            or model is None
        ):
            self.register_buffer("model", model)
        else:
            raise TypeError("`model` requires be `AudioEmbedderModel` "
                            "`ImageEmbedderModel`, `TextEmbedderModel, `"
                            f"`ModelRouter` or None given `{type(model)}`")

    def _set_threshold(self, threshold: Optional[float] = 0.0):
        if isinstance(threshold, float):
            if threshold < 0.0:
                raise ValueError(f"`threshold` requires be >= 0.0 given `{threshold}`")
            self.register_buffer("threshold", threshold)
        else:
            raise TypeError(f"`threshold` requires a float given `{type(threshold)}`")

    def _set_return_score(self, return_score: Optional[bool] = False):
        if isinstance(return_score, bool):
            self.register_buffer("return_score", return_score)
        else:
            raise TypeError(f"`threshold` requires a `bool` given `{type(return_score)}`")

    def _set_top_k(self, top_k: Optional[int] = 4):
        if isinstance(top_k, int):
            if top_k <= 0:
                raise ValueError(f"`top_k` requires be >= 1 given `{top_k}`")
            self.register_buffer("top_k", top_k)
        else:
            raise TypeError(f"`top_k` requires a int given `{type(top_k)}`")            

    def _set_dict_key(self, dict_key: str): # TODO: depreciado, ou não? nao, mas pode ter outro nome
        if isinstance(dict_key, str):
            self.register_buffer("dict_key", dict_key)
        else:
            raise TypeError(f"`dict_key` need be a string given `{type(dict_key)}`")

    # TODO: move to module
    def _set_task_inputs(self, inputs: Union[str, Dict[str, str]]):
        self.register_buffer("task_inputs", inputs)

    # TODO: move to module
    def _set_task_multimodal_inputs(self, multimodal_inputs: Union[str, Dict[str, str]]):
        self.register_buffer("task_multimodal_inputs", multimodal_inputs)

    # TODO: move to module
    def _set_response_template(self, response_template: Optional[str] = None):
        if isinstance(response_template, str):           
            self.register_buffer("response_template", response_template)
        else:
            raise TypeError("`response_template` requires a string not empty")

    # TODO: move to module
    def _set_response_mode(self, response_mode: str):
        if response_mode in ["plain_response", "response"] or response_mode.startswith(
            ("context", "outputs")
        ):
            self.register_buffer("response_mode", response_mode)
        else:
            raise ValueError(
                f"`response_mode={response_mode}` is not supported "
                "only `plain_response`, `context`, `outputs` and `response`"
            )
