import base64
import tempfile
from contextlib import contextmanager
from os import getenv
from typing import Any, Dict, List, Literal, Optional, Union
import gevent
import msgspec
try:
    import httpx
    import openai
    from openai import OpenAI
except:
    raise ImportError("`openai` client is not detected, please install"
                      "using `pip install msgflow[openai]`")

from msgflow.logger import logger
from msgflow.exceptions import KeyExhaustedError
from msgflow.models.base import BaseModel
from msgflow.telemetry.events import EventsTiming
from msgflow.models.response import Response, StreamResponse
from msgflow.models.tool_call_agg import ToolCallAggregator
from msgflow.models.types import (
    ASRModel,
    ChatCompletionModel,
    ImageTextToImageModel,
    TextEmbedderModel,
    TTSModel,
)
from msgflow.utils.chat import adapt_struct_schema_to_json_schema
from msgflow.utils.msgspec import struct_to_dict

# TODO: from response to modelresponse IF you want to add more types of output like this
# what may be necessary for greater tracing coverage

# log token usage
# support continuing generation by validating the reason

# TODO split sampling params into 2, normal and run


"""

response.headers.get('x-ratelimit-limit-tokens')
x-ratelimit-limit-rA exceção AllModelsFailedError recebe uma lista de exceções (exceptions) e uma lista de informações sobre os modelos (model_info).

A mensagem de erro é construída para incluir detalhes sobre cada modelo que falhou, incluindo o model_id, provider e a mensagem de erro associada.equests
x-ratelimit-limit-tokens
x-ratelimit-remaining-requests
x-ratelimit-remaining-tokens
x-ratelimit-reset-requests
x-ratelimit-reset-tokens
"""


class _BaseOpenAI(BaseModel):

    provider: str = "openai"
    _api_key: List[str] = []
    current_key_index: int = 0

    def _initialize_client(self):
        """Initialize the OpenAI client with empty API key."""
        max_retries = getenv("OPENAI_MAX_RETRIES", openai.DEFAULT_MAX_RETRIES)
        timeout = getenv("OPENAI_TIMEOUT", None)
        base_url = self._get_base_url()
        self.client = OpenAI(
            **self.sampling_params,
            base_url=base_url,
            api_key="",
            timeout=timeout,
            max_retries=max_retries,
            http_client=httpx.Client(
                limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
            ),
        )

    def _get_base_url(self):
        return None

    def _get_api_key(self):
        """Load API keys from environment variable."""
        keys = getenv("OPENAI_API_KEY")
        if not keys:
            raise ValueError(
                "The OpenAI key is not available. Please set `OPENAI_API_KEY`"
            )
        self._api_key = [key.strip() for key in keys.split(",")]
        if not self._api_key:
            raise ValueError("No valid API keys found")

    def _set_next_api_key(self):
        """Set the next API key in the rotation."""
        if self.current_key_index >= len(self._api_key) - 1:
            raise KeyExhaustedError("All API keys have been exhausted")
        self.current_key_index += 1
        self.client.api_key = self._api_key[self.current_key_index]

    def _execute_with_retry(self, **kwargs):
        """Execute the model with the current API key and handle retries."""
        try:
            return self._execute(**kwargs)
        except (openai.RateLimitError, openai.APIError) as e:
            print(e)
            # Try the next API key
            self._set_next_api_key()
            # Recursively try again with the new key
            return self._execute_with_retry(**kwargs)
        except Exception as e:
            # For other exceptions, we might want to retry with the same key
            raise e

    def _execute_model(self, **kwargs):
        """Main method to execute the model with automatic key rotation and retries."""
        # Set the initial API key
        self.client.api_key = self._api_key[self.current_key_index]

        try:
            return self._execute_with_retry(**kwargs)
        except KeyExhaustedError as e:
            # Reset the key index for future calls
            self.current_key_index = 0
            raise e


class OpenAIChatCompletion(_BaseOpenAI, ChatCompletionModel):
    r"""OpenAI Chat Completions

    Args:
        model_id: Optional[str] = "gpt-4o-mini",
        modalities: Optional[List[str]] = None,
        audio: Optional[Dict[str, str]] = None,
        max_tokens: Optional[int] = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        function_choice: Optional[Literal["auto", "required"]] = "auto",
        parallel_tool_calls: Optional[bool] = True,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,

    voice: Optional[Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]] = "alloy",
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "wav"}
    """

    def __init__(
        self,
        model_id: Optional[str] = "gpt-4o-mini",
        modalities: Optional[List[str]] = ["text"],
        audio: Optional[Dict[str, str]] = None,
        max_tokens: Optional[int] = 512,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
    ):
        super().__init__()        
        self.model_id = model_id
        self.sampling_params = {"organization": organization, "project": project}
        self.sampling_run_params = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "modalities": modalities,
            "audio": audio,
        }
        self._initialize_client()
        self._get_api_key()

    def _log_tokens_usage(self, usage): # deprecated
        """
        "usage": {
            "prompt_tokens": 2006,
            "completion_tokens": 300,
            "total_tokens": 2306,
            "prompt_tokens_details": {
                "cached_tokens": 1920
            },
            "completion_tokens_details": {
                "reasoning_tokens": 0
            }
        }
        """
        # TODO
        usage.completion_tokens
        usage.prompt_tokens
        usage.total_tokens
        usage.completion_tokens_details
        usage.completion_tokens_details.audio_tokens
        usage.completion_tokens_details.reasoning_tokens
        usage.prompt_tokens_details
        usage.prompt_tokens_details.audio_tokens
        usage.prompt_tokens_details.cached_tokens
        # print(usage)

    def _execute(self, **kwargs):
        if kwargs.get("tool_schemas"):
            kwargs["parallel_tool_calls"] = True
        prefilling = kwargs.pop("prefilling")   
        if prefilling:
            kwargs.get("messages").append(
                {"role": "assistant", "content": prefilling}
            )
        model_output = self.client.chat.completions.create(
            **kwargs, **self.sampling_run_params
        )
        return model_output

    def _generate(self, **kwargs):
        response = Response()
        metadata = {}
        events_timing = EventsTiming()
        events_timing.start("model_execution")
        
        generation_schema = kwargs.pop("generation_schema")
        if generation_schema:
            schema = msgspec.json.schema(generation_schema)
            json_schema = adapt_struct_schema_to_json_schema(schema)
            kwargs["response_format"] = json_schema

        events_timing.start("model_generation")

        model_output = self._execute_model(**kwargs)

        events_timing.end("model_generation")

        choice = model_output.choices[0]

        if choice.message.tool_calls:
            aggregator = ToolCallAggregator()
            response.set_response_type("tool_call")
            for call_index, tool_call in enumerate(choice.message.tool_calls):
                id = tool_call.id
                name = tool_call.function.name
                arguments = tool_call.function.arguments
                aggregator.process(call_index, id, name, arguments)
            response.add(aggregator)
        elif choice.message.content:
            if generation_schema:
                response.set_response_type("structured")
                #print(choice.message.content)
                struct = msgspec.json.decode(
                    choice.message.content, type=generation_schema
                )
                struct_parsed = struct_to_dict(struct)
                response.add(struct_parsed)
            else:
                response.set_response_type("text_generation")
                response.add(choice.message.content)
        elif choice.message.audio:
            # To multi turn conversation is necessary persist the audio id
            # https://platform.openai.com/docs/guides/audio#multi-turn-conversations
            audio_response = {
                "id": choice.message.audio.id,
                "audio": base64.b64decode(choice.message.audio.data),
            }
            if choice.message.audio.transcript:
                response.set_response_type("audio_text_generation")
                audio_response["text"] = choice.message.audio.transcript
            else:
                response.set_response_type("audio_generation")
            response.add(audio_response)

        events_timing.end("model_execution")

        if model_output.usage:
            metadata["token_usage"] = model_output.usage.dict()
            #self._log_tokens_usage(model_output.usage) deprecated :)

        metadata["timing"] = events_timing.get_events()
        
        model_info = self.get_model_info()
        model_info["stream"] = "false"
        metadata["model_info"] = model_info

        response.set_metadata(metadata)
        
        return response

    def _stream_generate(self, **kwargs):
        metadata = {}
        events_timing = EventsTiming()
        events_timing.start("model_execution")

        aggregator = ToolCallAggregator()
        stream_response = kwargs.pop("stream_response")
        generation_schema = kwargs.pop("generation_schema")

        if generation_schema:
            schema = msgspec.json.schema(generation_schema)
            kwargs["response_format"] = adapt_struct_schema_to_json_schema(schema)

        events_timing.start("model_generation")

        model_output = self._execute_model(**kwargs)        

        for chunk in model_output:
            if chunk.choices:
                if chunk.choices[0].delta.content:
                    if stream_response.response_type is None:
                        if generation_schema:
                            stream_response.set_response_type("structured")
                        else:
                            stream_response.set_response_type("text_generation")
                        stream_response.first_chunk_event.set()
                    stream_response.add(chunk.choices[0].delta.content)
                elif chunk.choices[0].delta.tool_calls:
                    if stream_response.response_type is None:
                        stream_response.set_response_type("tool_call")
                    tool_call = chunk.choices[0].delta.tool_calls[0]
                    call_index = tool_call.index
                    id = tool_call.id
                    name = tool_call.function.name
                    arguments = tool_call.function.arguments
                    aggregator.process(call_index, id, name, arguments)
            elif chunk.usage:  # TODO: revisar logs em streaming
                metadata["tokens_usage"] = chunk.usage.dict()
                #print(chunk.usage)

        events_timing.end("model_generation")
        events_timing.end("model_execution")

        metadata["timing"] = events_timing.get_events()

        model_info = self.get_model_info()
        model_info["stream"] = "true"
        metadata["model_info"] = model_info        
        
        stream_response.set_metadata(metadata)

        if aggregator.tool_calls:
            stream_response.add(aggregator)
            stream_response.first_chunk_event.set()

        stream_response.add(None)

    def __call__(
        self,
        messages: Union[str, Dict[str, Any]],
        *,
        system_prompt: Optional[str] = None,
        prefilling: Optional[str] = None,
        stream: Optional[bool] = False,
        generation_schema: Optional[msgspec.Struct] = None,
        tool_schemas: Optional[Dict] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Union[Response, StreamResponse]:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if isinstance(system_prompt, str):
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        if stream:
            stream_response = StreamResponse()
            gevent.spawn(
                self._stream_generate,
                messages=messages,
                prefilling=prefilling,
                stream=stream,
                stream_response=stream_response,
                stream_options={"include_usage": True},
                generation_schema=generation_schema,
                tools=tool_schemas,
                tool_choice=tool_choice,
            )
            stream_response.first_chunk_event.wait()
            return stream_response
        else:
            response = self._generate(
                messages=messages,
                prefilling=prefilling,
                generation_schema=generation_schema,
                tool_choice=tool_choice,
                tools=tool_schemas,
            )
            return response


class OpenAITTS(_BaseOpenAI, TTSModel):
    r"""OpenAI Text-to-Speech"""

    def __init__(
        self,
        model_id: Optional[str] = "tts-1",
        voice: Optional[
            Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        ] = "alloy",
        response_format: Optional[
            Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
        ] = "wav",
        speed: Optional[float] = 1.0,
        organization: Optional[str] = None,
        project: Optional[str] = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"organization": organization, "project": project}
        self.sampling_run_params = {
            "model": model_id,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }
        self._initialize_client()        
        self._get_api_key()

    @contextmanager
    def _execute_with_retry(self, **kwargs):
        while True:
            try:
                with self._execute(**kwargs) as result:
                    yield result
                break
            except (openai.RateLimitError, openai.APIError) as e:
                print(e)
                self._set_next_api_key()
            except Exception as e:
                raise e

    @contextmanager
    def _execute_model(self, **kwargs):
        self.client.api_key = self._api_key[self.current_key_index]
        try:
            with self._execute_with_retry(**kwargs) as result:
                yield result
        except KeyExhaustedError as e:
            self.current_key_index = 0
            raise e

    @contextmanager
    def _execute(self, **kwargs):
        with self.client.audio.speech.with_streaming_response.create(
            **kwargs, **self.sampling_run_params
        ) as model_output:
            yield model_output

    def _generate(self, **kwargs):
        response = Response()

        model_output = self._execute_model(**kwargs)

        with tempfile.NamedTemporaryFile(
            suffix=f".{self.response_format}", delete=False
        ) as temp_file:
            temp_file_path = temp_file.name
            model_output.stream_to_file(temp_file_path)

        response.set_response_type("audio_generation")
        response.add({"audio_path": temp_file_path})

        return response

    def _stream_generate(self, **kwargs):
        stream_response = kwargs.pop("stream_response")
        stream_response.response_type = "audio_generation"

        with self._execute_model(**kwargs) as model_output:
            for chunk in model_output.iter_bytes(chunk_size=1024):
                stream_response.add(chunk)
                if not stream_response.first_chunk_event.is_set():
                    stream_response.first_chunk_event.set()

        stream_response.add(None)

    def __call__(
        self, message: str, *, stream: Optional[bool] = False
    ) -> Union[Response, StreamResponse]:
        if stream:
            stream_response = StreamResponse()
            gevent.spawn(
                self._stream_generate, input=message, stream_response=stream_response
            )
            stream_response.first_chunk_event.wait()
            return stream_response
        else:
            response = self._generate(input=message)
            return response


class OpenAIImageTextToImage(_BaseOpenAI, ImageTextToImageModel):
    r"""OpenAI Image Generation"""

    def __init__(
        self,
        *,
        model_id: Optional[str] = "dall-e-2",
        size: Optional[
            Literal["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"]
        ] = "1024x1024",
        quality: Optional[Literal["standard", "hd"]] = "hd",
        organization: Optional[str] = None,
        project: Optional[str] = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.sampling_params = {"organization": organization, "project": project}        
        self.sampling_run_params = {"model": model_id, "size": size, "quality": quality}
        self._initialize_client()        
        self._get_api_key()

    def _execute(self, **kwargs):
        if kwargs.get("image"):
            model_output = self.client.images.edit(**kwargs, **self.sampling_run_params)
        else:
            model_output = self.client.images.generate(**kwargs, **self.sampling_run_params)
        return model_output

    def _generate(self, **kwargs):
        response = Response()

        model_output = self._execute_model(**kwargs)

        response.set_response_type("image_generation")

        if model_output.data[0].url:
            response.add(model_output.data[0].url)
        elif model_output.data[0].b64_json:
            response.add(model_output.data[0].b64_json)

        return response

    def _prepare_inputs(image, mask, response_format):
        inputs = {}
        inputs["inputs"] = "b64_json" if response_format else response_format
        if image:
            inputs["image"] = open(image, "rb")
        if mask:
            inputs["mask"] = open(mask, "rb")
        return inputs

    def __call__(
        self,
        prompt: str,
        *,
        image: Optional[str] = None,
        mask: Optional[str] = None,
        response_format: Optional[Literal["url", "base64"]] = "base64",
    ):
        inputs = self._prepare_inputs(image, mask, response_format)
        response = self._generate(prompt, **inputs)
        return response


class OpenAIASR(_BaseOpenAI, ASRModel):
    def __init__(
        self,
        *,
        model_id: Optional[str] = "whisper-1",
        temperature: Optional[float] = 0.0,
        organization: Optional[str] = None,
        project: Optional[str] = None,
    ):
        super().__init__()        
        self.model_id = model_id
        self.sampling_params = {"organization": organization, "project": project}        
        self.sampling_run_params = {"model": model_id, "temperature": temperature}
        self._initialize_client()        
        self._get_api_key()

    def _execute(self, **kwargs):
        model_output = self.client.audio.transcriptions.create(
            **kwargs, **self.sampling_run_params
        )
        return model_output

    def _generate(self, **kwargs):
        response = Response()

        model_output = self._execute_model(**kwargs)

        response.set_response_type("transcript")

        # TODO: log duration?

        transcript = {}

        if isinstance(model_output, str):
            transcript["text"] = model_output
        else:
            if model_output.text:
                transcript["text"] = model_output.text
            if model_output.words:
                words = [
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in model_output.words
                ]
                transcript["words"] = words
            if model_output.segment:
                segments = [
                    {
                        "id": seg.id,
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                    }
                    for seg in model_output.segments
                ]
                transcript["segments"] = segments

        response.add(transcript)

        return response

    def __call__(
        self,
        audio: bytes,
        *,
        language: Optional[str] = None,
        response_format: Optional[
            Literal["json", "text", "srt", "verbose_json", "vtt"]
        ] = "text",
        timestamp_granularities: Optional[List[str]] = None,
        prompt: Optional[str] = None,
    ):
        response = self._generate(
            file=audio,
            language=language,
            response_format=response_format,
            timestamp_granularities=timestamp_granularities,
            prompt=prompt,
        )
        return response


class OpenAITextEmbedder(_BaseOpenAI, TextEmbedderModel): 
    # TODO allow slice embedding
    def __init__(
        self,
        *,
        model_id: Optional[str] = "text-embedding-3-small",
        organization: Optional[str] = None,
        project: Optional[str] = None,
    ):
        super().__init__()        
        self.model_id = model_id
        self.sampling_params = {"organization": organization, "project": project}        
        self.sampling_run_params = {"model": model_id}
        self._initialize_client()        
        self._get_api_key()

    def _execute(self, **kwargs):
        model_output = self.client.embeddings.create(
            **kwargs, **self.sampling_run_params
        )
        return model_output

    def _generate(self, **kwargs):
        response = Response()
        response.set_response_type("text_embedding")
        model_output = self._execute_model(**kwargs)
        embedding = model_output.data[0].embedding
        response.add(embedding)
        return response

    def __call__(
        self,
        text: str,
    ):
        response = self._generate(text=text)
        return response
