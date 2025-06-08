import base64
import tempfile
from contextlib import contextmanager
from os import getenv
from typing import Any, Dict, List, Literal, Optional, Union

import msgspec
try:
    import httpx
    import openai
    from openai import OpenAI    
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor
except:
    raise ImportError("`openai` client is not detected, please install"
                      "using `pip install msgflow[openai]`")

from msgflow.logger import logger
from msgflow.exceptions import KeyExhaustedError
from msgflow.models.base import BaseModel
from msgflow.models.response import ModelResponse, ModelStreamResponse
from msgflow.models.tool_call_agg import ToolCallAggregator
from msgflow.models.types import (
    ASRModel,
    ChatCompletionModel,
    ImageTextToImageModel,
    ModerationModel,
    TextEmbedderModel,
    TTSModel,
)
from msgflow.nn import functional as F
from msgflow.utils.chat import adapt_struct_schema_to_json_schema
from msgflow.utils.encode import encode_data_to_bytes
from msgflow.utils.msgspec import struct_to_dict
from msgflow.utils.tenacity import model_retry
from msgflow.utils.xml import xml_to_typed_dict


OpenAIInstrumentor().instrument()

# support continuing generation by validating the reason

class _BaseOpenAI(BaseModel):

    provider: str = "openai"    

    def _initialize(self):
        """Initialize the OpenAI client with empty API key."""
        self.current_key_index = 0
        max_retries = getenv("OPENAI_MAX_RETRIES", openai.DEFAULT_MAX_RETRIES)
        timeout = getenv("OPENAI_TIMEOUT", None)
        base_url = self._get_base_url()
        self.client = OpenAI(
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

# TODO: if provider is openai change max_tokens to max_completion_tokens
class OpenAIChatCompletion(_BaseOpenAI, ChatCompletionModel):
    r"""OpenAI Chat Completions

    Args:
        model_id: 
            Model ID in provider.
        modalities:
            Types of output you would like the model to generate.
            Can be: ["text"], ["audio"] or ["text", "audio"].
        audio:
            Audio configurations. Define voice and output format.
        max_tokens:
            An upper bound for the number of tokens that can be 
            generated for a completion, including visible output 
            tokens and reasoning tokens.
        reasoning_effort:
            Constrains effort on reasoning for reasoning models. 
            Currently supported values are low, medium, and high. 
            Reducing reasoning effort can result in faster responses 
            and fewer tokens used on reasoning in a response.
            Can be: "low", "medium" or "high".
        temperature:
            What sampling temperature to use, between 0 and 2. 
            Higher values like 0.8 will make the output more random,
            while lower values like 0.2 will make it more focused and 
            deterministic.
        top_p:
            An alternative to sampling with temperature, called nucleus 
            sampling, where the model considers the results of the tokens 
            with top_p probability mass. So 0.1 means only the tokens 
            comprising the top 10% probability mass are considered.
    """

    def __init__(
        self,
        model_id: str,
        modalities: Optional[List[str]] = ["text"],
        audio: Optional[Dict[str, str]] = None,
        max_tokens: Optional[int] = 512,
        reasoning_effort: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        super().__init__()        
        self.model_id = model_id
        self.sampling_run_params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "modalities": modalities,
            "reasoning_effort": reasoning_effort,
            "audio": audio,
        }
        self._initialize()
        self._get_api_key()

    @model_retry
    def _execute(self, **kwargs):
        if kwargs.get("tool_schemas"):
            kwargs["parallel_tool_calls"] = True
        prefilling = kwargs.pop("prefilling")   
        if prefilling:
            kwargs.get("messages").append(
                {"role": "assistant", "content": prefilling}
            )
        # TODO: adapt params
        model_output = self.client.chat.completions.create(
            model=self.model_id, **kwargs, **self.sampling_run_params,
        )
        return model_output

    def _generate(self, **kwargs):
        response = ModelResponse()
        
        return_reasoning = kwargs.pop("return_reasoning")
        xml_to_dict = kwargs.pop("xml_to_dict")
        generation_schema = kwargs.pop("generation_schema")
        if generation_schema is not None and xml_to_dict is False:
            schema = msgspec.json.schema(generation_schema)
            json_schema = adapt_struct_schema_to_json_schema(schema)
            kwargs["response_format"] = json_schema

        model_output = self._execute_model(**kwargs)

        choice = model_output.choices[0]

        if (
            return_reasoning is True and
            hasattr(choice.message, "reasoning_content") and
            choice.message.reasoning_content is not None
        ):
            reasoning_content = choice.message.reasoning_content
            prefix_response_type = "reasoning_"
        else:
            reasoning_content = None
            prefix_response_type = ""

        if choice.message.tool_calls:
            aggregator = ToolCallAggregator(reasoning_content)
            response.set_response_type("{}tool_call".format(prefix_response_type))
            for call_index, tool_call in enumerate(choice.message.tool_calls):
                id = tool_call.id
                name = tool_call.function.name
                arguments = tool_call.function.arguments
                aggregator.process(call_index, id, name, arguments)
            response.add(aggregator)
        elif choice.message.content:
            if xml_to_dict is True:
                response.set_response_type("{}structured".format(prefix_response_type))
                dict_parsed = xml_to_typed_dict(choice.message.content)
                if generation_schema: # Type validation
                    dict_encoded = msgspec.json.encode(dict_parsed)
                    msgspec.json.decode(dict_encoded, type=generation_schema)
                if reasoning_content is not None: dict_parsed["think"] = reasoning_content
                response.add(dict_parsed)            
            elif generation_schema is not None:
                response.set_response_type("{}structured".format(prefix_response_type))
                struct = msgspec.json.decode(
                    choice.message.content, type=generation_schema
                )
                struct_parsed = struct_to_dict(struct)
                if reasoning_content is not None: struct_parsed["think"] = reasoning_content
                response.add(struct_parsed)
            else:
                response.set_response_type("{}text_generation".format(prefix_response_type))
                content = choice.message.content
                if reasoning_content is not None:
                    response.add({"think": reasoning_content, "answer": content})
                else:
                    response.add(content)
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
        
        return response

    async def _stream_generate(self, **kwargs):
        aggregator = ToolCallAggregator()

        return_reasoning = kwargs.pop("return_reasoning")
        stream_response = kwargs.pop("stream_response")

        model_output = self._execute_model(**kwargs)

        for chunk in model_output:            
            if chunk.choices:
                if (
                    return_reasoning is True and
                    hasattr(chunk.choices[0].delta, "reasoning_content") and
                    chunk.choices[0].delta.reasoning_content is not None
                ):
                    if stream_response.response_type is None:
                        stream_response.set_response_type("reasoning_text_generation")
                        stream_response.first_chunk_event.set()
                    stream_response.add(chunk.choices[0].delta.reasoning_content)
                elif chunk.choices[0].delta.content:
                    if stream_response.response_type is None:
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
        xml_to_dict: Optional[bool] = False,
        return_reasoning: Optional[bool] = False
    ) -> Union[ModelResponse, ModelStreamResponse]:
        """
        Args:
            messages: 
                Conversation history. Can be simple string or list of messages.
            system_prompt:
                A set of instructions that defines the overarching behavior and role of the model across all interactions.
            prefilling:
                Forces an initial message from the model. From that message it will continue its response from there.
            stream:
                Whether generation should be in streaming mode.
            generation_schema:
                Schema that defines how the output should be structured.
            tool_schemas:
                JSON schema containing available tools.
            tool_choice:
                By default the model will determine when and how many tools to use. 
                You can force specific behavior with the tool_choice parameter.
                    1. Auto: 
                        (Default) Call zero, one, or multiple functions. tool_choice: "auto"
                    2. Required: 
                        Call one or more functions. tool_choice: "required"
                    3. Forced Function: 
                        Call exactly one specific function. tool_choice: {"type": "function", "function": {"name": "get_weather"}}    
            xml_to_dict:
                Converts the model output, which should be typed-XML, into a typed-dict.
            return_reasoning:
                If the model returns the `reasoning` field it will be added along with the response.
                TODO: falar como fica o formato
        """        
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if isinstance(system_prompt, str):
            messages.insert(0, {"role": "system", "content": system_prompt})

        generation_params = dict(
            messages=messages,
            prefilling=prefilling,
            generation_schema=generation_schema,
            tool_choice=tool_choice,
            tools=tool_schemas,
            return_reasoning=return_reasoning,
        )

        if stream is True:
            if generation_schema is not None:
                raise ValueError("`generation_schema` is not `stream=True` compatible")

            if xml_to_dict is True:
                raise ValueError("`xml_to_dict=True` is not `stream=True` compatible")
            
            if return_reasoning is True and tool_schemas is not None:
                raise ValueError("`tool_schemas` is not `return_reasoning=True` compatible"
                                 " when `stream=True`")

            stream_response = ModelStreamResponse()
            F.background_task(
                self._stream_generate,
                **generation_params,
                stream=stream,
                stream_response=stream_response,
                stream_options={"include_usage": True},
            )
            F.wait_for_event(stream_response.first_chunk_event)
            return stream_response
        else:
            response = self._generate(**generation_params)
            return response


class OpenAITTS(_BaseOpenAI, TTSModel):
    r"""OpenAI Text-to-Speech
    
        alloy
        ash
        ballad
        coral
        echo
        fable
        onyx
        nova
        sage
        shimmer    
    """

    def __init__(
        self,
        model_id: str,
        voice: Optional[
            Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        ] = "alloy",
        speed: Optional[float] = 1.0,
    ):
        super().__init__()
        self.model_id = model_id
        self.sampling_run_params = {
            "voice": voice,
            "speed": speed,
        }
        self._initialize()        
        self._get_api_key()

    @contextmanager
    def _execute_with_retry(self, **kwargs):
        while True:
            try:
                with self._execute(**kwargs) as result:
                    yield result
                break
            except (openai.RateLimitError, openai.APIError) as e:
                print(e) # TODO
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
    @model_retry
    def _execute(self, **kwargs):
        with self.client.audio.speech.with_streaming_response.create(
            model=self.model_id, **kwargs, **self.sampling_run_params
        ) as model_output:
            yield model_output

    def _generate(self, **kwargs):
        response = ModelResponse()

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
        stream_response.set_response_type("audio_generation")

        with self._execute_model(**kwargs) as model_output:
            for chunk in model_output.iter_bytes(chunk_size=1024):
                stream_response.add(chunk)
                if not stream_response.first_chunk_event.is_set():
                    stream_response.first_chunk_event.set()

        stream_response.add(None)

    def __call__(
        self, 
        data: str, 
        *, 
        stream: Optional[bool] = False, 
        prompt: Optional[str] = None,
        response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = "opus",
    ) -> Union[ModelResponse, ModelStreamResponse]:
        params = {"input": data, "response_format": response_format}
        if prompt:
            params["instructions"] = prompt
        if stream:
            stream_response = ModelStreamResponse()
            params["stream_response"] = stream_response
            F.background_task(self._stream_generate, **params)
            F.wait_for_event(stream_response.first_chunk_event)          
            return stream_response
        else:
            response = self._generate(**params)
            return response


class OpenAIImageTextToImage(_BaseOpenAI, ImageTextToImageModel):
    """OpenAI Image Generation"""

    def __init__(
        self,
        *,
        model_id: str,
        size: Optional[
            Literal["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"]
        ] = "1024x1024",
        quality: Optional[Literal["standard", "hd"]] = "hd",
    ):
        super().__init__()
        self.model_id = model_id
        self.sampling_run_params = {"size": size, "quality": quality}
        self._initialize()
        self._get_api_key()

    @model_retry
    def _execute(self, **kwargs):
        if kwargs.get("image"):
            model_output = self.client.images.edit(
                model=self.model_id, **kwargs, **self.sampling_run_params
            )
        else:
            model_output = self.client.images.generate(
                model=self.model_id, **kwargs, **self.sampling_run_params
            )
        return model_output

    def _generate(self, **kwargs):
        response = ModelResponse()

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
            inputs["image"] = encode_data_to_bytes(image) # TODO: validate
        if mask:
            inputs["mask"] = encode_data_to_bytes(mask)
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
        model_id: str,
        temperature: Optional[float] = 0.0,
    ):
        super().__init__()        
        self.model_id = model_id
        self.sampling_run_params = {"temperature": temperature}
        self._initialize()        
        self._get_api_key()

    @model_retry
    def _execute(self, **kwargs):
        model_output = self.client.audio.transcriptions.create(
            model=self.model_id, **kwargs, **self.sampling_run_params
        )
        return model_output

    def _generate(self, **kwargs):
        response = ModelResponse()

        model_output = self._execute_model(**kwargs)

        response.set_response_type("transcript")

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

    def _stream_generate(self, **kwargs):
        stream_response = kwargs.pop("stream_response")
        stream_response.set_response_type("transcript")

        model_output = self._execute_model(**kwargs)

        for event in model_output:
            chunk = event.transcript.text.delta
            if chunk:
                stream_response.add(chunk)
                if not stream_response.first_chunk_event.is_set():
                    stream_response.first_chunk_event.set()
            elif event.transcript.text.done:
                stream_response.add(None)
                
        return stream_response

    def __call__(
        self,
        data: bytes,
        *,
        stream: Optional[bool] = False,
        response_format: Optional[
            Literal["json", "text", "srt", "verbose_json", "vtt"]
        ] = "text",
        timestamp_granularities: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        language: Optional[str] = None,        
    ):
        params = {
            "file": data,
            "language": language,
            "response_format": response_format,
            "timestamp_granularities": timestamp_granularities,
            "prompt": prompt,
        }
        if stream:
            stream_response = ModelStreamResponse()
            params["stream_response"] = stream_response
            params["stream"] = stream
            F.background_task(self._stream_generate, **params)
            F.wait_for_event(stream_response.first_chunk_event)
            return stream_response
        else:                
            response = self._generate(**params)
            return response


class OpenAITextEmbedder(_BaseOpenAI, TextEmbedderModel): 
    # TODO allow slice embedding
    def __init__(
        self,
        *,
        model_id: str,
    ):
        super().__init__()        
        self.model_id = model_id
        self._initialize()        
        self._get_api_key()

    @model_retry
    def _execute(self, **kwargs):
        model_output = self.client.embeddings.create(
            model=self.model_id, **kwargs,
        )
        return model_output

    def _generate(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("text_embedding")
        model_output = self._execute_model(**kwargs)
        embedding = model_output.data[0].embedding
        response.add(embedding)
        return response

    def __call__(
        self,
        data: str,
    ):
        response = self._generate(text=data)
        return response


class OpenAIModeration(_BaseOpenAI, ModerationModel): 

    def __init__(
        self,
        *,
        model_id: str,
    ):
        super().__init__()        
        self.model_id = model_id
        self._get_api_key()
        self._initialize()

    @model_retry
    def _execute(self, **kwargs):
        model_output = self.client.moderations.create(
            model=self.model_id, **kwargs,
        )
        return model_output

    def _generate(self, **kwargs):
        response = ModelResponse()
        response.set_response_type("moderation")
        model_output = self._execute_model(**kwargs)
        moderation = model_output.results
        moderation["safe"] = moderation["flagged"]
        response.add(moderation)
        return response

    def __call__(
        self,
        data: Union[str, List[Dict[str, Any]]],
    ):
        response = self._generate(input=data)
        return response
