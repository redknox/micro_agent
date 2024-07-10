"""本模块定义了用于生成对话完成的请求的数据模型。"""
from typing import List, Dict, Literal, Optional

from openai.types.chat.chat_completion_chunk import \
    ChoiceDelta  # stream模式下返回的消息
from openai.types.chat.chat_completion_message import \
    ChatCompletionMessage  # 问答模式下返回的消息
from pydantic import BaseModel, Field


class Completion(BaseModel):
    class Config:  # 配置类
        validate_assignment = True

    class Message(BaseModel):
        __abstract__ = True
        role: Literal['system', 'user', 'tool', 'assistant']

    class UserMessage(Message):
        class TextContent(BaseModel):
            text: str
            type: Literal['text']

        class ImageContent(BaseModel):
            # todo: 使用ImageContent时，只能使用gpt-4-visual-preview等特定模型才可以
            class ImageUrl(BaseModel):
                url: str
                detail: Literal['auto', 'high', 'low']

            image_url: ImageUrl
            type: Literal['image_url']  # image_url

        content: str | List[
            TextContent | ImageContent]
        name: Optional[str] = Field('user', alias='name')

    class SystemMessage(Message):
        content: str
        name: Optional[str] = Field('system', alias='name')

    class ToolMessage(Message):
        content: str
        tool_call_id: str

    class AssistantMessage(Message):
        class ToolCall(BaseModel):
            class Function(BaseModel):
                name: str
                arguments: str  # json字符串

            id: str
            type: Literal['function']
            function: Function

        content: Optional[str] = None  # 除非指定了tool_call_id，否则content必须
        name: Optional[str] = Field('assistant', alias='name')
        tool_calls: Optional[List[ToolCall]]

    class ResponseMode(BaseModel):
        class Config:
            validate_assignment = True

        type: Literal['json_object', 'text']

    class StreamOptions(BaseModel):
        include_usage: Optional[bool] = False

    messages: List[
        SystemMessage | UserMessage | ToolMessage | ChatCompletionMessage | ChoiceDelta | AssistantMessage]
    model: str
    frequency_penalty: Optional[float] = Field(default=0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20)
    max_tokens: Optional[int] = None
    n: Optional[int] = Field(default=1)
    presence_penalty: Optional[float] = Field(default=0, ge=-2.0, le=2.0)
    response_format: Optional[ResponseMode] = ResponseMode(type='text')
    seed: Optional[int] = None
    service_tier: Optional[Literal['auto', 'default']] = None
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = StreamOptions(include_usage=False)
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Literal['none', 'auto', 'required']] = "auto"
    parallel_tool_calls: Optional[bool] = True
    user: Optional[str] = None


class Tool(BaseModel):
    class Function(BaseModel):
        class Parameter(BaseModel):
            name: str
            type: Literal['str', 'int', 'float', 'bool', 'json']

        description: Optional[str] = None
        name: str
        parameters: Optional[Parameter] = None

    type: Literal['function'] = Field(default='function')
    function: Function
