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
        content: str
        role: str

    class SystemMessage(Message):
        name: Optional[str] = Field('system', alias='name')

    class UserMessage(Message):
        name: Optional[str] = None

    class ToolMessage(Message):
        tool_call_id: str

    class ResponseMode(BaseModel):
        class Config:
            validate_assignment = True

        type: Literal['json_object', 'text']

    messages: List[
        SystemMessage | UserMessage | ToolMessage | ChatCompletionMessage | ChoiceDelta]
    model: str
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[ResponseMode] = ResponseMode(type='text')
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = Field(1.0, gt=0.0, lt=2.0)
    top_p: Optional[float] = 1.0
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[str] = "auto"
    user: Optional[str] = None


if __name__ == '__main__':
    my_completion = Completion(
        messages=[],
        model='gpt-3.5-turbo',
        type='json_object',
        temperature=1.0
    )
    my_completion.n = 2
    my_completion.temperature = 1.8
    # my_completion.response_format.type = 'json_object'
    my_completion.response_format.type = 'json_object'
    print(my_completion.model_dump(exclude_defaults=True))
