"""
myAgent 的抽象基类，完成以下功能：
1、对调用OpenAI的聊天API进行了封装；
2、能自动识别类中用OpenAI调用的方法，生成对应的API调用参数；

每次调用模式为：
1. 调用input方法，传入用户输入的内容；
2. 调用set_prompt方法，更新系统消息；
3. 调用create_messages方法，构造调用api所需的消息列表；本函数会根据类的配置，裁剪消息列表；
4. 调用api方法，传入消息列表，调用api；
5. 如果api返回的stop原因为"max_tokens"，则调用create_messages方法，构造调用api所需的消息列表，传入api，继续调用api；
6. 如果api返回的stop原因为"output_length"，则调用create_messages方法，构造调用api所需的消息列表，传入api，继续调用api；

"""
import logging
import time
from typing import List, Dict, Optional, Callable, Iterator

import openai
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from micro_agent.config import MAX_TOKEN_LENGTH, DEFAULT_MODEL, GREEN, RESET, \
    RETRY
from micro_agent.errors import AgentExecToolError
from micro_agent.models import Completion
from micro_agent.utils import generate_function_schema, merge, \
    generate_random_name

load_dotenv()

logger = logging.getLogger(__name__)


def set_tool(method):
    """装饰器，给方法设置一个tool_schema属性，用于存储方法的参数信息。"""
    schema = generate_function_schema(method)
    method.tool_schema = schema
    return method


class MicroAgent:
    completion_model: dict = Completion(messages=[],
                                        model=DEFAULT_MODEL).model_dump()  # 用于存储模型的配置信息

    def __init__(self,
                 *,
                 name: str = "",  # 代理名称,不同的代理最好使用不同的名字
                 user_name="user",  # 用户名称，用于构造消息，使用同一个用户名称有助于openAI更好的理解对话
                 prompt: str = "You are a helpful assistant.",  # 系统消息，也就是提示词
                 model: str = DEFAULT_MODEL,  # 模型名称
                 input_token_ratio: float = 0.9,  # 输入token占总token窗口总长度的比例
                 need_user_input: bool = False,  # 是否需要用户介入对话，默认为不需要
                 max_round: int = 10,
                 # 对话窗口中的最大对话轮数，为0时，表示无限对话，直到用户主动结束对话或超出最大对话窗口长度被裁剪。
                 ):
        """
        初始化方法
        :param name: 代理的名称，用于标识不同的代理，有助于在于openAI对话时保持上下文
        :param user_name: 用户的名称，用于构造消息，使用同一个用户名称有助于openAI更好的理解对话
        :param prompt: 提示词，用于构造消息，也就是系统消息
        :param model: 所使用的模型名称
        :param input_token_ratio: 输入token占总token窗口总长度的比例，默认为0.9。输入token占总token窗口的比例越大，保持上下文的长度就越长，但是也会导致openAI回复的内容变少。
        :param need_user_input: 是否需要用户介入对话，需要向用户提问，以获取额外信息时，设置为True。默认为False。
        :param max_round: 最大对话轮数，默认为10轮。如果为0，则表示无限对话，直到用户主动结束对话或超出最大对话窗口长度被裁剪。1round为用户发起一个问题得到一个回复。如果中间涉及到tool调用，则也算一轮。
        """
        self.completion = Completion(  # 调用api所需的参数对象
            messages=[],
            model=model,
        )
        self.completion_model = self.completion.model_dump()  # 调用api所需的参数字典
        if model not in MAX_TOKEN_LENGTH:
            raise AgentExecToolError(f"不支持的模型名称：{model}")
        logging.info(f"设置使用的模型：{self.model}")
        self.model: str = model  # 设置使用的模型

        self.name: str = generate_random_name() if not name else name  # 代理的名称,如果没传入则随机生成一个
        self.user_name: str = user_name
        self.prompt: str = prompt

        self.history_messages: List[
            Completion.Message | ChatCompletionMessage] = []  # 历史对话消息
        self.max_round_in_message_window: int = max_round  # 消息窗口最大对话轮数,如果为0，则表示无限对话，直到用户主动结束对话或超出最大对话窗口长度被裁剪。不为0，则超过对话轮数的消息会被裁剪
        self.message_window_round_count = 0  # 当前对话窗口对话轮数
        self.message_windows: Dict[str, int] = {
            "head": 0,
            "tail": 0,
        }  # 用于存储当前消息窗口的头尾指针

        self.last_prompt_tokens: int = 0  # 上一次调用api发送的prompt的token数
        self.last_completion_tokens: int = 0  # 上一次调用api一共消耗的token数
        self.last_assistant_response: Optional[
            ChatCompletion] = None  # 上一次调用api返回的结果

        # 设置输入token占总token上限的比例
        if isinstance(input_token_ratio, float) and 0 < input_token_ratio < 1:
            self.input_token_ratio: float = input_token_ratio
        else:
            self.input_token_ratio: float = 0.9
        logging.info(f"设置输入token占总token上限的比例为：{self.input_token_ratio}")

        # 设置输入token的最大长度
        self.max_input_token: int = int(
            MAX_TOKEN_LENGTH[self.model] * self.input_token_ratio)
        self.max_output_token: int = MAX_TOKEN_LENGTH[
                                         self.model] - self.max_input_token

        self.max_retry_times = len(RETRY)

        logging.info(
            f"设置输入token的最大长度为：{self.max_input_token},输出token的最大长度为：{self.max_output_token}")

        # 初始化openAI客户端
        try:
            self.open_ai_client: OpenAI = OpenAI()
            logging.info("连接openAI服务成功！")
        except Exception as e:
            logging.error(f"无法初始化OpenAI客户端！: {e}")
            raise e

        # 读取类中以tool_开头的方法，构造对应的schema
        self.tool_list: List[Dict] = []
        for attr in dir(self):
            if attr.startswith("tool_") and callable(getattr(self, attr)):
                self.tool_list.append(
                    generate_function_schema(getattr(self, attr)))
        logging.info(f"注册类tool: {self.tool_list}")

        # 读取类中被装饰器set_tool修饰的方法，构造对应的schema
        for attr in dir(self):
            if hasattr(getattr(self, attr), "tool_schema"):
                self.tool_list.append(getattr(self, attr).schema)
                print(getattr(self, attr).tool_schema)
        logging.info(f"注册装饰器tool: {self.tool_list}")

        # 如果需要用户介入对话，则添加用户输入工具
        if need_user_input:
            self.tool_list.append(
                generate_function_schema(self._tool_ask_user))
            logging.info("需要用户介入对话！")

        # 将生成的工具列表传入对话completion
        if self.tool_list:
            self.completion.tools = self.tool_list
            logging.info(f"设置对话工具：{self.completion.tools=}")

    def __str__(self):
        return f"""
        MyAgent(name='{self.name}', 
        user_name='{self.user_name}', 
        prompt='{self.prompt}', 
        model='{self.model}', 
        input_token_ratio={self.input_token_ratio}),
        max_input_token={self.max_input_token},
        max_output_token={self.max_output_token},
        tools={self.tool_list},
        messages={self.history_messages},
        message_windows={self.message_windows},
        messages_in_message_windows={self.history_messages[self.message_windows["head"]:self.message_windows["tail"]]},
        last_prompt_tokens={self.last_prompt_tokens},
        last_completion_tokens={self.last_completion_tokens},
        last_assistant_response={self.last_assistant_response}
"""

    # 如果设置的是completion中的属性，则直接设置self.completion中的值
    def __setattr__(self, key, value):
        if key == "completion":
            super().__setattr__(key, value)
        else:
            try:
                if key in self.completion_model:
                    if key in ["response_format",
                               "model"]:  # response_format是一个枚举类型，需要通过setter来设置
                        super(MicroAgent, self).__setattr__(key, value)
                        return
                    setattr(self.completion, key, value)
                    return
            except AttributeError:
                # 如果 self.completion 还未初始化，则正常设置属性
                pass
            super().__setattr__(key, value)

    # 如果获取的是completion中的属性，则直接获取completion中的属性
    def __getattr__(self, item: str):
        if item in self.completion_model:

            # 使用 getattr 来安全地获取属性值
            try:
                return getattr(self.completion, item)
            except AttributeError:
                pass  # 如果需要，可以在这里处理特定逻辑

        # 提供简化版错误信息，并且使用 type(self).__name__ 获取类名
        raise AttributeError(
            f"Attribute '{item}' not found in {type(self).__name__}")

    def __call__(self, input_text: str) -> str:
        """
        我是一个基于gpt的智能程序，我可以根据用户输入的文本，返回对话结果
        :param input_text: 用户输入的文本
        :return: 按照用户要求返回文本或者json格式的对话结果
        """
        try:
            self.user_input(input_text)
            return self.create()
        except Exception as e:
            logging.error(f"对话出现错误: {e}")
            return f"对话出现错误: {e},无法返回对话结果！"

    #########################
    # 以下是设置属性
    #########################

    # 设置response_format属性，用户控制返回的消息格式，是文本还是json
    @property
    def response_format(self):
        return self.completion.response_format.type

    @response_format.setter
    def response_format(self, value: str):
        self.completion.response_format.type = value

    # 设置提示词属性，会同时更新系统消息
    @property
    def prompt(self):
        return self._prompt

    @prompt.setter
    def prompt(self, value):
        self._prompt = value
        self.system_message = self._create_message("system", self._prompt)

    # 设置模型
    @property
    def model(self):
        return self.completion.model

    @model.setter
    def model(self, value):
        if value in MAX_TOKEN_LENGTH:
            self.completion.model = value
        else:
            raise ValueError(f"Invalid model name: {value}.")

    #########################
    # 以下是内部方法
    #########################

    # 与用户交互，默认为通过控制台输入输出，需要重写以实现其他交互方式
    def _tool_ask_user(
            self,
            message: str
    ) -> str:
        """
        用于向用户询问问题。返回用户的回答。

        :param message: 询问的问题。
        :return: 用户的回答。
        """
        if self.stream:  # 如果是stream模式，则不显示提示信息
            return input()
        else:
            return input(f"{self.name}: {message}")

    # 构造消息
    def _create_message(
            self,
            role: str,
            content: str,
            tool_call_id: str = None
    ) -> Completion.Message:
        """
        根据输入的角色和内容构造消息字典。注意本方法只适用于由用户构造的内容。也就是user，
        system和tool。对于assistant的消息，由API返回的结果中提取。

        :param role: 角色类型，可选值为"user", "system", "tool"。
        对于"assistant"角色的消息，由API返回的结果中提取。
        :param content: 消息内容。
        :param tool_call_id: 当role为"tool"时，此参数用于标识调用的是哪个工具。
        对于其他角色，此参数应为None。
        :return: 构造后的消息字典。

        :raises ValueError: 如果role不在预期范围内或者当role为"tool"但没有提供
        `tool_call_id`时抛出。
        """

        # 确保role值有效
        if role not in ["user", "system", "tool"]:
            raise ValueError(
                f"role must be one of ['user', 'system', 'tool'], but got {role}"
            )

        if role == 'system':  # 如果是系统消息，设置名称为类名称
            message = Completion.SystemMessage(
                role=role,
                content=content,
                name=self.name
            )
        elif role == 'user':  # 如果是用户消息，设置名称为用户名称
            message = Completion.UserMessage(
                role=role,
                content=content,
                name=self.user_name
            )
        else:  # 如果是工具消息
            if tool_call_id is None:
                raise ValueError(
                    "When role is 'tool', `tool_call_id` must be provided.")
            message = Completion.ToolMessage(
                role=role,
                content=content,
                tool_call_id=tool_call_id
            )

        return message

    def _call_openai_api(self):

        # 调用openAI接口，最多尝试三次
        attempt = 0
        while attempt < self.max_retry_times:
            try:
                return self.open_ai_client.chat.completions.create(
                    **self.completion.model_dump(exclude_defaults=True))
            # 需要报错并中断
            except (
                    openai.APIConnectionError,
                    openai.BadRequestError,
                    openai.AuthenticationError,
                    openai.NotFoundError,
                    openai.PermissionDeniedError,
            ) as e:
                logging.error(
                    f"Open AI API returned an error! can't continue... {e}")
                raise e
            # 需要稍后重新尝试的错误
            except (
                    openai.APITimeoutError,
                    openai.ConflictError,
                    openai.InternalServerError,
                    openai.RateLimitError,
                    openai.UnprocessableEntityError
            ) as e:
                logging.error(f"OpenAI API returned an API Error: {e}")
                attempt += 1
                if attempt == self.max_retry_times:
                    raise e
                logging.info(f"{RETRY[attempt]}秒后重试第{attempt}次...")
                time.sleep(RETRY[attempt])

    @staticmethod
    def _merge_and_display_stream_chunks(
            trunks: Iterator[ChatCompletionChunk]) -> ChatCompletionChunk:
        # 如果是stream模式，则在屏幕上输出每次返回的消息，最终将返回的消息合并成一个
        logging.info("stream 模式...")
        response = None
        for index, trunk in enumerate(trunks):
            if index == 0:
                response = trunk.model_copy()  # 用于存储返回的结果
                # 确保第一条返回结果为空
                response.choices[0].delta.content = ''
            if trunk.choices[0].delta.content is not None:
                print(
                    f'{GREEN}{trunk.choices[0].delta.content}{RESET}',
                    end='')
            response = merge(response, trunk)  # 将返回的一系列trunk合并成一个
        if response.choices[0].delta.content is not None:
            print()  # 输出换行
        return response

    def _push_message(
            self,
            message: Completion.Message | ChatCompletionMessage
    ) -> None:
        """
        将消息压入消息队列。

        :param message: 消息字典。
        """
        self.history_messages.append(message)
        self.message_windows["tail"] = len(self.history_messages)

    def _assistant_input(
            self,
            message: ChatCompletionMessage
    ) -> None:
        """
        将Assistant的回复压入消息队列。
        :param message: llm返回的信息，直接来自API。
        :return: 无
        """
        self._push_message(message)

    def _tool_input(
            self,
            content: str,
            tool_call_id: str
    ) -> None:
        """
        将llm回复中，指定执行的函数运行产生的回复压入消息队列。
        :param content: 产生的回复
        :param tool_call_id: 指定执行的函数的assistant的id
        :return: 无
        """
        self._push_message(self._create_message('tool', content, tool_call_id))

    def _call_method(
            self,
            method_name: str,
            *args,
            **kwargs
    ) -> str:
        """
        调用指定的方法
        :param method_name: 需要运行函数的名称
        :param args: 需要运行函数的位置参数
        :param kwargs: 需要运行函数的指名参数
        :return: 函数运行的结果
        """
        method = getattr(self, method_name)
        # 调用方法
        try:
            logging.info(f"执行了方法{method_name}({args},{kwargs})")
            response = method(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error calling function: {e}")
            raise AgentExecToolError(f"Error calling function: {e}")
        # 将返回结果加入到历史消息中
        return response

    def _create_messages(
            self
    ) -> List[Completion.Message]:
        """
         返回系统消息和消息窗口中的消息,用于发送给openai
        :return: 用于发送的消息列表
        :todo 研究一下如何获取tools_call相关的内容（包含发给openAI的函数说明以及openAI返回的函数调用参数说明）的token值，可以在这里根据token消耗来裁剪消息。
        """
        return [self.system_message] + self.history_messages[
                                       self.message_windows['head']:
                                       self.message_windows["tail"]]

    ###########################
    # 以下是外部方法
    ###########################
    def register_agent(self, name: str, agent: "MicroAgent"):
        """
        注册一个其他MyAgent类，以便在对话中调用其他agent的服务。
        注意：注册后，还需要在当前agent提示词中维护对应的调用方法。
        :param name: 注册的agent的名称
        :param agent: 其他agent对象
        :return: None
        """

        # 获取agent()的签名
        agent_schema = generate_function_schema(agent.__call__)
        # 修正agent()的签名
        agent_schema['function']['name'] = name
        setattr(self, name, agent.__call__)
        # 将agent()的签名加入到tools列表中
        self.tool_list.append(agent_schema)
        self.completion.tools = self.tool_list

    def register_tool(self, tool: Callable):
        """
        注册工具方法,用于在对话中调用。注意：注册的方法名称不能重复，否则会覆盖。
        :param tool: 工具方法
        :return:
        """
        if callable(tool):
            try:
                self.tool_list.append(generate_function_schema(tool))
            except Exception as e:
                logger.error(f"Error registering tool: {e}")
                return False

        if self.tool_list:
            logger.info(f"Found tools: {self.tool_list}")
            self.completion.tools = self.tool_list

    def trim_history(
            self,
            reset=False
    ) -> None:
        """
        将消息窗口的起点移动到下一个用户消息。缩小消息窗口的范围。如果reset为True，则
        将消息窗口的起点移动到头部。用户更换话题。
        :param reset: 是否重置消息窗口
        :return:
        """

        if reset:  # 重置消息窗口
            self.message_windows["head"] = self.message_windows["tail"]
            return
        for item in range(self.message_windows["head"] + 1,
                          self.message_windows["tail"]):

            if self.history_messages[item].role == "user":
                self.message_windows["head"] = item
                self.message_window_round_count -= 1  # 当前消息窗口对话轮数减一
                break
        else:  # 如果没找到下一个user信息，则说明窗口中已经没有user信息，此时清空整个窗口
            self.message_windows['head'] = self.message_windows['tail']
            self.message_window_round_count = 0

    def user_input(
            self,
            content: str
    ) -> None:
        """
        将用户输入的信息压入消息队列。
        :param content: 用户输入的信息。
        :return: 无
        """
        self._push_message(self._create_message('user', content))

    def create(
            self
    ) -> str:
        """
        从openai处获取返回信息,并对返回的信息进行处理:
        1. 构造要发送给openai的消息
        2. 调用openai的api
        3. 处理返回结果
            1. 如果返回表示的stop的正常返回，则返回结果
            2. 如果返回标志为tool_call,则调用标明的工具，并将调用结果加入到消息队列中，再次调用openai，直到返回stop
            3. 如果返回标志为content_filter，则进行告警
            4. 如果返回标志为length，则说明内容超出了长度限制，填写用户信息"请继续"之后再继续调用openai，将答案拼接在一起，直到返回stop
        :return: 文本格式的openAI返回结果
        """
        # 构造要发送给openai的消息

        total_content = ''

        while True:
            # 创建要发送到openAI的消息
            msg = self._create_messages()
            self.completion.messages = msg

            # 调用openAI接口
            response = self._call_openai_api()

            # 处理返回结果，如果是stream方式，则需要合并生成的消息
            if self.completion.stream:
                response = self._merge_and_display_stream_chunks(response)
            else:
                # 直接获取返回消息
                self.last_prompt_tokens = response.usage.prompt_tokens
                self.last_completion_tokens = response.usage.total_tokens - self.last_prompt_tokens

            # 如果返回的token消耗超过了限制，则裁剪一条历史消息
            # 虽然有可能裁剪后prompt_token数还是超限，但最少腾出了一轮对话的空间。
            # 所以，当你期待llm产生大量回复时，要小心规划prompt_token的比例关系
            # 注意:stream模式下没有返回token消耗信息,且openAI官方给出的token计算代码不可靠,以下代码不起作用
            if hasattr(response,
                       'usage') and response.usage.total_tokens > self.max_input_token:
                self.trim_history(reset=False)

            # 如果设置了最大对话窗口轮次，则根据窗口轮次进行裁剪
            # 注意：如果llm返回的stop_reason为tool，或者说tool调用轮次不受窗口最大窗口轮次影响
            # 也就是说，调用tool发生的交互不单独记为一轮对话
            while response.choices[0].finish_reason != 'tool_calls' and \
                    0 < self.max_round_in_message_window < self.message_window_round_count:
                self.trim_history(reset=False)

            choice = response.choices[0]

            # 将返回的消息压入消息队列
            self.last_assistant_response = choice.delta if self.stream else choice.message
            self._assistant_input(
                self.last_assistant_response)

            # 根据finish_reason分别进行处理
            finish_reason = choice.finish_reason
            if finish_reason == "stop":
                total_content += self.last_assistant_response.content
                self.message_window_round_count += 1
                return total_content
            elif finish_reason == "tool_calls":
                logging.info(
                    f"收到{len(self.last_assistant_response.tool_calls)}个函数调用")
                for _i, tool_call in enumerate(
                        self.last_assistant_response.tool_calls,
                        start=1):
                    logging.info(f"正在处理第{_i}个函数调用")
                    function_call_result = self._call_method(
                        method_name=tool_call.function.name,
                        **eval(tool_call.function.arguments)
                    )
                    # 将返回值加入消息列表，并重新调用api
                    self._tool_input(function_call_result, tool_call.id)
                logging.debug("本地api调用处理完毕，重新调用openAI api..")
            elif response.choices[0].finish_reason == "length":
                total_content += self.last_assistant_response.content
                self.message_window_round_count += 1
                self.user_input("请继续")  # 尝试让openai继续回答
            elif choice.finish_reason == "content_filter":
                logging.warning(
                    f"Warning! content_filter: {self.last_assistant_response.content}")
                self.message_window_round_count += 1
                return total_content + choice.message.content
            else:
                logging.error(f"Error! Unknown finish_reason: ")
                raise ValueError(f"Error! Unknown finish_reason: ")
