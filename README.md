# MICRO_AGENT

micro-agent 是一个轻量级的使用openAI模型服务的智能代理，将程序调用openAI API的过程进行了封装，使得用户可以调用函数一样与大模型进行对话。同时，micro-agent
包含了简单的会话上下文窗口管理，使得发送给大模型的对话历史可以根据token数量进行调整。同时，micro-agent
支持将本地函数注册到代理中，使得代理可以调用本地函数，使用最简单的方式构造agent。

## 安装

使用pip安装：

```shell
pip install wee_agent
```

## 使用

### 1. 使用前

使用前，应该将 OPEN_API_KEY 配置到环境变量中：

    ```shell
    export OPEN_API_KEY=your_open_api_key
    ```

或者，在代码的根目录中，创建一个`.env`文件，写入：

    ```shell
    OPEN_API_KEY=your_open_api_key
    ```

### 2. 基础使用

只需实例化一个代理，然后可以像调用函数一样直接调用openAI API的对话功能。

```python
from wee_agent import WeeAgent

agent = WeeAgent()
ret = agent("hello")
print(ret)
```

要实现多轮对话，只需要重复调用即可：

```python
from wee_agent import WeeAgent

agent = WeeAgent()
print(agent('hi，I am Bob. '))
# 显示：Hello, Bob! How can I assist you today?
print(agent('What is my name?'))
# 显示：Your name is Bob. How can I help you today, Bob?
```

代理会记录与用户沟通的历史对话，并能自动根据token数量调整发送给llm的对话历史窗口。

### 添加tool

micro_agent可以非常方便的将本地函数注册到agent中，使得micro_agent对象成为一个智能体。
注册的函数必须满足以下要求：

* 注册的函数，必须进行类型注释，指定参数的类型
* 函数的返回值，必须为文本格式，以便于大模型理解。目前不支持使用其他格式。

先使用set_tool装饰器，装饰本地函数，然后调用register_tool方法注册函数，最后调用agent对象，传入提示词，即可调用本地函数。

```python
from wee_agent import WeeAgent, set_tool


# set_tool是一个装饰器，用于注册函数
# 定义本地函数
@set_tool
def plus(a: int, b: int) -> str:
   """
    计算两个数的和
    :param a: 第一个数
    :param b: 第二个数
    :return: 两个数的和
    """
   return str(a + b)


agent = WeeAgent()

# 注册函数
agent.register_tool(name='plus', tool=plus)

# 调用函数
print(agent("计算1+2等于多少"))

# 输出：1+2等于3
```

### 3. 高级用法

#### 3.1 自定义提示词

在实例化agent时，可以传入写好的提示词作为参数：

```python
from wee_agent import WeeAgent

agent = WeeAgent(prompt="你是一名心里咨询师，善于解决心里问题...")
print(agent("我感到很烦躁"))
```

或者：

```python
from wee_agent import WeeAgent

agent = WeeAgent()
agent.set_prompts("你是一名心里咨询师，善于解决心里问题...")
print(agent("我感到很烦躁"))
```

#### 3.2 流式输出
在初始化类时，可以设置streaming参数为True，使得输出更加流畅：

```python
from wee_agent import WeeAgent

agent = WeeAgent(stream=True)

agent("hello")

# 在终端上逐token输出：Hello! How can I assist you today?
```



#### 3.3 控制对话窗口问答比例
micro_agent可以控制每次问答时，发送给大模型的对话历史占整个对话历史的比例。默认为0.9，即每次问答时，发送给大模型的对话历史占整个对话历史的90%。对话历史里包含了prompt。如果你需要大模型回答更多内容，可以将这个比例调低。同时，这也会导致对话历史信息降低。

```python
from wee_agent import WeeAgent

agent = WeeAgent(
   input_token_ratio=0.5  # 提问和回答各占对话窗口的50%
)
...
```
#### 3.4 重构本类
为了更加方便的使用，可以继承MicroAgent类，然后使用装饰器注册函数：

```python
from wee_agent import WeeAgent, set_tool
from datetime import datetime


class MyAgent(WeeAgent):

   @staticmethod
   @set_tool
   def get_time():
      """
      获取当前时间
      :return: 字符串格式的当前时间
      """
      return f"现在是{datetime.now().strftime('%Y年%m月%d日%H时%M分%S秒')}"


agent = MyAgent()
print(agent("请告诉我现在的时间"))

# 输出：现在是2024年3月19日18时39分24秒。如果您需要更准确的时间，请让我知道，我可以为您提供更具体的时分秒信息。
```

请注意，注册的函数，必须满足以下要求：

1. 注册的函数，必须进行类型注释，指定参数和返回值的类型，
2. 同时必须提供详细的，严格格式的函数文档，用于生成函数的schema信息，供大模型理解。其中：第一行必须为函数的功能说明。参数的说明，
    * 必须以:param 开头，
    * 返回值必须使用:return 开头。
3. 函数的返回值必须为文本格式，以便于大模型理解。目前不支持使用其他格式。

----

## 下一步计划

因为目前openAI的api格式几乎成为了大模型调用的标杆，修改本模块，让它可以支持其他开源模型，例如Ollama、vllm等。