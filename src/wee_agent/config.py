"""本模块用于存储配置信息"""

DEFAULT_MODEL = "gpt-4o"  # 默认模型
# 模型对应最大token长度,以下模型只适用于对话模型
MAX_TOKEN_LENGTH = {
    # gpt-4o
    "gpt-4o": 128000,
    # 我们最先进的，多模式的旗舰模型，比 GPT-4 Turbo 更便宜，更快。当前指向 gpt-4o-2024-05-13 。
    "gpt-4o-2024-05-13": 128000,  # gpt-4o-2024-05-13
    # gpt-4-turbo # 带有视觉功能
    "gpt-4-0125-preview": 128000,
    # GPT-4 Turbo 预览模型旨在减少模型不完成任务的“懒惰”情况。返回最多 4096 个输出令牌。了解更多。
    "gpt-4-1106-preview": 128000,
    # GPT-4 Turbo 预览模型具有改进的指令跟踪、JSON 模式、可复制输出、并行函数调用等功能。返回最多 4096 个输出令牌。这是一个预览模型。了解更多。
    "gpt-4-turbo-preview": 128000,  # GPT-4 Turbo 预览模型。当前指向 gpt-4-0125-preview 。
    # gpt-4
    "gpt-4-0613": 8192,  # 2023 年 6 月 13 日的 gpt-4 快照，具有改进的函数调用支持。
    "gpt-4": 8192,  # 当前指向 gpt-4-0613 。请查看持续的模型升级。
    # gpt 4 v
    "gpt-4-turbo": 128000,
    # 最新地具有视觉能力的 GPT-4 Turbo 模型。视觉请求现在可以使用 JSON 模式和函数调用。当前指向 gpt-4-turbo-2024-04-09 。
    "gpt-4-turbo-2024-04-09": 128000,
    # 带有视觉模型的 GPT-4 Turbo。视觉请求现在可以使用 JSON 模式和函数调用。 gpt-4-turbo 当前指向此版本。
    "gpt-4-vision-preview": 128000,
    # 具有理解图像能力的 GPT-4 模型，此外还具有所有其他 GPT-4 Turbo 功能。这是一个预览模型，我们建议开发者现在使用 gpt-4-turbo ，其中包括视觉功能。目前指向 gpt-4-1106-vision-preview 。
    "gpt-4-1106-vision-preview": 128000,
    # GPT-4 模型具有理解图像的能力，此外还具有所有其他 GPT-4 Turbo 的功能。这是一个预览模型，我们建议开发者现在使用 gpt-4-turbo ，它包含了视觉功能。返回的输出令牌最多为 4096 个。了解更多。
    # gpt-3.5
    "gpt-3.5-turbo-1106": 16385,
    # GPT-3.5 Turbo 模型，具有改进的指令跟踪、JSON 模式、可复制输出、并行函数调用等更多功能。返回的最大输出令牌数为 4096。了解更多。
    "gpt-3.5-turbo": 16385,  # 当前指向 gpt-3.5-turbo-0125
    "gpt-3.5-turbo-0125": 16385
    # 最新的 GPT-3.5 Turbo 模型，提高了以请求格式回应的准确性，并修复了一个导致非英语语言函数调用的文本编码问题的错误。返回最多 4,096 个输出令牌。了解更多。
}

# 设置颜色
RED = "\033[31m"  # 红色文本
GREEN = "\033[32m"  # 绿色文本
YELLOW = "\033[33m"  # 黄色文本
RESET = "\033[0m"  # 重置颜色
MAGENTA = "\033[35m"  # 洋红色文本
CYAN = "\033[36m"  # 青色文本

# 设置重试: 每个数字表示多少秒后重试, tuple的长度表示重试次数
RETRY = (3, 30, 60)
