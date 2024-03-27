"本模块用于存储配置信息"

DEFAULT_MODEL = "gpt-4-turbo-preview"  # 默认模型
# 模型对应最大token长度,以下模型只适用于对话模型
MAX_TOKEN_LENGTH = {
    "gpt-4-0125-preview": 128000,
    "gpt-4-1106-preview": 128000,
    "gpt-4-turbo-preview": 128000,  # 当前指向 gpt-4-0125-preview
    "gpt-4-0613": 8192,
    "gpt-4": 8192,  # 当前指向 gpt-4-0613
    "gpt-3.5-turbo-1106": 16385,
    "gpt-3.5-turbo": 16385,  # 当前指向 gpt-3.5-turbo-1106
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
