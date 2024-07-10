from wee_agent.agents.coder import Coder
from wee_agent.agents.python_code_executor import CodeExecutor

require_text = """
本函数的功能是：获取指定时间当周开始和结束的时间戳。
输入参数：一个合法的时间戳。或者未输入。如果未输入，则默认为当前时间。
返回值：返回一个包含开始时间戳和结束时间戳的tuple。
raise：如果输入的时间戳不合法，则raise ValueError。
"""
#
# coder_executor = CodeExecutor()
#
coder = Coder()
coder.temperature = 0.0

code = coder(require_text)
print(code)

# ret = coder_executor(code)
#
# print(ret, coder_executor.code)
