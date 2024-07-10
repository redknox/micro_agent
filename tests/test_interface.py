"""测试用户界面装饰器"""
import unittest

from wee_agent import WeeAgent
from wee_agent.decorators import user_interface


# 测试类
@user_interface
class TestUserInterfaceAgent(WeeAgent):
    pass


class MyTestCase(unittest.TestCase):
    def test_Q_and_A_interface(self):
        test_agent = TestUserInterfaceAgent()
        test_agent.response_format = 'json_object'
        test_agent.prompt = "我会问你历届美国总统是谁，请用json格式回答。格式为{'第几任': '姓名'}"
        test_agent.interface()
        self.assertTrue(True)

    def test_chat_interface(self):
        test_agent = TestUserInterfaceAgent()
        test_agent.prompt = "你是一个历史学家，专业研究美国政治史，对美国政治制度的变迁了如指掌。请回答我关于美国总统的问题。"
        test_agent.chat()
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
