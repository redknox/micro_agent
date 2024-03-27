import unittest

from micro_agent.agents import GoogleSearchAgent


class MyTestCase(unittest.TestCase):
    def test_something(self):
        agent = GoogleSearchAgent()
        print(agent)
        re = agent("openAI chat API 如何计算tool_calls 的token？")
        print(re)
        print()
        # self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
