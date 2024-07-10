import unittest

from wee_agent.agents import GoogleSearchAgent


class MyTestCase(unittest.TestCase):
    def test_something(self):
        agent = GoogleSearchAgent()
        agent.stream = True
        agent.stream_options = True
        print(agent)
        re = agent("邓超和孙俪是哪年结婚的？")
        print(re)
        print()
        # self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
