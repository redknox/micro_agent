import unittest

from micro_agent.agents import GoogleSearchAgent


class MyTestCase(unittest.TestCase):
    def test_something(self):
        agent = GoogleSearchAgent()
        print(agent)
        re = agent("python的官方网站是什么？")
        print(re)
        # self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
