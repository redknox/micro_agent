import unittest

from micro_agent import MicroAgent


class MyTestCase(unittest.TestCase):
    def test_init(self):
        test_agent = MicroAgent(
            name='test_agent',
            user_name='test_user',
            model='gpt-4-turbo-preview',
            prompt='我们来玩成语接龙吧，我先说，你接着说。可以同音不同字。',
            need_user_input=True,
            input_token_ratio=0.5,
            max_round=3
        )
        print(test_agent)
        self.assertEqual(test_agent.name, 'test_agent')
        self.assertEqual(test_agent.user_name, 'test_user')
        self.assertEqual(test_agent.model, 'gpt-4-turbo-preview')
        self.assertEqual(test_agent.prompt, '我们来玩成语接龙吧，我先说，你接着说。可以同音不同字。')
        self.assertEqual(test_agent.input_token_ratio, 0.5)
        self.assertEqual(test_agent.max_round, 3)

        # self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
