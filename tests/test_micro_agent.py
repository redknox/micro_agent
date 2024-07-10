import logging
import unittest

from micro_agent import MicroAgent


class MyTestCase(unittest.TestCase):
    # 测试gpt4v
    def test_image(self):
        test_agent = MicroAgent(
            name='test_agent',
            model='gpt-4o',
            prompt='你是一名优秀的前端开发人员，擅长根据设计师设计的原型图，编写前端的html+css+js代码。请根据以下设计图，编写html+css代码。',
        )
        with open('截屏2024-05-14 15.07.12.png', 'rb') as f:
            image_message = f.read()
            test_agent.user_image_input(img_url=image_message)
            a = test_agent._create_messages()
            print(test_agent.create())
            # print(test_agent)

    def test_init(self):
        logging.basicConfig(level=logging.INFO)
        test_agent = MicroAgent(
            name='test_agent',
            user_name='test_user',
            model='gpt-4o',
            prompt='我说一个成语，你给我讲出这个成语的故事。',
            need_user_input=False,
            input_token_ratio=0.5,
            max_round=3,
            stream=True,
        )
        # test_agent.stream = True
        # test_agent.stream_options = True

        while True:
            my_cy = input("请输入成语：")
            ret = test_agent(my_cy)
            # print(ret)

    def test_token_control(self):
        test_agent = MicroAgent(
            name='test_agent',
            user_name='test_user',
            model='gpt-4o',
            prompt='我们来万成语接龙，我先来一个“天马行空”。你接一个成语，最后一个字和我的成语的第一个字相同。',
            need_user_input=False,
            input_token_ratio=0.5,
            max_round=3,
            stream=True,
        )
        # test_agent.stream = True
        # test_agent.stream_options = True

        while True:
            my_cy = input("请输入成语：")
            ret = test_agent(my_cy)
            print(test_agent)

    def test_draw_image(self):
        test_agent = MicroAgent(
            name='test_agent',
            user_name='test_user',
            model='gpt-4o',
            prompt='你是一名有用的ai助手，完成我的下达的任务。',
            need_user_input=False,
            input_token_ratio=0.5,
            max_round=3,
            stream=True,
            draw_image=True,
        )
        re = test_agent('请生成一张三只小猫互相拥抱的照片。')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
