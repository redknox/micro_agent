import logging
import unittest

from wee_agent import WeeAgent


class MyTestCase(unittest.TestCase):

    def test_daily_todo(self):
        # 测试让llama.1 生成每日待办
        test_agent = WeeAgent(
            name='test_agent',
            model='llama3.1',
            base_url='http://10.60.84.212:11434/v1',
            stream=False,
            content_length=4096,

        )
        test_agent.response_format = 'json_object'
        prompt = "你是一名文档分析师，能够敏锐的从与我的对话中，提取生成待办事项。再与我确认后，生成json格式的待办事项。例如：我说：我今天得完成一个文档分析报告，你生成如下待办事项：{'title':'文档分析报告','content':'今天完成文档分析报告'}。"

        test_agent.prompt = prompt

        while True:
            my_word = input("请输入：")
            print(test_agent(my_word))

    def test_update(self):
        test_agent = WeeAgent(
            model="llama3.1",
            content_length=4096,
            base_url='http://10.60.84.212:11434/v1',
            stream=False,
        )
        test_agent.temperature = 0
        print(test_agent)
        print(test_agent('hi！'))
        print(test_agent.history_messages)

    # 测试gpt4v
    def test_image(self):
        test_agent = WeeAgent(
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
        test_agent = WeeAgent(
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
        test_agent = WeeAgent(
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
        test_agent = WeeAgent(
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
