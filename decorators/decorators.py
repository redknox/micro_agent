"""本模块用于放置给micro_agent模块的装饰器。"""
import gradio as gr

from wee_agent import WeeAgent


def user_interface(cls: WeeAgent):
    """装饰器函数，为MicroAgent的子类添加两个用户界面的方法。第一个为对话界面，用于进行多轮对话测试，第二为输入输入界面，用于问答类代理。"""

    if not issubclass(cls, WeeAgent):
        raise TypeError(
            "The decorated class should be a subclass of MicroAgent.")

    def chat(self):
        gr.ChatInterface(
            fn=self.__call__,
            title=self.name,
            description=self.prompt,
        ).launch()

    def interface(self):
        gr.Interface(
            fn=self.__call__,
            inputs=['text'],
            outputs=[
                'json' if self.response_format == 'json_object' else 'markdown'],
            title=self.name,
            description=self.__doc__,
            article=self.prompt,
        ).launch()

    cls.chat = chat
    cls.interface = interface

    return cls


def feishu_robot(cls: WeeAgent):
    """装饰器函数，为MicroAgent的子类添加一个飞书机器人接口。
    装饰后的类，会自动添加一个feishu_robot方法，运行后会启动一个服务器，监听指定的端口，接收飞书机器人的请求，并返回对应的回复。
    """

    if not issubclass(type(cls), WeeAgent):
        raise TypeError(
            "The decorated class should be a subclass of MicroAgent.")

    def feishu(self):
        pass

    cls.feishu_robot = feishu

    return cls
