"""本模块用于处理prompt
一个结构化的提示词，可以分为以下几个部分：
1. 角色 - role 你期待llm扮演的角色。
2. 技能 - skills 你期待llm具备的技能。
3. 能力 - capability 在这里明确可以调用的函数
3. 个性 - personality 你期待llm具备的个性。或者回复的风格。（非必要）
4. 背景或上下文 - background 任务或目标的背景或上下文。
5. 目标 - goal 你期待llm达到的目标。
6. 任务或工作流 - workflow 你期待llm完成的任务或工作流。
7. 输入 - input 你期待llm输入的内容。
8. 处理 - process 你期待llm处理的内容。
9. 输出 - output 你期待llm输出的内容。
10. 评估 - evaluation 你期待llm对任务的评估。
11. 原则 - rule 你期待llm遵循的原则。
11. 例子 - example 你给llm的例子。
12. 初始化 - init 你期待llm进入对话时的信息。
"""
import pickle
from typing import List

from pydantic import BaseModel


class Prompt(BaseModel):
    """
    prompt 类，可以用来创建和管理prompt
    1. 一步一步，协助用户创建一个格式化的prompt
    2. 展示prompt的内容
    3. 将prompt保存起来
    4. 从文件中加载prompt
    5. 可以针对性的优化具体的prompt
    """

    class Profile(BaseModel):
        author: str
        version: str
        language: str
        description: str

    class Config:  # 配置类,验证输入
        validate_assignment = True

    profile: Profile = Profile(author="Haifeng Kong",
                               version="0.1",
                               language="zh",
                               description="")
    role: str = ""
    skills: List[str] | str = ''
    Tools_and_Capabilities: List[str] | str = ''
    personality: List[str] | str = ""
    background: List[str] | str = ""
    goal: List[str] | str = ""
    output_format: List[str] | str = ""
    workflow: List[str] | str = ''
    evaluation: List[str] | str = ""
    rule: List[str] | str = ''
    example: List[str] | str = ""
    init: List[str] | str = ""

    def __str__(self):
        ret = ""
        for k, v in self.dict().items():
            format_value = self.print_info(v)
            if format_value:  # 过滤掉空值
                ret += f"{k}: {format_value}\n"
        return ret

    @staticmethod
    def input_with_default(_prompt, default):
        user_input = input(_prompt)
        if not user_input:
            return default
        return user_input

    @staticmethod
    def collect_info(info):
        user_input = input(info)
        if not user_input:
            return ''

        continue_input = input("输入下一条？(回车退出): ")
        if continue_input != '':
            traits = [user_input, continue_input]
            while True:
                user_input = input("请输入下一条：(回车退出): ")
                if user_input == '':
                    break
                traits.append(user_input)

            if len(traits) == 1:
                return traits[0]
            else:
                return traits
        else:
            return user_input

    @staticmethod
    def print_info(item):
        if isinstance(item, dict):
            ret = ''
            for k, v in item.items():
                ret += f"\n  - {k}: {v}"
            return ret
        elif isinstance(item, list):
            return "\n  - " + "\n  - ".join(item)
        else:
            return item

    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def create(self, item=None):
        if item and item in self.dict().keys():
            print(f"create a new {item}")
            setattr(self, item, self.collect_info(f"请输入{item}："))
        else:
            print("create a new prompt")
            self.profile.author = self.input_with_default("你的名字：",
                                                          'Haifeng Kong')
            self.profile.version = self.input_with_default("版本号：", '0.1')
            self.profile.language = self.input_with_default("语言：", '中文')
            self.profile.description = input("描述：")
            self.role = input("你希望llm来扮演谁？: ")
            self.skills = self.collect_info("你希望llm具备哪些技能？:")
            self.personality = self.collect_info("你希望llm具备哪些个性？: ")
            self.background = self.collect_info("任务或目标的背景或上下文：")
            self.goal = self.collect_info("你希望llm达到的目标：")
            self.workflow = self.collect_info("你希望llm完成的任务或工作流：")
            self.evaluation = self.collect_info("你希望llm对如何对结果进行评估：")
            self.rule = self.collect_info("你希望llm遵循的原则：")
            self.example = self.collect_info("你给llm的例子：")
            self.init = self.collect_info("你希望llm进入对话时的信息：")


if __name__ == '__main__':
    p = Prompt()
    p.profile.author = 'llm'
    p.create()
    p.save('test.pkl')
    temp = Prompt.load('test.pkl')
    print(temp)

    # p.create()
    # c = p.collect_info("你希望llm具备哪些技能？:")
    # print(c)
    # p.skills = ["a", "b", "c"]
    # print(p)
    # for k, v in p.dict().items():
    #     print(k, v)
    # print(v)
#
#
# def create_prompt() -> str:
#     """
#     创建一个新的prompt
#     :return: 新的prompt
#     """
#     steps = {
#         # 角色
#         "role": "",
#         "skills": "",
#         "personality": "",
#
#         # 背景或上下文
#         "background": "",
#         "context": "",
#         "scenario": "",
#         "situation": "",
#         "complications": "",
#
#         # 意图或密保
#         "target": "",
#         "objectives": "",
#         "expectation": "",
#         "experiment": "",
#         "goal": "",
#         "format": "",
#         "problem": "",
#         "result": "",
#         "request": "",
#
#         # 任务或行动
#         "task": "",
#         "action": "",
#         "rules": "",
#         "workflow": "",
#         "steps": "",
#         "plan": "",
#
#         # 输入
#         "input": "",
#
#         # 评估
#         "evaluation": "",
#
#         # 例子
#         "example": "",
#     }
#
#     return "``` ```"
