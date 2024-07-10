"""
本模块用定义一个可以在Docker中执行传入代码的Agent

todo：保持一个容器运行，每次执行代码时，不需要重新创建容器
todo：指定一个requirements.txt文件，可以预先安装需要的库，避免每次安装库
todo：可以指定python版本，下载对应的镜像
todo：可以指定一个项目，将代码放入项目中执行 ->
"""
import ast
import logging
import uuid
from typing import List

import docker
import requests.exceptions
from docker.errors import APIError, ImageNotFound

from wee_agent import WeeAgent, set_tool

PYTHON_IMG_NAME = 'python:latest'
BASE_PROMPT = """
你是一名资深的python程序员，熟悉python的开发与代码调试。你的职责是帮助我调试代码：
需要调试的代码位于本段落最后，用三个反引号```包裹。
第一步：请先阅读源代码，判断如果正常执行，返回值会是怎样。
例如，源代码为"print('hello world')",则输出应该为"hello world"。
第二步：请调用exec_python_code方法，执行这段代码，获得执行的结果。
第三部：请分析执行结果，如果执行报错，请根据报错信息进行分析：
    如果错误原因是运行环境中缺少某个库，请调用install_runtime_lib工具进行安装。
    如果是代码本身有问题，则请直接对代码进行修改，并将修改后的代码用save_code()进行保存。
        如果你生成的代码保存失败，请根据save_code()返回的错误信息再次修改代码。
然后再次回到第二步。重复上面的步骤，直到程序测试无误，或者出现的错误无法通过添加库或修改代码来实现。如：调用的接口不通、读取的文件不存在、代码中用的库通过install_runtime_lib工具安装失败等。
如果代码调试无误，请返回"代码调试成功！"
如果无法调试，则返回"无法调试！"
以下用三个反引号包裹的内容为需要调试的代码。这些内容会随着你对代码的修改而变动。
``` ```"""


class CodeExecutor(WeeAgent):
    def __init__(self, *args, code="", **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'CodeDebugger'
        super().__init__(
            *args,
            **kwargs
        )
        self.base_prompt: str = BASE_PROMPT
        self.code: str = ''

        if code and self.save_code(code) != 'success':
            raise ValueError("传入的代码不是python 代码！")

        self.container_name: str = str(
            uuid.uuid4())  # 确保container名字不与当前其他container冲突

        # 获取docker环境
        try:
            self.docker_client: docker.client = docker.from_env()
        except APIError as e:
            logging.error("调用API收到错误返回码")
            raise e
        except docker.errors.DockerException as e:
            logging.error("无法从环境变量中初始化配置")
            raise e
        except requests.exceptions.ConnectionError as e:
            logging.error("连接Docker守护进程失败，Docker服务可能为运行！")
            raise e
        except Exception as e:
            logging.error(f"未知错误{e}")
            raise e

        # 使用python镜像创建容器
        try:
            self.docker_client.ping()
            logging.info("docker环境测试正常")
            self.docker_client.images.get(PYTHON_IMG_NAME)
            logging.info(f"镜像{PYTHON_IMG_NAME}已存在！")
        except APIError:
            raise RuntimeError("Docker 服务不可用，请确保Docker正在运行。")
        except ImageNotFound:
            logging.info(f"未能找到镜像{PYTHON_IMG_NAME},尝试拉取...")
            try:
                self.docker_client.images.pull(PYTHON_IMG_NAME)
                logging.info(f"镜像拉取成功！")
            except APIError as e:
                raise RuntimeError(f"无法拉取镜像{PYTHON_IMG_NAME}:{e}")
        try:
            self.container = self.docker_client.containers.create(
                image=PYTHON_IMG_NAME,
                name=self.container_name,
                command="/bin/bash -c 'while true; do echo hello world; sleep 1000; done'",
                # 执行死循环指令，保持容器运行状态
                detach=True
            )
            logging.info('容器创建完毕。')
            self.container.start()
            logging.info("容器启动完毕。")
        except APIError as e:
            raise RuntimeError(f"无法通过镜像{PYTHON_IMG_NAME}创建容器！{e}")

    # def __del__(self):
    #     """
    #     析构函数，清理代码调试时使用的容器
    #     :return:
    #     """
    #     try:
    #         self.container.stop()
    #         self.container.remove()
    #         logging.info("容器已经删除！")
    #     except Exception as e:
    #         logging.warning(f"删除容器失败！{e}")

    @set_tool
    def save_code(self, code: str) -> str:
        """
        保存修改后的代码。执行后会修改system提示词中包含的代码部分。
        :param code: 需要保存的代码信息
        :return: 如果传入的代码没有错误，则返回"success"；如果传入的代码有语法错误，则返回"failure:{错误信息}"
        """
        try:
            ast.parse(code)
            self.code = code
            self.prompt = self.base_prompt.replace("``` ```",
                                                   f"```{self.code}```")
            return 'success'
        except SyntaxError as e:
            print(e)
            return f'failure:{e}'

    @set_tool
    def install_runtime_lib(self, lib: str) -> str:
        """
        安装python开发环境需要的模块。
        :param lib:模块的名称
        :return:执行安装模块命令 'pip install {模块名}' 后返回的安装系信息。
        """
        # 构建安装库和执行代码的命令
        if lib is not None:
            pip_install_cmd = ['pip', 'install', lib]
            logging.info(f"安装库{lib}中...")
            return self.exec_command(pip_install_cmd)

    @set_tool
    def exec_python_code(self, code: str) -> str:
        """
        在开发环境中执行输入的代码
        :param code: 待执行的python代码
        :return: 代码执行后的输出信息
        """
        return self.exec_command(
            command=['python', '-c', code]
        )

    def exec_command(self, command: List[str] | str) -> str:
        """
        执行代码
        :param command:
        :return:
        """
        try:
            exec_instance = self.docker_client.api.exec_create(
                container=self.container.id,
                cmd=command,
                stdout=True,
                stderr=True
            )

            output = self.docker_client.api.exec_start(
                exec_id=exec_instance['Id']
            )
            logging.info(f"执行命令{command} 结果:{output.decode('utf-8')}")
            return output.decode('utf-8')

        except APIError as e:
            print(f'执行错误！i{e}')

    def debug_code(self):
        """
        调试代码：
        :return:
        """
        if self.code == '':
            logging.error('未传入待调试的代码！')
            exit()

        ret = self.create()
        print(ret)

    def delete_container(self):
        """
        删除容器
        :return:
        """
        try:
            self.container.stop()
            self.container.remove()
            logging.info("容器已经删除！")
        except Exception as e:
            logging.warning(f"删除容器失败！{e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    python_code = """
load_dotenv()
"""
    ta = CodeExecutor()
    ta.save_code(python_code)
    ta.debug_code()
    print(f"最终通过测试的代码为：\n```python\n{ta.code}\n```")
    ta.delete_container()
    # re = ta.install_runtime_lib('requests')
    # print(re)
    # python_code = "print('hello world')"
    # print(ta.exec_code(python_code))
