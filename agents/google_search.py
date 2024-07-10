"""本模块用于调用google搜索api搜索信息，可以用自然语言对ai提出搜索的具体要求，由ai生成google搜索关键词。"""
import json
import logging
import os
from typing import Optional, Dict

import requests
from dotenv import load_dotenv

from micro_agent import MicroAgent, set_tool

load_dotenv()

BASE_PROMPT = """你是一名google工程师，非常擅长应用google搜索的各种高级功能。
你能够根据我的意图，帮我写出最佳的google搜索词，并调用tool_google_search方法来获得结果，并结合我的需求和搜索结果来回答我的需求：例如：
我的需求是帮我查询HONG KONG YC DIGITAL TECHNOLOGY公司是否有任何涉及加密货币、高利贷、等违法问题。"
你生成如下查询词：'"HONG KONG YC DIGITAL TECHNOLOGY" AND (cryptocurrency OR "high interest" OR illegal OR scam OR fraud) -jobs -review'
然后调用tool_google_search方法，获得以下搜索结果：
['香港鈺程數字技術有限公司 https://www.ycdigitals.com/ 香港鈺程數字技術有限公司. HONG KONG YC DIGITAL TECHNOLOGY LIMITED. 是一家技术驱动发展的国际化智能营销服务公司，致力于为客户提供全球营销推广服务，通过效果营销 ...',]
然后你结合搜索结果和我的疑问回答我："根据搜索结果，并未发现该公司有任何涉及加密货币、高利贷、等违法问题。"
请注意，如果公司名字为中文，或者名字中出现HONG KONG，请同时搜索中文内容。
""".strip()


def google_search(query: str, num: int = 3, api_key=None,
                  ces_id=None) -> Optional[Dict]:
    """
    调用google search api获取信息
    :param ces_id:
    :param api_key:
    :param query: 查询字符串
    :param num: 获取最大记录条数
    :return: 查询结果
    """
    # 检查api_key和ces_id是否存在
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    ces_id = ces_id or os.getenv("GOOGLE_CES_ID")
    if not api_key or not ces_id:
        logging.error(
            "Google Custom Search API key or Custom Search Engine ID not found")
        raise ValueError(
            "Google Custom Search API key or Custom Search Engine ID not found")

    url = 'https://www.googleapis.com/customsearch/v1'
    params = {'q': query, 'key': api_key, 'cx': ces_id, 'num': num}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print(response.status_code)
        print("Failed to get response from Google Custom Search API")
        return None


class GoogleSearchAgent(MicroAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt = BASE_PROMPT

    @staticmethod
    @set_tool
    def google_search(query: str) -> str:
        """
        使用查询字符串通过google search api查询
        :param query: 待搜索的字符串
        :return: 查询结果，由google搜索结果中的信息构成,为一个json字符串，结构为:[{"title":{网站名称},"snippet":{摘要} }...]
        """
        search_result = google_search(query, num=10)
        ret = []
        for i in search_result.get('items', []):
            ret.append({
                "title": i['title'],
                "snippet": i['snippet']
            })

        return json.dumps(ret)


if __name__ == "__main__":
    mg = GoogleSearchAgent()
    mg.steam = True
    mg.stream_options = True
    # mg.user_input("大模型 claude 的官网网址是？")
    # ret = mg.create()
    # print(ret)
    print(mg("大模型 claude 的官网网址是？"))
