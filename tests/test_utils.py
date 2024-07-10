import json
import unittest

from micro_agent.core.utils import num_tokens_from_messages, \
    create_jsonschema_from_example, validate_json, extract_json, get_image_encoding

messages = [
    {
        "content": """你是一名google工程师，非常擅长应用google搜索的各种高级功能。
你能够根据我的意图，帮我写出最佳的google搜索词，并调用tool_google_search方法来获得结果，并结合我的需求和搜索结果来回答我的需求：例如：
我的需求是帮我查询HONG KONG YC DIGITAL TECHNOLOGY公司是否有任何涉及加密货币、高利贷、等违法问题。"
你生成如下查询词：'"HONG KONG YC DIGITAL TECHNOLOGY" AND (cryptocurrency OR "high interest" OR illegal OR scam OR fraud) -jobs -review'
然后调用tool_google_search方法，获得以下搜索结果：
['香港鈺程數字技術有限公司 https://www.ycdigitals.com/ 香港鈺程數字技術有限公司. HONG KONG YC DIGITAL TECHNOLOGY LIMITED. 是一家技术驱动发展的国际化智能营销服务公司，致力于为客户提供全球营销推广服务，通过效果营销 ...',]
然后你结合搜索结果和我的疑问回答我："根据搜索结果，并未发现该公司有任何涉及加密货币、高利贷、等违法问题。"
请注意，如果公司名字为中文，或者名字中出现HONG KONG，请同时搜索中文内容。
""".strip(),
        "role": "system",
    },
    {"content": 'python的官方网站是什么？',
     "role": 'user',
     "name": 'user'
     },  # 450
    {
        "content": None,
        "role": 'assistant',
        "function_call": None,
        "tool_calls":
            [
                {
                    # "id": 'call_QDEnP3bBGkzaKKMVIHysVKdj',
                    "function": {
                        "arguments": '{"query":"official website of Python"}',
                        "name": 'tool_google_search'
                    },
                    # "type": 'function'
                }
            ]
    },  # 18
    {
        # "content": '[{"title": "Welcome to Python.org", "snippet": "The official home of the Python Programming Language."}, {"title": "The Python Tutorial \\u2014 Python 3.12.2 documentation", "snippet": "Python is an easy to learn, powerful programming language ... site, https://www ... This page is licensed under the Python Software Foundation License Version 2."}, {"title": "Download Python | Python.org", "snippet": "The official home of the Python Programming Language. ... This site hosts the \\"traditional\\" implementation of Python (nicknamed CPython). ... There is also a\\u00a0..."}, {"title": "Python Docs", "snippet": "This is the official documentation for Python 3.12. ... For C/C++ programmers. Python\'s C API C API ... This page is licensed under the Python Software Foundation\\u00a0..."}, {"title": "Python For Beginners | Python.org", "snippet": "The official home of the Python Programming Language. ... There is a list of tutorials suitable for experienced programmers on the BeginnersGuide/Tutorials page."}, {"title": "What are the best websites to learn Python? I found that the official ...", "snippet": "Mar 31, 2013 ... \\u201cCodeacademy. com\\u201d is the best website to learn python from ground. This website has a course of python. First sign up on this site and start\\u00a0..."}, {"title": "Eric IDE documentation? - Python Help - Discussions on Python.org", "snippet": "Jan 31, 2020 ... ... Python programming language itself. That being said, I did find the following page on their official website for the Eric IDE: The Eric Python\\u00a0..."}, {"title": "pandas - Python Data Analysis Library", "snippet": "built on top of the Python programming language. Install pandas now! Getting ... Documentation (web) \\u00b7 Download source code. Follow us. Recommended books. Python\\u00a0..."}, {"title": "Python", "snippet": "r/Python: The official Python community for Reddit! Stay up to date with the latest news, packages, and meta information relating to the Python\\u2026"}, {"title": "Python (programming language) - Wikipedia", "snippet": "While Python 2.7 and older is officially unsupported, a different unofficial Python implementation, PyPy, continues to support Python 2, i.e. \\"2.7.18+\\" (plus\\u00a0..."}]',
        "content": '[{"title": "Welcome to Python.org", "snippet": "The official home of the Python Programming Language."}, {"title": "The Python Tutorial \\u2014 Python 3.12.2 documentation", "snippet": "The Python interpreter and the extensive standard library are freely available in source or binary form for all major platforms from the Python web site\\u00a0..."}, {"title": "Download Python | Python.org", "snippet": "The official home of the Python Programming Language."}, {"title": "Free Download | Anaconda", "snippet": "Anaconda\'s open-source Distribution is the easiest way to perform Python/R data science and machine learning on a single machine."}, {"title": "pandas - Python Data Analysis Library", "snippet": "pandas. pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming\\u00a0..."}, {"title": "Anaconda: Unleash AI Innovation and Value", "snippet": "Accelerate growth efficiently for everyone with the AI and data science experts."}, {"title": "PyCharm: The Python IDE for data science and web development by ...", "snippet": "Our website uses some cookies and records your IP address for the purposes of accessibility, security, and managing your access to the telecommunication network\\u00a0..."}, {"title": "PyPI \\u00b7 The Python Package Index", "snippet": "The Python Package Index (PyPI) is a repository of software for the Python programming language."}, {"title": "Is this the official Python Facebook page / group? - PSF ...", "snippet": "Nov 22, 2022 ... For those who are new to python programming who would like to learn first log in to the official website. www.python.org Tutorial -..."}, {"title": "Django: The web framework for perfectionists with deadlines", "snippet": "Django is a high-level Python web framework that encourages rapid development and clean, pragmatic design. ... Join the Django Discord Community; Official Django\\u00a0..."}]',
        "role": 'tool',
        "tool_call_id": 'call_QDEnP3bBGkzaKKMVIHysVKdj'
    },  # 916
    {
        "content": 'Python的官方网站是 [Python.org](https://www.python.org)。这是Python编程语言的官方主页。',
        "role": 'assistant',
        "function_call": None,
        "tool_calls": None
    }  # 34
]


class MyTestCase(unittest.TestCase):

    def test_get_image_encoding(self):
        with open('asuka.png', 'rb') as f:
            image_message = f.read()
        image_content = get_image_encoding(image_message)
        print(image_content)
        # print(len(image_content))
        # self.assertEqual(len(image_content), 137621)

    def test_extract_json(self):
        test_json = '我们一起说 {"name": "John", "age": 30, "city": "New York"} 吧'
        self.assertEqual(extract_json(test_json), [
            json.loads('{"name": "John", "age": 30, "city": "New York"}')])

    def test_create_jsonschema_from_example(self):
        test_json = '{"name": "John", "age": 30, "city": "New York"}'
        schema = create_jsonschema_from_example(test_json)
        print(schema)
        self.assertTrue(validate_json(json.loads(test_json), schema))

    def test_num_token(self):
        for message in messages:
            print(num_tokens_from_messages([message],
                                           model="gpt-4-turbo-preview"))
        # self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
