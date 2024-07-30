"""本模块用于存放公共函数"""
import base64
import inspect
import json
import logging
import random
import re
from typing import Any, Callable

import pydantic
import tiktoken
from genson import SchemaBuilder
from jsonschema import validate
from jsonschema.exceptions import ValidationError
from pydantic import BaseModel


# 将图片转换成base64编码
def image_to_base64(image: bytes) -> str:
    """
    将图片转换成base64编码
    :param image: 图片
    :return: base64编码
    """
    ext = get_image_encoding(image)
    base64_image = base64.b64encode(image).decode('utf-8')
    return f"data:{ext};base64,{base64_image}"


def extract_json(text: str) -> list:
    """
    从文本中提取json
    :param text: 待提取的文本
    :return: 提取的json
    """
    pattern = r"\{[^{}]*\}"
    matches = re.findall(pattern, text, re.DOTALL)

    json_objects = []
    for match in matches:
        try:
            json_object = json.loads(match)
            json_objects.append(json_object)
        except json.JSONDecodeError as e:
            print(f'json解析失败 {e}')

    return json_objects


# 从json样例文件创建jsonschema
def create_jsonschema_from_example(example: [str, dict]) -> dict:
    """
    从json样例文件创建jsonschema
    :param example: json样例文件
    :return: jsonschema
    """
    if isinstance(example, str):
        example = json.loads(example)
    builder = SchemaBuilder()
    builder.add_object(example)
    jsonschema = builder.to_schema()
    if "$schema" in jsonschema:
        del jsonschema["$schema"]
    return jsonschema


# 验证一个json是否符合jsonschema
def validate_json(json_data: dict, schema: dict) -> bool:
    """
    验证一个json是否符合jsonschema
    :param json_data: 待验证的json数据
    :param schema: jsonschema
    :return: 是否符合jsonschema
    """
    try:
        validate(json_data, schema)
        return True
    except ValidationError:
        return False


# 随机生成一个英文姓名，用于命名代理
def generate_random_name() -> str:
    """
    随机生成一个英文姓名，用于命名代理
    :return: 生成的英文姓名
    """
    first_names = [
        "John", "Jane", "Alice", "Bob", "Charlie", "David", "Eve", "Frank",
        "Grace", "Henry", "Ivy", "Jack", "Kate", "Lily", "Mike", "Nancy",
        "Oliver", "Pam", "Quentin", "Rose", "Steve", "Tina", "Ursula",
        "Victor", "Wendy", "Xander", "Yvonne", "Zack"
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller",
        "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White",
        "Harris", "Martin", "Thompson", "Garcia", "Martinez", "Robinson",
        "Clark", "Rodriguez", "Lewis", "Lee", "Walker", "Hall", "Allen",
        "Young", "Hernandez"
    ]
    return f"{random.choice(first_names)}_{random.choice(last_names)}"


def merge(base_model_1: BaseModel, base_model_2: BaseModel) -> BaseModel:
    """
    合并两个基于BaseModel的实例，返回一个新的实例
    业务逻辑如下：
    1. 如果两个属性都是BaseModel的实例，则递归合并
    2. 如果其中一个值为None且另一个不为None，则保留非None值
    3. 如果其中一个值为默认值且另一个不是（包括None），则保留非默认值
    4. 对于字符串类型，如果两个值不同，则拼接；否则保留其中之一
    5. 其他情况下，暂时选择第一个非None的值作为合并结果（或者可以根据需要调整）
    :param base_model_1: 合并的第一个BaseModel实例
    :param base_model_2: 合并的第二个BaseModel实例
    :return: 合并后的实例
    """
    if type(base_model_1) != type(base_model_2):
        raise ValueError("Cannot merge instances of different classes.")

    merged_data = {}

    for field_name in base_model_1.model_fields.keys():
        value1 = getattr(base_model_1, field_name)
        value2 = getattr(base_model_2, field_name)

        if isinstance(value1, BaseModel):
            merged_data[field_name] = merge(value1, value2)
        # 如果其中一个值为None且另一个不为None，则保留非None值
        elif isinstance(value1, list) and isinstance(value2, list):
            # 如果两个属性都是list，则检查大小是否一致
            if len(value1) != len(value2):
                merged_data[
                    field_name] = value1 if value1 is not None else value2
                continue  # 如果大小不一致，则不合并
            merged_list = []
            for item1, item2 in zip(value1, value2):
                # 如果列表项是BaseModel的实例，则递归合并
                if isinstance(item1, BaseModel) and isinstance(item2,
                                                               BaseModel):
                    merged_list.append(merge(item1, item2))
                else:
                    # 对于非BaseModel类型的列表项，目前只支持相同项（或根据需求自定义处理方式）
                    merged_list.append(item1 if item1 == item2 else None)
            merged_data[field_name] = merged_list
        elif value1 is None and value2 is not None:
            merged_data[field_name] = value2
        elif value2 is None and value1 is not None:
            merged_data[field_name] = value1
        # 如果其中一个值为默认值且另一个不是（包括None），则保留非默认值
        elif (value1 == base_model_1.model_fields[field_name].default and
              value2 != base_model_2.model_fields[field_name].default):
            merged_data[field_name] = value2
        elif (value2 == base_model_2.model_fields[field_name].default and
              value1 != base_model_1.model_fields[field_name].default):
            merged_data[field_name] = value1
        # 对于字符串类型，如果两个值不同，则拼接；否则保留其中之一
        elif isinstance(value1, str) and isinstance(value2, str):
            if value1 != value2:
                merged_data[field_name] = f"{value1}{value2}"
            else:
                merged_data[field_name] = value1
        else:
            # 其他情况下，暂时选择第一个非None的值作为合并结果（或者可以根据需要调整）
            merged_data[field_name] = value1 if value1 is not None else value2

    try:
        return type(base_model_1)(**merged_data)
    except pydantic.ValidationError as e:
        logging.error(
            f"将数据转换成对象时发生错误：{type(base_model_1)} {merged_data}")
        raise e


def num_tokens_from_messages(messages: list, model="gpt-3.5-turbo-0613"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    logging.debug(f"Encoding for model {model}: {encoding}")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:  # note: future models may deviate from this
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4
        tokens_per_name = -1
    elif "gpt-3.5-turbo" in model:
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        tokens_per_message = 3
        tokens_per_name = 1
    #     raise NotImplementedError(
    #         f"Model {model} not supported for token counting."
    #     )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if key not in ("role", "name", "content"):
                continue
            if value is None:
                value = ""
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def python_type_to_json_schema(python_type: Any) -> str:
    """
    Maps a Python type annotation to a JSON schema type.

    Parameters:
    - python_type (Any): The Python type annotation.

    Returns:
    - str: The corresponding JSON schema type as a string.
    """
    type_mapping = {
        int: 'integer',
        float: 'number',
        str: 'string',
        bool: 'boolean',
        list: 'array',
        tuple: 'array',
        dict: 'object',
        type(None): 'none'
    }

    # Handle typing module types like List[int] or Dict[str, int]
    origin_type = getattr(python_type, '__origin__', None)

    if origin_type is not None:
        return type_mapping.get(origin_type, 'none')

    return type_mapping.get(python_type, 'none')


def parse_docstring(docstring: str) -> (str, dict, str):
    """
    Parses the docstring to extract the main description and parameter descriptions.

    Parameters:
    - docstring (str): The docstring of a function.

    Returns:
    - str: The main description of the function.
    - dict: A dictionary with parameter names as keys and their descriptions as values.
    - str: The return description of the function.
    """
    lines = docstring.strip().splitlines()

    # Extract the main description (assumes it is before the :param lines)
    main_description_lines = []
    return_description_lines = []

    for line in lines:
        if line.strip().startswith(":param"):
            break
        if line and not line.isspace():
            main_description_lines.append(line.strip())

    for line in lines:
        if line.strip().startswith(":return:"):
            return_description_lines.append(line.strip())
            break

    main_description = " ".join(main_description_lines)
    return_description = " ".join(return_description_lines)

    param_descriptions = {}

    for line in lines:
        line = line.strip()

        # Check if the line contains parameter documentation
        if line.startswith(":param"):
            # Extract the parameter name and description
            param_name, param_name_desc = line[7:].split(":", 1)
            param_descriptions[param_name] = param_name_desc

    return main_description, param_descriptions, return_description


def generate_function_schema(func: Callable) -> dict:
    """
        Generates a JSON schema for the given function.

        Parameters:
          func (Callable): The function to generate schema for.

          Returns:
          dict: A dictionary representing the JSON schema of the function.
        """

    signature = inspect.signature(func)
    docstring = func.__doc__ or ""

    # Extract main description and parameter descriptions from the docstring
    main_description, param_descriptions, return_description = parse_docstring(
        docstring)

    properties = {}
    required = []

    for name, param in signature.parameters.items():
        annotation_type = python_type_to_json_schema(
            param.annotation) if param.annotation != inspect.Parameter.empty else 'none'

        if name == 'self':
            continue
        properties[name] = {
            "type": annotation_type,
            "description": param_descriptions.get(name, "")
        }

        if param.default is inspect.Parameter.empty:
            required.append(name)

    # Determine return type from annotations (if available)
    return_annotation = func.__annotations__.get('return', None)
    return_type_annotation = python_type_to_json_schema(
        return_annotation) if return_annotation is not None else 'none'

    schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": main_description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
            "returns": {
                "type": return_type_annotation,
                "description": return_description
                # Placeholder for potential future return value descriptions
            }
        }
    }

    return schema


# 将对象转换成字典
def deep_vars(obj):
    """
    递归地将对象及其嵌套对象转换为字典。
    """
    # 如果obj支持__dict__属性，则尝试递归转换
    if hasattr(obj, '__dict__'):
        return {key: deep_vars(value) for key, value in vars(obj).items()}
    elif isinstance(obj, list):  # 对列表中的每个元素进行递归处理
        return [deep_vars(item) for item in obj]
    elif isinstance(obj, dict):  # 对字典中的每个值进行递归处理
        return {key: deep_vars(value) for key, value in obj.items()}
    else:
        return obj  # 基本类型直接返回


def content_str(content: list[dict[str, Any]] | str | None) -> str:
    """
    将内容转换为字符串。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError(
            f"Invalid content type: {type(content)},must be None,str or list")
    rst = ''
    for item in content:
        if not isinstance(item, dict):
            raise TypeError(
                f"Invalid content type: {type(content)},must be dict")
        assert "type" in item, "Wrong content format. Missing 'type' key in content's dict."
        if item["type"] == "text":
            rst += item["content"]
        elif item["type"] == "image_url":
            rst += f"![image]({item['content']})"
        else:
            raise ValueError(f"Invalid content type: {item['type']}")
        return rst

    if isinstance(content, list):
        return "\n".join(
            [f"{item['role']}: {item['content']}" for item in content])
    raise ValueError(f"Invalid content type: {type(content)}")


# 获取图片编码格式
def get_image_encoding(image: bytes) -> str:
    """
    获取图片编码格式
    :param image: 图片
    :return: 图片编码格式
    """
    if image.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'image/png'
    elif image.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    elif image.startswith(b'GIF87a') or image.startswith(b'GIF89a'):
        return 'image/gif'
    else:
        raise ValueError("Unsupported image format")
