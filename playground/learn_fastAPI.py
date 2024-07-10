import base64
import hashlib
import json
import os
from typing import List, Optional

import requests
import uvicorn
from Crypto.Cipher import AES
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel

load_dotenv()


class AESCipher(object):
    def __init__(self, key):
        self.bs = AES.block_size
        self.key = hashlib.sha256(AESCipher.str_to_bytes(key)).digest()

    @staticmethod
    def str_to_bytes(data):
        u_type = type(b"".decode('utf8'))
        if isinstance(data, u_type):
            return data.encode('utf8')
        return data

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s) - 1:])]

    def decrypt(self, enc):
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size:]))

    def decrypt_string(self, enc):
        enc = base64.b64decode(enc)
        return self.decrypt(enc).decode('utf8')


# 解密收到的数据
def decrypt_string(key, data):
    cipher = AESCipher(key)
    return cipher.decrypt_string(data).decode('utf8')


# encrypt = "P37w+VZImNgPEO1RBhJ6RtKl7n6zymIbEG1pReEzghk="
# cipher = AESCipher("test key")
# print("明文:\n{}".format(cipher.decrypt_string(encrypt)))
#
#
# def verify_signature(timestamp: str, nonce: str, encrypt_key: str, body: bytes):
#     bytes_b1 = (timestamp + nonce + encrypt_key).encode('utf-8')
#     bytes_b = bytes_b1 + body
#     h = hashlib.sha256(bytes_b)
#     signature = h.hexdigest()
#     return signature


app = FastAPI()


class ChallengeEvent(BaseModel):
    challenge: str


class MessageEvent(BaseModel):
    class Header(BaseModel):
        event_id: str
        event_type: str
        create_time: str
        token: str
        app_id: str
        tenant_key: str

    class Event(BaseModel):
        class SenderId(BaseModel):
            union_id: str
            user_id: str
            open_id: str

        class Sender(BaseModel):
            sender_id: "MessageEvent.Event.SenderId"
            sender_type: str
            tenant_key: str

        class Message(BaseModel):
            class Mention(BaseModel):
                key: str
                id: "MessageEvent.Event.SenderId"
                name: str
                tenant_key: str

            message_id: str
            root_id: Optional[str] = None
            parent_id: Optional[str] = None
            create_time: str
            update_time: str
            chat_id: str
            thread_id: Optional[str] = None
            chat_type: str
            message_type: str
            content: str
            mentions: Optional[
                List["MessageEvent.Event.Message.Mention"]] = None
            user_agent: Optional[str] = None

        sender: Sender
        message: Message

    schema: str
    header: Header
    event: Event


class MainEvent(BaseModel):
    header: Optional[MessageEvent.Header] = None
    event: Optional[MessageEvent.Event] = None
    challenge: Optional[str] = None


def send_message_by_feishu(user_id: str, message: str):
    # 获取tenant token
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    headers = {
        "Content-Type": "application/json; charset=utf-8"
    }
    data = {
        "app_id": os.getenv("APP_ID"),
        "app_secret": os.getenv("APP_SECRET")
    }
    response = requests.post(url, headers=headers, json=data)
    _response_data = response.json()
    tenant_access_token = _response_data["tenant_access_token"]

    # 发送消息
    url = "https://open.feishu.cn/open-apis/im/v1/messages"
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": "Bearer " + tenant_access_token
    }
    params = {
        'receive_id_type': 'user_id'
    }
    data = {
        "receive_id": user_id,
        "msg_type": "text",
        "content": json.dumps({
            "text": message
        })
    }
    response = requests.post(url, headers=headers, params=params, json=data)
    _response_data = response.json()
    return _response_data


@app.post("/feishu/webhook/")
async def handle_event(request: Request):
    key = os.getenv("ENCRYPT_KEY")
    if not key:
        raise ValueError("ENCRYPT_KEY is not set")
    data = await request.body()
    event = decrypt_string(key, data)
    event = MessageEvent(**event)

    if event.challenge:
        return {"challenge": event.challenge}
    else:

        return {"err_no": 0, "errmsg": "ok", "data": {}}


if __name__ == '__main__':
    ret = send_message_by_feishu(
        user_id="858a4ea5",
        message="Hello, World!"
    )
    print(ret)
    uvicorn.run(app, host="0.0.0.0", port=7001, log_level="info")

    # response_data = {'schema': '2.0',
    #                  'header': {'event_id': '54bec0cec9bc4a29e89837f90ff88819',
    #                             'token': '6epiZitMxCpzgC81u09w8fPxxY2aKyQj',
    #                             'create_time': '1712112411644',
    #                             'event_type': 'im.message.receive_v1',
    #                             'tenant_key': '2c080fd455ce575f',
    #                             'app_id': 'cli_a6815dfbabb89013'}, 'event': {
    #         'message': {'chat_id': 'oc_a14d85b2536a46bb035b97ee7a05af9b',
    #                     'chat_type': 'p2p', 'content': '{"text":"hello"}',
    #                     'create_time': '1712112411328',
    #                     'message_id': 'om_ccf385d672b3d3b0f1d8a185a4b2c0ae',
    #                     'message_type': 'text', 'update_time': '1712112411328'},
    #         'sender': {
    #             'sender_id': {'open_id': 'ou_9594ba5b5dda5e5ac45c0b2a42c33852',
    #                           'union_id': 'on_dbc1edf78215e2d76550ab605ec4e3b6',
    #                           'user_id': '858a4ea5'}, 'sender_type': 'user',
    #             'tenant_key': '2c080fd455ce575f'}}}
    #
    # t = MessageEvent(**response_data)
    #
    # print(t)
