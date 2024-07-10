"""学习向量数据库milvus"""
import openai
from pymilvus import (
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    connections
)


def get_embedding(text: str) -> list:
    """
    获取文本的向量表示
    :param text: 输入文本
    :return: 文本的向量表示
    """
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        encoding_format='float'
    )
    embedding_vector = response.data[0].embedding
    print(len(embedding_vector))
    return embedding_vector


# 连接milvus
def connect_milvus():
    connections.connect(host='10.60.84.212', port='19530')

    # 定义字段和集合模式
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True,
                    auto_id=True),
        FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=1536)
    ]
    collection_schema = CollectionSchema(fields=fields,
                                         description='example collection')

    # 创建集合
    collection_name = 'example_collection'
    collection = Collection(name=collection_name, schema=collection_schema)
    return collection


# 插入向量
def insert_vector(collection, embedding_vector):
    mr = collection.insert([[embedding_vector]])
    print(f"Insert IDs:{mr.primary_keys}")


# 查询向量
def search_vector(collection, embedding_vector, top_k=3):
    load_collection(collection)
    print(type(embedding_vector))
    print(type(embedding_vector[0]))
    if not all(isinstance(i, float) for i in embedding_vector):
        raise TypeError("The type of embedding vector must be float")

    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }

    search_result = collection.search(
        data=[embedding_vector],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["id"]
    )
    for hits in search_result:
        print(hits.ids,hits.distances)
    return search_result

def create_index(collection):
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 2048}
    }
    collection.create_index(field_name="vector", index_params=index_params)
    print('Index created successfully')

def load_collection(collection):
    try:
        collection.load()
        print('Collection loaded successfully')
    except Exception as e:
        print(f'Error: {e}')
        print('Collection not loaded')


def main():
    text = "Hello, world"
    embedding_vector = get_embedding(text)
    collection = connect_milvus()
    # create_index(collection)
    # insert_vector(collection, embedding_vector)
    search_vector(collection, embedding_vector)


if __name__ == '__main__':
    main()
