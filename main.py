import json

import weaviate
# from langchain.document_loaders import DirectoryLoader, WebBaseLoader
from sentence_transformers import SentenceTransformer, __version__
from df import sentence_data
import pandas as pd

print("__version__", __version__)


def main():
    # 定义client
    client = weaviate.Client(url='http://localhost:9090')
    class_name = 'chat_list'  # class的名字

    # class_obj = {
    #     'class': class_name,  # class的名字
    #     'vectorIndexConfig': {
    #         'distance': 'l2-squared',  # 这里的distance是选择向量检索方式，这里选择的是欧式距离
    #     },
    # }

    # client.schema.create_class(class_obj)

    model = SentenceTransformer("/usr/ai/models/chinese-models", device="cpu")  # embeddings模型路径

    # 句子向量化
    sentence_embeddings = model.encode(sentence_data)
    # sentence_embeddings

    # 将句子和embeddings后的数据整合到DataFrame里面
    data = {
        'sentence': sentence_data,
        'embeddings': sentence_embeddings.tolist()
    }
    df = pd.DataFrame(data)
    # df

    # with client.batch(
    #         batch_size=100
    # ) as batch:
    #     for i in range(df.shape[0]):
    #         #         if i%20 == 0:
    #         print('importing data: {}'.format(i + 1))
    #         # 定义properties
    #         properties = {
    #             'sentence_id': i + 1,  # 这里是句子id, [1, 2, 3, ...]
    #             'sentence': df.sentence[i],  # 这里是句子内容
    #             #             'embeddings': df.embeddings[i],
    #         }
    #         custom_vector = df.embeddings[i]  # 这里是句子向量化后的数据
    #         # 导入数据
    #         client.batch.add_data_object(
    #             properties,
    #             class_name=class_name,
    #             vector=custom_vector
    #         )
    print('import completed')

    query = model.encode(['还有哪些事情没完成'])[0].tolist()  # 这里将问题进行 embeddings
    nearVector = {
        'vector': query
    }

    response = (
        client.query
        .get(class_name, ['sentence_id', 'sentence'])  # 第一个参数为class名字，第二个参数为需要显示的信息
        .with_near_vector(nearVector)  # 用向量检索，nearVector为输入问题的向量形式
        .with_limit(3)  # 返回个数(TopK)，这里选择返回3个
        .with_additional(['distance'])  # 选择是否输出距离
        .do()
    )

    print("response", json.dumps(response, indent=2, ensure_ascii=False))  # 看下输出


def delete_class():
    client = weaviate.Client(url='http://localhost:9090')
    # 删除 chat_list 类
    client.schema.delete_class('chat_list')


if __name__ == '__main__':
    # print("hello world")
    # delete_class()
    main()
