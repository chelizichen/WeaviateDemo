import pandas as pd

df = pd.read_csv('./chats_202503241644.csv', encoding='utf-8')
# 导入数据
# 转成list形式
sentence_data = df.answer.tolist()

# print(sentence_data)
