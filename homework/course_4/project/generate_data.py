import json
import os

# 输入你的名字
name = 'GitEason'
# 重复次数
n = 10000

data = [
    {
        "conversation": [
            {
                "input": "请做一下自我介绍",
                "output": "我是{}的小助手，内在是上海AI实验室书生·浦语的7B大模型哦".format(name)
            }
        ]
    }
]

for i in range(n):
    data.append(data[0])


saved_dir = "./datasets/personal_assistant"
if not os.path.exists(saved_dir):
    os.makedirs(saved_dir)

with open(os.path.join(saved_dir,'personal_assistant.json'), 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)