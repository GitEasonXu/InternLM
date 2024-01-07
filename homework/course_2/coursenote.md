# 第2节：轻松玩转书生·浦语大模型趣味 Demo

### 目录

- 环境准备
- `internlm-chat-7b`模型下载
- Demo演示

### 1. 环境准备

安装必要的库，版本torch版本需要根据自己的CUDA版本确定。

```bash
# 创建虚拟环境
conda create --name internlm python=3.10
conda activate internlm
# 升级pip
python -m pip install --upgrade pip
## 可选项 配置pip源
pip config set global.index-url https://mirrors.cernet.edu.cn/pypi/web/simple

# 安装pip库 
pip install modelscope==1.9.5
pip install torch==2.0.1
pip install transformers==4.35.2
pip install streamlit==1.24.0
pip install sentencepiece==0.1.99
pip install accelerate==0.24.1
```



### 2. 模型下载

#### 2.1 使用modelscope下载

创建一个以下内容的python程序，然后运行即可

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b', cache_dir='/root/model', revision='v1.0.3')
```

#### 2.2 使用huggingface-cli下载

没有安装`huggingface-cli`的执行`pip install -U huggingface_hub`完成安装。

```bash
# 在终端中输入以下命令
huggingface-cli download --resume-download internlm/internlm-chat-7b --local-dir 'your_path'
```



### 3. Demo演示

#### 3.1 InternLM对话机器人-终端演示

```bash
cd homework/course_2
python terminal_demo.py
```

#### 3.2 InternLM对话机器人-Web演示

```bash
cd homework/course_2
streamlit run web_demo.py --server.address 127.0.0.1 --server.port 6006
```

![故事](images/故事.png)

#### 3.3 Lagent演示

##### 3.3.1 代码准备

```bash
cd homework/course_2
git clone https://gitee.com/internlm/lagent.git
cd lagent
git checkout 511b03889010c4811b1701abb153e02b8e94fb5e
pip install -e .
```

##### 3.3.2 运行代码

```bash
cd homework/course_2
streamlit run lagent_demo.py --server.address 127.0.0.1 --server.port 6006
```

