#lang chain 应用
#ollama pull llama2:7b-chat

#pip install arxiv langchain_community langchain gpt4all qdrant-client gradio

import os
import time
import arxiv
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings

# 创建目录
dirpath = "arxiv_papers"
if not os.path.exists(dirpath):
    os.makedirs(dirpath)

#在arxiv上搜索LLM
client = arxiv.Client()
search = arxiv.Search(
    query="LLM",
    max_results=10,
    sort_order=arxiv.SortOrder.Descending
)

# 保存和下载相关论文
for result in client.results(search):
    while True:
        try:
            result.download_pdf(dirpath=dirpath)
            print(f"-> Paper id {result.get_short_id()} with title '{result.title}' is downloaded.")
            break
        except (FileNotFoundError, ConnectionResetError) as e:
            print("Error occurred:", e)
            time.sleep(5)

#从目录中加载论文
papers = []
loader = DirectoryLoader(dirpath, glob="./*.pdf", loader_cls=PyPDFLoader)
try:
    papers = loader.load()
except Exception as e:
    print(f"Error loading file: {e}")
print("Total number of pages loaded:", len(papers)) 

#将所有页面的内容连接成一个字符串
full_text = ''
for paper in papers:
    full_text += paper.page_content

#删除空行并将行连接成单个字符串
full_text = " ".join(line for line in full_text.splitlines() if line)
print("Total characters in the concatenated text:", len(full_text)) 

# 将文本分成小块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
paper_chunks = text_splitter.create_documents([full_text])

# Create Qdrant 创建象限张量存储
qdrant = Qdrant.from_documents(
    documents=paper_chunks,
    embedding=GPT4AllEmbeddings(),
    path="./tmp/local_qdrant",
    collection_name="arxiv_papers",
)
retriever = qdrant.as_retriever()

# 定义提示模板
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

#初始化Ollama LLM
ollama_llm = "llama2:7b-chat"
model = ChatOllama(model=ollama_llm)

# 定义处理链
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)

# 添加输入类型
class Question(BaseModel):
    __root__: str

# 将输入类型应用于链
chain = chain.with_types(input_type=Question)
result = chain.invoke("Explain about Vision Enhancing LLMs")
print(result)