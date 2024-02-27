
from langchain_community.document_loaders import UnstructuredHTMLLoader
from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from os import environ

import logging

OPENAI_API_KEY = environ.get("OPENAI_API_KEY")
GITHUB_URL_LANGCHAIN = r"https://github.com/langchain-ai/langchain/releases/tag/"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LangChain Demo")

def init():
    """read a local file `tags.md`. line by line, save as a list"""
    url_tags = []
    with open("tags.md", "r") as f:
        lines = f.readlines()
        import math
        # LIMIT = math.inf
        LIMIT = 20
        count = 0
        for line in lines:
            if count > LIMIT:
                break
            count += 1
            tag = line.strip()
            url_tags.append(f"{GITHUB_URL_LANGCHAIN}" + tag)
    return url_tags
    


def retrieve():
    loader = UnstructuredHTMLLoader()

def get_promps():
    pass

def llm():
    pass


def temp():
    vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4-turbo-preview")

    chain = ( 
            { "context": retriever, "question": RunnablePassthrough()}
            | prompt 
            | model
            | StrOutputParser()
             
             )
    
    result = chain.invoke("where did harrison work?")
    logger.info(result)



def main():
    logger.info(f"default key {OPENAI_API_KEY}")
    urls = init()
    # logger.info(f"found tags:\n{urls}")
    temp()



if __name__ == "__main__":
    main()