
import os
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import CharacterTextSplitter
from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from os import environ
from bs4 import BeautifulSoup

from rich.console import Console

import logging

OPENAI_API_KEY = environ.get("OPENAI_API_KEY")
GITHUB_URL_LANGCHAIN = r"https://github.com/langchain-ai/langchain/releases/tag/"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LangChain Demo")

def get_changelogs():
    """read a local file `tags.md`. line by line, save as a list"""
    url_tags = []
    with open("tags.md", "r") as f:
        lines = f.readlines()
        import math
        LIMIT = math.inf
        LIMIT = 100
        count = 0
        for line in lines:
            if count >= LIMIT:
                break
            count += 1
            tag = line.strip()
            url_tags.append(f"{GITHUB_URL_LANGCHAIN}" + tag)

    logger.info("\n".join(url_tags))
    import requests
    collection = []
    # make dir at `./tag_pages/`
    if "tag_pages" not in os.listdir():
        os.mkdir("tag_pages")

    for url in url_tags:

        # first, check `./tag_pages/` to see if the page has already been downloaded.
        # if it has, load the file 
        # if it hasn't, continue to load the web page
        
        tagname = url.split('/')[-1]
        if f"{tagname}.html" in os.listdir("tag_pages"):

            
            with open(f"tag_pages/{tagname}.html", "r") as f:
                content = f.read()
                # do data cleansing...
                content = BeautifulSoup(content, "html.parser").get_text()
                # remove leading and trailing whitespaces
                content = content.strip()
                # remove duplicated line breaks: `\n`
                content = "\n".join([line for line in content.split("\n") if line.strip() != ""])
                logger.info(content[:100])
                collection.append(content)
            
            logger.info(f"[cached]loading content from {url}")
            
        else: 
            
            logger.info(f"loading content from remote: {url}")
            response = requests.get(url)
            
            # convert html to plain text, using `BeautifulSoup`
            soup = BeautifulSoup(response.text, "html.parser")

            # write the HTML content to a file at `./tag_pages/`
            with open(f"tag_pages/{tagname}.html", "w") as f:
                f.write(soup.prettify())

            plain_text = soup.get_text()
            logger.info(f"plain_text: {plain_text}")
            collection.append(plain_text)

    return collection
    
def main():

    content = get_changelogs()
    
    logger.debug(f"content: {content}")

    logger.info("preparing vectorstore ... ")
    if False and "faiss.index" in os.listdir():
        from langchain.vectorstores.utils import DistanceStrategy
        logger.info(f"Found existed vectorstore.")
        vectorstore = FAISS.load_local( "./faiss.index", OpenAIEmbeddings(), distance_strategy=DistanceStrategy.COSINE)
    else:

        embedding = OpenAIEmbeddings()
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=32)
        docs = []
        for item in content:
            docs.extend(text_splitter.split_text(item))
        # docs = [ text_splitter.split_text(item) for item in content ]
        vectorstore = FAISS.from_texts(
            docs, embedding=embedding
        )
        vectorstore.save_local("faiss.index")

    QUESTION = """what is the major system design changes in these versions?"""

    # retreived_docs = vectorstore.similarity_search(QUESTION)

    # logger.info(f"retreived_docs: {retreived_docs[0].page_content}")

    retriever = vectorstore.as_retriever()
    # r_docs = retriever.invoke(QUESTION)
    # logger.info(f"r_docs: {r_docs[0].page_content}")

    from langchain import hub

    prompt = hub.pull("rlm/rag-prompt")


    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4-turbo-preview")


    def format_docs(docs):
        logger.info(f"docs to format : {len(docs)}")
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    result = rag_chain.invoke(QUESTION)


    print(result)

    

    

if __name__ == "__main__":
    
    assert OPENAI_API_KEY is not None, "OpenAI API key is not set"

    
    main()