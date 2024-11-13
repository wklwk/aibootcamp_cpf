######################
# Import packages
######################

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import streamlit as st
from openai import OpenAI

# Common imports
import os
from dotenv import load_dotenv
import json
import tiktoken

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

import hmac

######################
# Paswword
######################
def check_password():  
    """Returns `True` if the user had the correct password."""  
    def password_entered():  
        """Checks whether a password entered by the user is correct."""  
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):  
            st.session_state["password_correct"] = True  
            del st.session_state["password"]  # Don't store the password.  
        else:  
            st.session_state["password_correct"] = False  
    # Return True if the passward is validated.  
    if st.session_state.get("password_correct", False):  
        return True  
    # Show input for password.  
    st.text_input(  
        "Password", type="password", on_change=password_entered, key="password"  
    )  
    if "password_correct" in st.session_state:  
        st.error("😕 Password incorrect")  
    return False 
# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()

##############
#Interface
##############
st.title("CPF Healthcare Financing")
"""
In this page, you can ask about healthcare financing related to CPF.

Enter your question below:

"""

##################
# Prompting
##################
def get_user_prompt():
    return st.chat_input(placeholder="Can I use my Medisave for hospitalisation?")


######################
# Loading API
######################
# Load the environment variables
# If the .env file is not found, the function will return `False
load_dotenv('.env')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")

client = OpenAI(api_key=OPENAI_API_KEY)

def count_tokens(text):
    encoding = tiktoken.encoding_for_model('gpt-4o-mini')
    return len(encoding.encode(text))

def count_tokens_from_message_rough(messages):
    encoding = tiktoken.encoding_for_model('gpt-4o-mini')
    value = ' '.join([x.get('content') for x in messages])
    return len(encoding.encode(value))

##################################
# Setting up Credential LangChain
#################################
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# embedding model that we will use for the session
embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')

# llm to be used in RAG pipeplines in this notebook
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, seed=42)

######################
# RAG
######################
loader1 = WebBaseLoader(["https://www.cpf.gov.sg/member/healthcare-financing/using-your-medisave-savings/using-medisave-for-outpatient-treatments"])
loader2 = WebBaseLoader(["https://www.cpf.gov.sg/member/healthcare-financing/medishield-life/what-medishield-life-covers-you-for"])
loader3 = WebBaseLoader(["https://www.cpf.gov.sg/member/healthcare-financing/using-your-medisave-savings/using-medisave-for-hospitalisation"])
loader4 = WebBaseLoader(["https://www.cpf.gov.sg/member/healthcare-financing/careshield-life"])
loader5 = WebBaseLoader(["https://www.cpf.gov.sg/member/healthcare-financing/eldershield"])
loader6 = WebBaseLoader(["https://www.cpf.gov.sg/member/healthcare-financing/medisave-care-for-long-term-care-needs"])


loaderlist = [loader1, loader2, loader3, loader4, loader5, loader6]

list_of_documents_loaded=[]
for loader in loaderlist:
    try: 
        # markdown_path = os.path.join('notes', file)
        # loader = TextLoader(markdown_path)
        docs = loader.load()
        list_of_documents_loaded.extend(docs)
    # except Exception as e:
    except IndexError or ModuleNotFoundError or AttributeError:
        continue

# st.write(list_of_documents_loaded)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=10, length_function=count_tokens)

# Split the documents into smaller chunks
splitted_documents = text_splitter.split_documents(list_of_documents_loaded)

# Create the vector database
vectordb = Chroma.from_documents(
    documents=splitted_documents,
    embedding=embeddings_model,
    collection_name="naive_splitter", # one database can have multiple collections
    persist_directory="./vector_db"
)

rag_chain = RetrievalQA.from_llm(retriever=vectordb.as_retriever(), llm=llm)

def main():
    try:
        user_prompt = get_user_prompt()
        # rag_chain = setup_rag()
        # process_prompt(user_prompt, rag_chain)
        llm_response = rag_chain.invoke(user_prompt)
        st.write(llm_response['result'])
    except TypeError or IndexError or ModuleNotFoundError or AttributeError:
        pass
    
if __name__=="__main__":
    main()
