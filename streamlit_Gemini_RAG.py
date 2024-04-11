import os
import io
import json
import tempfile
import pandas as pd
from pathlib import Path
from pprint import pprint
import streamlit as st
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import textwrap
from langchain_community.document_loaders import JSONLoader
import urllib
import warnings
from pathlib import Path as p


from langchain import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma




warnings.filterwarnings("ignore")

file_path='iCliniq.json'
data = json.loads(Path(file_path).read_text())
data1 = json.loads(Path('GenMedGPT-5k.json').read_text())
data = data + data1

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
import os
GOOGLE_API_KEY='YOUR API KEY'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name = "gemini-pro")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
context = "\n\n".join(str(data[i]) for i in range(len(data)))
texts = text_splitter.split_text(context)

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)

vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":5})

def generate_answer(query):
  relevant_documents = vector_index.get_relevant_documents(query)
  prompt_template = """As a AI healthcare professional, you have to give suggestions at any cost to the user, the answer should be detailed and mention some medicine(if possibel not necessary)  also.
    You must give his/her next steps, how to recover from this things."

Context: The user has shared the following information about their situation: {context}.

Question: The user is asking: {question}.

Answer:
"""

  prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


  stuff_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
  stuff_answer = stuff_chain(
    {"input_documents": relevant_documents, "question":query}, return_only_outputs = True
    )
  return stuff_answer['output_text']



st.title('Personalized AI Chat Doctor')

if 'messages' not in st.session_state:
    st.session_state.messages = []
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])



# Chat input for health issues
if prompt := st.chat_input("Enter your health issues"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)



if prompt is not None:
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        full_response += generate_answer(prompt)
        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    #st.chat_message("assistant").write(full_response)



