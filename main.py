from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_chat import message
import asyncio
import streamlit as st

def load_T5():
    llm_t5 = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":64})
    return llm_t5

st.set_page_config(page_title="Crisis Facts: Global Edition", page_icon=":shark:", layout="wide", initial_sidebar_state="expanded")
col1, col2, col3 = st.columns([6,14,6])

with col1:
    st.write("")

with col2:
    st.image("logo.png")

with col3:
    st.write("")

left_column, right_column = st.columns(2)
llm_t5 = load_T5() 
with left_column:
    st.subheader("Please Enter the Context:")
    input_text = st.text_area(label="Crisis Context Input", height=500,label_visibility='collapsed', placeholder="Context...", key="context_input")
    def get_text():
        return input_text

context_input = get_text()

with right_column:
    st.subheader("Please Enter the Query:")
    input_query = st.text_input(label="Query Input",label_visibility='collapsed', placeholder="Query...", key="query_input")
    
    def get_query():
        return input_query
    query_text = get_query()

    def get_t5_output(texts):
        with st.spinner(f"Running T5..."):
            embeddings = HuggingFaceEmbeddings()
            docsearch = Chroma.from_texts(texts, embeddings)
            qa = RetrievalQA.from_chain_type(llm=llm_t5, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}),input_key="question")
            output_t5 = qa.run(query_text)
            return output_t5
    
    def get_openai_output(texts):
        with st.spinner(f"Running OpenAI..."):
            openai_embeddings = OpenAIEmbeddings()
            openai_docsearch = Chroma.from_texts(texts, openai_embeddings)
            qa_openai = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0), chain_type="stuff", retriever=openai_docsearch.as_retriever(search_kwargs={"k": 1}),input_key="question")
            output_openai = qa_openai.run(query_text)
            return output_openai
           

    st.markdown("### Output:")
    t5_selected = st.checkbox("Google Flan-T5", value=True)
    openai_selected = st.checkbox("OpenAI text-davinci-003", value=True)
    get_answer_button = st.button("Click for answer", type='secondary')
    t5_col,openai_col = st.columns([5,5])
    with t5_col:
        st.subheader('_Output from :red[T5]:_')

    with openai_col:
        st.subheader('_Output from :red[OpenAI]:_ ')

    if get_answer_button:
        if(len(context_input)>3000):
            st.error('Please enter smaller context.', icon="🚨")
        if(len(query_text)==0):
            st.error('Query is empty.', icon="🚨")
        if(len(context_input)==0):
            st.error('Context is empty.', icon="🚨")
        if not(openai_selected or t5_selected):
            st.error('Please select atleast 1 Model.', icon="🚨")

        if context_input and query_text:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=200,length_function = len)
            texts = text_splitter.split_text(context_input)
            with t5_col:
                if(t5_selected):
                    output_t5 = get_t5_output(texts)
                    st.write(output_t5)
            with openai_col:
                if(openai_selected):
                    output_openai = get_openai_output(texts)
                    st.write(output_openai)
        






