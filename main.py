from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_chat import message
import streamlit as st


template = """
    
    QUERY: {query}
    CONTEXT: {context}
    
"""

def load_LLM():
    llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature":0, "max_length":64},huggingfacehub_api_token="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")
    return llm

st.set_page_config(page_title="Crisis Facts", page_icon=":robot:")
st.header("Crisis Facts")

llm = load_LLM()

st.markdown("## Enter Context")

def get_text():
    input_text = st.text_area(label="Crisis Context Input", label_visibility='collapsed', placeholder="Context...", key="context_input")
    return input_text

context_input = get_text()

def get_query():
    input_text = st.text_area(label="Query Input", label_visibility='collapsed', placeholder="Query...", key="query_input")
    return input_text

query_text = get_query()

def get_answer():
    if context_input and query_text:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=200,length_function = len)
        texts = text_splitter.split_text(context_input)
        embeddings = HuggingFaceEmbeddings()
        docsearch = Chroma.from_texts(texts, embeddings)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}),input_key="question")
        output = qa.run(query_text)
        st.write(output)
        st.session_state.past.append(context_input)
        st.session_state.generated.append(output)

st.markdown("### Your Answer:")
st.button("Click for answer", type='secondary', on_click=get_answer())
