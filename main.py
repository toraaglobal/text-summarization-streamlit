import streamlit as st   
from langchain_openai import OpenAI     
from langchain.docstore.document import Document    
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain


def generate_response(txt, openai_api_key):
    llm = OpenAI(
        temperature=0.5,
        openai_api_key=openai_api_key,
    )
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)  

    docs = [
        Document(page_content=text) for text in texts
    ]  

    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
    )
    return chain.run(docs)

## page layout
st.set_page_config(
    page_title="Writing Text Summarization",
    page_icon="🔗",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Text Summarization with OpenAI")

text_input = st.text_area(
    "Enter text to summarize",
    "The quick brown fox jumps over the lazy dog.",
    height=200,
)

result = []

with st.form('summarize_form', clear_on_submit=True):
    openai_api_key = st.text_input("Enter OpenAI API Key", type="password",disabled= not text_input)
    if st.form_submit_button("Summarize") and openai_api_key.startswith("sk-"):
        response = generate_response(text_input, openai_api_key)
        result.append(response)
        del openai_api_key

if len(result) > 0:
    st.info(response)