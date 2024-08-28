from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)


st.title('Langchain With LLAMA3 API')
input_text = st.text_input("Search the topic you want")

api_key = os.getenv("LANGCHAIN_API_KEY")
if not api_key:
    st.error("API key is not set. Please set the LANGCHAIN_API_KEY environment variable.")
else:
    llm = Ollama(model="llama3")
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    if input_text:
        try:
            response = chain.invoke({"question": input_text})
            st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
