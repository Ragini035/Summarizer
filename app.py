import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader

# Streamlit App
st.set_page_config(page_title="LangChain: Summarize Website Text", page_icon="🦜")
st.title("🦜 LangChain: Summarize Website Text")
st.subheader('Summarize URL')

# Get the Groq API Key and website URL to be summarized
with st.sidebar:
    groq_api_key = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("Enter Website URL")

# Prompt Template
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content from Website"):
    # Validate all the inputs
    if not groq_api_key.strip():
        st.error("Please provide your Groq API key.")
    elif not generic_url.strip() or not validators.url(generic_url):
        st.error("Please enter a valid website URL.")
    else:
        try:
            with st.spinner("Processing..."):
                # Load the website data
                loader = UnstructuredURLLoader(
                    urls=[generic_url],
                    ssl_verify=False,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                                      "Chrome/116.0.0.0 Safari/537.36"
                    }
                )
                docs = loader.load()

                # Initialize LLM inside the try block
                llm = ChatGroq(model="gemma-2b-it", groq_api_key=groq_api_key)

                # Chain For Summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.error(f"Error: {e}")
