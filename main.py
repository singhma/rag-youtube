from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st

from rag_utilities import *


def main():
    load_dotenv()
    with st.form("my_form"):
    
        st.title("Sample RAG Application")
        st.subheader("Provides ability to query Youtube Videos")
        video_id = st.selectbox("Select Video ID",("jmmW0F0biz0","Gfr50f6ZBvo"))
        user_query = st.text_input("Enter your query")
        
        col1, col2 = st.columns(2, border=True)

        with col1:
            st.write("Select Chunking Parameters")
            chunk_size = st.slider("Chunk Size", min_value=100, max_value=1000, step=100)
            chunk_overlap = st.slider("Chunk Overlap", min_value=100, max_value=200, step=10)

        with col2:
            st.write("Select Retriever Parameters")
            search_type = st.selectbox("Search Type", ["similarity","mmr","similarity_score_threshold"])
            documents = st.slider("Documents to Retrieve", min_value=3, max_value=10)
        submitted = st.form_submit_button("Submit") 

        if submitted:

            transcript = load_transcript(video_id)
            chunks = perform_chunking(transcript, chunk_size, chunk_overlap)
            vectore_store = store_embedding_vector_store(chunks)

            # Retriever
            retriever = vectore_store.as_retriever(search_type=search_type, search_kwargs={"k": documents})

            # Prompt
            prompt = PromptTemplate.from_template(
            """
            You are a helpful assistant.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.
            <context>
            {context}
            </context>
            """)
            llm = OpenAI()
            document_chain = create_stuff_documents_chain(llm,prompt)
            final_chain = create_retrieval_chain (retriever,document_chain)
            response = final_chain.invoke({"input":user_query})
            if response:
                st.header("RAG Response")
                st.success(response['answer'])
                print(response['answer'])
if __name__ == "__main__":
    main()
