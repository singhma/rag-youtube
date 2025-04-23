
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def load_transcript(video_id: str) -> str:
    try:
        # If you don’t care which language, this returns the “best” one
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        # Flatten it to plain text
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript

    except TranscriptsDisabled:
        print("No captions available for this video")

def perform_chunking(transcript, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap = chunk_overlap)
    chunks = text_splitter.create_documents([transcript])
    return chunks

def store_embedding_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vectore_store = FAISS.from_documents(chunks,embeddings)
    return vectore_store
