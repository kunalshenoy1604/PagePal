from pathlib import Path
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Function to generate audio from text
def generate_audio(input_text, file_path):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=input_text
    )
    response.stream_to_file(file_path)

# Function to get vectorstore from URL
def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store

# Function to create context retriever chain
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain
    
# Function to create conversational RAG chain
def get_conversational_rag_chain(retriever_chain): 
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Function to get response based on user input
def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']

# App title and configuration
st.set_page_config(page_title="Langchain Chat", page_icon=":speech_balloon:")
st.title("Welcome to PagePal ")

# Sidebar for settings
with st.sidebar:
    st.header("Website Link Input")
    website_url = st.text_input("Website URL")

# Main content area
if website_url is None or website_url == "":
    st.info("Please enter a website URL")
else:
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    # User input
    user_query = st.text_input("You:", key="user_input")
    if st.button("Send", key="send_button"):
        if user_query:
            # Get response
            response_text = get_response(user_query)

            # Append user query to chat history
            st.session_state.chat_history.append(HumanMessage(content=user_query))

            # Generate audio file for the response
            speech_file_path = Path(__file__).parent / "response.mp3"
            generate_audio(response_text, speech_file_path)

            # Append text response and audio response to chat history
            st.session_state.chat_history.append(AIMessage(content=response_text))
            st.session_state.chat_history.append(AIMessage(content="Audio response"))
            st.audio(str(speech_file_path), format='audio/mp3')

    # Display chat history
    st.subheader("Chat History")
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.container():
                st.markdown("🤖")
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.container():
                st.markdown("👤")
                st.write(message.content)






