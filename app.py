import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from fake_review_detection import run_fake_review_detection 

# Load environment variables
load_dotenv(dotenv_path='.env.example')
  # Or '.env.example' only if it's your actual env file
def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """Split text into smaller chunks for processing."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    """Create a FAISS vector store from text chunks."""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Create a conversation chain for handling user queries."""
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_user_query(conversation_chain, user_question):
    """Process user query and display the conversation."""
    response = conversation_chain({'question': user_question})
    answer = response.get('result', "No answer could be generated.")
    chat_history = response.get('chat_history', [])

    # Display chat history
    for i, message in enumerate(reversed(chat_history)):
        if i % 2 == 0:
            st.write(f"User: {message.content}")
        else:
            st.write(f"AI: {message.content}")

    return answer


def main():
    """Streamlit app main function."""
    load_dotenv()
    st.set_page_config(page_title="Lokesh Kollepara | AI/ML Engineer", page_icon=":books:", layout="wide")

    # Store page state in session_state
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    # Store uploaded files and conversation state in session_state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = None

    # Home Page
    if st.session_state.page == 'home':
        st.markdown("""
        <div style="text-align: center; background-color: #204399; padding: 20px; border-radius: 10px;">
            <h1 style="color: white; margin-bottom: 10px;">Lokesh Kollepara</h1>
            <p style="color: white; font-size: 18px;">AI/ML Engineer | Data Enthusiast</p>
            <p style="color: white;">Email: <a href="mailto:kolleparalokesh@gmail.com" style="color: #87CEEB;">kolleparalokesh@gmail.com</a></p>
            <p style="color: white;">Phone: +1-508-714-9704</p>
            <p><a href="https://www.linkedin.com/in/lokesh-kollepara-4698b8227/" target="_blank" style="color: white; background-color: #1DA1F2; padding: 10px 15px; border-radius: 5px; text-decoration: none;">LinkedIn Profile</a></p>
        </div>
        """, unsafe_allow_html=True)

        # New section for Try out projects with no background color but with font color
        st.markdown("""
        <h2 style="text-align: center; color: #204399; padding: 10px 0; margin-top: 20px;">Try out my projects here</h2>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Align buttons horizontally in the center with equal space from the edge
        col1, col2, col3 = st.columns([2,2,1])

        # Add the following content after the buttons:
        st.markdown("""
        <h2 style="color: #204399;">Work Experience</h2>

        

        <h3>AI/ML Engineer Intern, Trilio Inc.</h3>
        <p><em>Jul 2024 - Dec 2024 | Framingham, MA</em></p>
        <ul>
            <li>Performed in-depth analysis on large-scale cloud log data, identifying usage patterns and trends to improve resource optimization and performance monitoring.</li>
            <li>Implemented anomaly detection using ML algorithms like k-means clustering and Isolation Forests, achieving 92% accuracy in detecting irregular patterns.</li>
            <li>Developed a Retrieval-Augmented Generation (RAG) model using OpenAI GPT-4 for real-time log query handling, selecting GPT-4 for deployment after evaluating multiple LLMs.</li>
            <li>Deployed the anomaly detection and RAG systems on AWS infrastructure, leveraging Lambda, DynamoDB, and SageMaker for seamless integration and scalability.</li>
            <li>Integrated Pinecone as a vector database to enable fast, scalable document retrieval using embeddings for similarity search, ensuring efficient query handling.</li>
        </ul>

        <h3>Data Engineer, Tech Cloud Limited</h3>
        <p><em>Nov 2023 - Feb 2024 | Bengaluru, India</em></p>
        <ul>
            <li>Integrated data from various departmental modules into a centralized data warehouse on AWS Redshift, enabling real-time data flow and reporting.</li>
            <li>Designed and implemented efficient ETL processes using AWS Glue, ensuring data consistency and accuracy for business intelligence analysis.</li>
            <li>Developed data models on Redshift and optimized SQL queries for real-time analytics dashboards and reports, supporting decision-making for manufacturing operations.</li>
            <li>Deployed and managed data pipeline infrastructure on AWS using Redshift for storage and processing, alongside S3 for data storage.</li>
            <li>Collaborated with the development team to identify data requirements and produced automated reports and data insights, reducing operational costs for manufacturing clients.</li>
        </ul>

        <h2 style="color: #204399;">Projects</h2>
        <ul>
            <li><strong>Smart Travel Planning</strong>: Developed an itinerary chatbot using GPT-3.5, LLaMA2, Wizard LLM models, LangChain for NLP, and RAG architecture with Cassandra.</li>
            <li><strong>AI-Driven Stress Analysis and Management</strong>: Designed an AI-driven stress analysis and management system using ChatGPT and NoSQL databases via API connections.</li>
            <li><strong>SecureNet</strong>: Created a PySpark-based network intrusion detection system utilizing machine learning and advanced feature engineering for enhanced cybersecurity.</li>
        </ul>

        <h2 style="color: #204399;">Certifications</h2>
        <ul>
            <li><strong>Generative AI with Large Language Models</strong> by OpenAI (Coursera)</li>
            <li><strong>Machine Learning Specialization</strong> by Stanford University (Coursera)</li>
            <li><strong>Full Stack Development</strong> by AlgoExpert</li>
        </ul>

        <h2 style="color: #204399;">Technical Skills</h2>
        <ul>
            <li><strong>Languages:</strong> Python, Java, JavaScript</li>
            <li><strong>Frameworks:</strong> React.js, Node.js, Express.js, Flask</li>
            <li><strong>Cloud/Databases:</strong> AWS Cloud, MySQL, MongoDB, RedShift</li>
            <li><strong>AI Technologies:</strong> Generative AI (RAG Development), Machine Learning, Deep Learning</li>
        </ul>

        <h2 style="color: #204399;">Research Publications</h2>
        <ul>
            <li>
                The research paper titled "PySpark-Powered ML Models for Accurate Spam Detection in Messages" was submitted at the 2023 2nd International Conference on Futuristic Technologies (INCOFT) Karnataka, India. 
                <a href="https://ieeexplore.ieee.org/document/10425231" target="_blank">Link</a>
            </li>
            <li>
                The research paper titled "Detecting Fraudulent Reviews in E-commerce Platforms" was submitted at the 2024 15th International Conference on Computing Communication and Networking Technologies (ICCCNT). 
                <a href="https://ieeexplore.ieee.org/document/10724352" target="_blank">Link</a>
            </li>
        </ul>

        <h2 style="color: #204399;">Education</h2>
        <p><strong>Bachelor of Computer Science with specialization in Artificial Intelligence (2021 - 2024)</strong></p>
        <p>Amrita School of Engineering, Bengaluru, India</p>
        """, unsafe_allow_html=True)
        with col1:
            if st.button("AskYourDocs"):
                st.session_state.page = 'chat_pdf'
                st.experimental_rerun()
        with col2:
            if st.button("AskYourTable"):
                st.session_state.page = 'chat_csv'
                st.experimental_rerun()
        with col3:
            if st.button("Fake Review Detection"):
                st.session_state.page = 'fake_review'
                st.experimental_rerun()

    # Chat with PDF Page
    elif st.session_state.page == 'chat_pdf':
        st.header("Chat with your PDFs :books:")

        if st.button("Home"):
            st.session_state.page = 'home'
            st.experimental_rerun()

        uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files

        if st.session_state.uploaded_files and st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(st.session_state.uploaded_files)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

        if "conversation" in st.session_state and st.session_state.conversation:
            user_question = st.text_input("Ask a question about the PDF(s):")
            if user_question:
                with st.spinner("Generating response..."):
                    answer = handle_user_query(st.session_state.conversation, user_question)
                    st.write(f"Answer: {answer}")

    # Chat with CSV Page
    elif st.session_state.page == 'chat_csv':
        st.header("Chat with your CSVs :bar_chart:")

        if st.button("Home"):
            st.session_state.page = 'home'
            st.experimental_rerun()

        csv_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
        if csv_files:
            st.session_state.uploaded_files = csv_files

        if st.session_state.uploaded_files and st.button("Process CSVs"):
            with st.spinner("Processing CSVs..."):
                raw_text = "".join(file.read().decode('utf-8') for file in st.session_state.uploaded_files)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

        if "conversation" in st.session_state and st.session_state.conversation:
            user_question = st.text_input("Ask a question about the CSV(s):")
            if user_question:
                with st.spinner("Generating response..."):
                    answer = handle_user_query(st.session_state.conversation, user_question)
                    st.write(f"Answer: {answer}")

    elif st.session_state.page == 'fake_review':
        st.header("Fake Review Detection")
        if st.button("Home"):
            st.session_state.page = 'home'
            st.experimental_rerun()

        # Call the functionality from the imported module
        run_fake_review_detection()


# Run the main function
if __name__== "__main__":
    main()
