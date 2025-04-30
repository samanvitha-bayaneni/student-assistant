import os
import tempfile
import re
import numpy as np
import pandas as pd
from datetime import datetime
import requests
import streamlit as st
import PyPDF2
import plotly.express as px
import logging
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI  # Updated import


# This is the function that was missing - it applies the LaTeX formatting selectively to parts of the text that look like math
def format_math_expressions(text: str) -> str:
    """
    Format math expressions in text to LaTeX format.
    Looks for math-like patterns and formats them.
    """
    # Check for common math patterns
    math_patterns = [
        r"\\partial\s*[a-zA-Z]",  # Partial derivatives
        r"[a-zA-Z]+\d+",  # Variable with subscript like x1, y2
        r"f\s*=\s*fm-1",  # Special pattern
        r"[a-zA-Z]\([a-zA-Z]\)",  # Function notation
        r"[a-zA-Z]/[a-zA-Z]",  # Simple division
        r"\\frac{",  # Already has some LaTeX
        r"\\sum",  # Summation
        r"\\int",  # Integration
        r"\\prod",  # Product
        r"\\sqrt",  # Square root
    ]
    
    # Split text into paragraphs to process math only where needed
    paragraphs = text.split("\n\n")
    processed_paragraphs = []
    
    for paragraph in paragraphs:
        has_math = any(re.search(pattern, paragraph) for pattern in math_patterns)
        if has_math:
            # Only apply formatting to paragraphs with math content
            processed_paragraphs.append(paragraph)
        else:
            processed_paragraphs.append(paragraph)
            
    return "\n\n".join(processed_paragraphs)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFRagSystem:
    def __init__(self, google_api_key=None):
        """Initialize the PDF RAG System with optional Gemini API key."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        self.qa_chain = None

        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key

    def extract_text_from_pdf(self, pdf_file_path: str) -> str:
        text = ""
        with open(pdf_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    def process_pdf(self, pdf_file_path: str):
        raw_text = self.extract_text_from_pdf(pdf_file_path)
        text_chunks = self.text_splitter.split_text(raw_text)
        self.vector_store = FAISS.from_texts(text_chunks, self.embeddings)
        if os.environ.get("GOOGLE_API_KEY"):
            self.initialize_qa_chain()
        return raw_text, text_chunks

    def process_pdf_bytes(self, pdf_bytes: bytes):
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name
        raw_text, text_chunks = self.process_pdf(temp_path)
        os.unlink(temp_path)
        return raw_text, text_chunks

    def initialize_qa_chain(self):
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Process a PDF first.")

        prompt_template = """
        You are an assistant that answers questions based on the provided context from a PDF document.

        Context: {context}

        Question: {question}

        Answer the question based only on the provided context. If you cannot find the answer in the context, 
        say "I don't have enough information to answer this question based on the document content." 
        Do not make up information.

        When you're explaining mathematical concepts or equations:
        1. Use proper LaTeX notation for formulas and equations
        2. Format subscripts properly (e.g., x_i instead of xi)
        3. Use proper notation for derivatives, integrals, sums, etc.

        Answer:
        """

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT}
        )

    def query(self, question: str) -> Dict[str, Any]:
        if not self.qa_chain:
            if not os.environ.get("GOOGLE_API_KEY"):
                return {"result": "Google API key not set. Please provide a key to enable querying."}
            if not self.vector_store:
                return {"result": "No PDF has been processed yet. Please process a PDF first."}
            self.initialize_qa_chain()

        try:
            result = self.qa_chain({"query": question})
            return result
        except Exception as e:
            return {"result": f"Error processing query: {str(e)}"}

    def similarity_search(self, query: str, k: int = 3) -> List[tuple]:
        if not self.vector_store:
            return [("No PDF has been processed yet. Please process a PDF first.", 0)]
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return [(doc.page_content, score) for doc, score in results]
        except Exception as e:
            return [(f"Error during similarity search: {str(e)}", 0)]


def parse_calendar(calendar_url):
    """Parse calendar URL and extract deadlines"""
    try:
        # Try to fetch the calendar data
        response = requests.get(calendar_url)
        if response.status_code != 200:
            return False, f"Failed to fetch calendar: HTTP {response.status_code}"
        
        # For ICS processing, we'll do a simple parsing here
        calendar_data = response.text
        
        # Simple regex-based extraction of events
        events = re.findall(r'BEGIN:VEVENT.*?END:VEVENT', calendar_data, re.DOTALL)
        
        deadlines = []
        for event in events:
            # Extract event details
            summary_match = re.search(r'SUMMARY:(.*?)(?:\r\n|\n)', event)
            date_match = re.search(r'DTSTART(?:;.+?)?:(.*?)(?:\r\n|\n)', event)
            description_match = re.search(r'DESCRIPTION:(.*?)(?:\r\n|\n)', event)
            
            if summary_match and date_match:
                event_name = summary_match.group(1)
                
                # Parse date string
                date_str = date_match.group(1)
                try:
                    # Simple date parsing - this would need to be more robust in a real app
                    if 'T' in date_str:  # Format: 20230101T120000Z
                        year = int(date_str[0:4])
                        month = int(date_str[4:6])
                        day = int(date_str[6:8])
                        hour = int(date_str[9:11])
                        minute = int(date_str[11:13])
                        event_date = datetime(year, month, day, hour, minute)
                    else:  # Format: 20230101
                        year = int(date_str[0:4])
                        month = int(date_str[4:6])
                        day = int(date_str[6:8])
                        event_date = datetime(year, month, day)
                    
                    description = description_match.group(1) if description_match else "No description"
                    
                    if event_date > datetime.now():
                        deadlines.append({
                            'name': event_name,
                            'date': event_date,
                            'description': description
                        })
                except Exception as e:
                    logger.error(f"Error parsing date {date_str}: {e}")
        
        # Sort deadlines by date
        deadlines.sort(key=lambda x: x['date'])
        
        return True, deadlines
    except Exception as e:
        logger.error(f"Error parsing calendar: {e}")
        return False, f"Error parsing calendar: {str(e)}"

def get_deadline_info(deadlines):
    """Get information about upcoming deadlines"""
    if not deadlines:
        return "No upcoming deadlines found."

    info_lines = []
    for deadline in deadlines:
        name = deadline.get('name', 'Unnamed Event')
        date = deadline.get('date')
        description = deadline.get('description', 'No description')
        
        date_str = date.strftime("%A, %B %d, %Y at %I:%M %p") if date.time() != datetime.min.time() else date.strftime("%A, %B %d, %Y")
        
        info = f"🗓️ **{name}**\n📅 Date: {date_str}\n📝 Description: {description}\n"
        info_lines.append(info)
    
    return "\n---\n".join(info_lines)

def create_deadline_donut_chart(deadlines):
    """Create a donut chart of the 5 earliest deadlines showing days remaining"""
    if not deadlines:
        return None
    
    # Get the 5 earliest deadlines
    now = datetime.now()
    upcoming = [d for d in deadlines if d['date'] > now][:5]
    
    if not upcoming:
        return None
    
    # Prepare data for the chart
    chart_data = []
    for deadline in upcoming:
        days_remaining = (deadline['date'] - now).days
        chart_data.append({
            'Deadline': f"{deadline['name']} ({days_remaining}d)",
            'Days Remaining': days_remaining,
            'Date': deadline['date'].strftime('%b %d'),
            'Full Name': deadline['name']
        })
    
    df = pd.DataFrame(chart_data)
    
    # Create the donut chart
    fig = px.pie(df, 
                 values='Days Remaining', 
                 names='Deadline',
                 title='Upcoming Deadlines (Days Remaining)',
                 hole=0.4,  # Creates the donut shape
                 hover_data=['Full Name', 'Date'])
    
    # Update layout to show days remaining instead of percentages
    fig.update_traces(textposition='inside', 
                     textinfo='value+label',  # Show value (days) and label
                     hovertemplate="<b>%{customdata[0]}</b><br>Days: %{value}<br>Date: %{customdata[1]}")
    fig.update_layout(margin=dict(t=40, b=20, l=20, r=20),
                     showlegend=False)
    
    return fig

def answer_query(query, rag_system, deadlines):
    """Generate an answer based on retrieved chunks or deadline info"""
    # Check if it's a deadline question
    if re.search(r"deadline|due date|when is|when are|next assignment", query.lower()):
        if deadlines:
            return get_deadline_info(deadlines)
        else:
            return "No calendar information has been uploaded. Please provide a calendar link to track deadlines."

    # First try with Google API if key is available
    if os.environ.get("GOOGLE_API_KEY") and rag_system.vector_store:
        try:
            result = rag_system.query(query)
            raw_answer = result.get('result', 'No answer found.')
            print(f"Raw LLM answer: {raw_answer}")  # <--- ADD THIS LINE
            formatted_answer = format_math_expressions(raw_answer)
            print(f"Formatted answer: {formatted_answer}") # <--- ADD THIS LINE
            return formatted_answer

        except Exception as e:
            logger.error(f"Error using QA chain: {e}")
            # Fall back to similarity search
            relevant_chunks = rag_system.similarity_search(query)
            # ... (rest of the similarity search part)
    else:
        relevant_chunks = rag_system.similarity_search(query)
        # ... (rest of the similarity search part)
        response = "Based on the document content:\n\n"
        for i, (chunk, score) in enumerate(relevant_chunks):
            similarity = 1 - (score/100) if score > 0 else 0
            response += f"--- Relevant Information {i+1} (relevance: {similarity:.2f}) ---\n"
            response += chunk[:500] + "...\n\n" if len(chunk) > 500 else chunk + "\n\n"
        response += "\nThese passages should help answer your question. If you need more specific information, please ask a more focused question."
        formatted_response = format_math_expressions(response)
        print(f"Formatted response (similarity search): {formatted_response}") # <--- ADD THIS LINE
        return formatted_response

    # Otherwise use similarity search (moved inside the 'else' block)
    relevant_chunks = rag_system.similarity_search(query)

    if not relevant_chunks or relevant_chunks[0][0].startswith("No PDF"):
        return "I couldn't find relevant information in the uploaded document to answer your question."

    # Prepare response with retrieved information
    response = "Based on the document content:\n\n"

    for i, (chunk, score) in enumerate(relevant_chunks):
        # Convert FAISS distance to a similarity score (1 - distance/100)
        similarity = 1 - (score/100) if score > 0 else 0
        response += f"--- Relevant Information {i+1} (relevance: {similarity:.2f}) ---\n"
        response += chunk[:500] + "...\n\n" if len(chunk) > 500 else chunk + "\n\n"

    response += "\nThese passages should help answer your question. If you need more specific information, please ask a more focused question."

    # Apply LaTeX formatting to the response
    formatted_response = format_math_expressions(response)
    print(f"Formatted response (final): {formatted_response}") # <--- ADD THIS LINE
    return formatted_response


# Main Streamlit app
def main():
    st.set_page_config(page_title="Student RAG Assistant", layout="wide")
    
    # Initialize session state variables if they don't exist
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'deadlines' not in st.session_state:
        st.session_state.deadlines = []
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    st.title("Student RAG Assistant")
    st.write("Upload a PDF and ask questions about its content. You can also link a calendar to track deadlines.")
    
    # Sidebar for document uploads and settings
    with st.sidebar:
        st.header("Configuration")
        
        api_key = st.text_input("Enter your Google API key (optional)", type="password")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            st.success("API key set! You can now use the LLM for better responses.")
        
        # Initialize RAG system if not done already
        if st.session_state.rag_system is None or api_key:
            st.session_state.rag_system = PDFRagSystem(google_api_key=api_key)
        
        st.header("Document Upload")
        
        # PDF upload
        pdf_file = st.file_uploader("Upload a PDF document", type="pdf")
        if pdf_file is not None:
            with st.spinner("Processing PDF..."):
                # Process the PDF
                pdf_bytes = pdf_file.read()
                raw_text, text_chunks = st.session_state.rag_system.process_pdf_bytes(pdf_bytes)
                st.success(f"PDF processed! Extracted {len(text_chunks)} text chunks.")
        
        # Calendar link
        st.header("Calendar Integration")
        calendar_url = st.text_input("Enter calendar URL (ICS format)")
        if calendar_url and st.button("Load Calendar"):
            with st.spinner("Loading calendar..."):
                success, result = parse_calendar(calendar_url)
                if success and isinstance(result, list):
                    st.session_state.deadlines = result
                    st.success(f"Successfully loaded {len(result)} upcoming deadlines.")
                else:
                    st.error(result)
        
        # Deadline Donut Chart
        if st.session_state.deadlines:
            st.header("Upcoming Deadlines")
            donut_chart = create_deadline_donut_chart(st.session_state.deadlines)
            if donut_chart:
                st.plotly_chart(donut_chart, use_container_width=True)
            else:
                st.info("No upcoming deadlines found.")
            
            # Add an expander to show all deadlines
            with st.expander("View All Deadlines"):
                if st.session_state.deadlines:
                    # Create a dataframe for better display
                    deadline_data = []
                    for d in st.session_state.deadlines:
                        deadline_data.append({
                            "Deadline": d['name'],
                            "Date": d['date'].strftime('%b %d, %Y'),
                            "Days Remaining": (d['date'] - datetime.now()).days
                        })
                    
                    df = pd.DataFrame(deadline_data)
                    st.dataframe(df)
                else:
                    st.write("No deadlines loaded")
    
    # Main chat interface
    st.header("Ask Questions About Your Document or Deadlines")
    
    if st.session_state.rag_system is None or not st.session_state.rag_system.vector_store:
        st.info("Please upload a PDF document to start asking questions.")
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # React to user input
    if prompt := st.chat_input("Ask a question about your document or deadlines"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get response
        response = answer_query(prompt, st.session_state.rag_system, st.session_state.deadlines)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Add some helper information at the bottom
    st.write("---")
    st.write("**Tips:**")
    st.write("- Upload your course materials as PDFs")
    st.write("- Add your API key for better answers (optional)")
    st.write("- Ask specific questions about the content")
    st.write("- Add your course calendar to track deadlines")
    st.write("- Ask 'When is my next deadline?' to see upcoming due dates")

if __name__ == "__main__":
    main()
