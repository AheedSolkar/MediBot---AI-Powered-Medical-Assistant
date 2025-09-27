import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
import re

load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

# Custom CSS for ChatGPT-like UI
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f7f7f8;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #202123;
        color: white;
    }
    
    /* Chat messages */
    .user-message {
        background-color: #f7f7f8;
        color: #000000; 
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #19c37d;
    }
    
    .assistant-message {
        background-color: #ffffff;
        color: #000000; 
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #10a37f;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Header tabs */
    .header-tabs {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sugar chart container */
    .chart-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        border-radius: 5px;
        background-color: #10a37f;
        color: white;
        border: none;
        padding: 10px;
    }
    
    .stButton button:hover {
        background-color: #0d8c6c;
    }
    
    /* Input styling */
    .stTextInput input {
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    
    /* Sidebar chat items */
    .sidebar-chat-item {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    
    .sidebar-chat-item:hover {
        background-color: #2d2d2d;
    }
    
    .sidebar-chat-item.active {
        background-color: #343541;
    }
</style>
""", unsafe_allow_html=True)

# Constants
EMERGENCY_RESPONSE = """**MEDICAL EMERGENCY NOTICE**

Your query suggests a potential medical emergency. Please:

1. **Call your local emergency number immediately** (911, 112, etc.)
2. **Go to the nearest emergency room**
3. **Do not rely on AI for emergency guidance**

This is a critical situation requiring immediate professional medical attention."""

UNCERTAIN_RESPONSE_TEMPLATE = """I've searched our medical knowledge base, but I couldn't find specific information about this topic in our current resources.

For accurate information about this medical question, I recommend:

• Consulting with a qualified healthcare professional
• Contacting relevant medical specialists
• Reaching out to appropriate medical organizations

I can help with many other medical topics, particularly diabetic foot ulcers. Would you like to ask about something else?"""

# Medical categories for context tracking
MEDICAL_CATEGORIES = {
    'diabetes': ['diabetes', 'diabetic', 'blood sugar', 'insulin', 'glucose', 'a1c', 'hypoglycemia', 'hyperglycemia'],
    'foot_care': ['foot', 'ulcer', 'wound', 'infection', 'neuropathy', 'amputation', 'podiatry'],
    'cardiology': ['heart', 'blood pressure', 'cholesterol', 'cardiac', 'hypertension', 'stroke'],
    'medication': ['medication', 'drug', 'prescription', 'dose', 'side effect', 'treatment'],
    'symptoms': ['pain', 'fever', 'headache', 'symptom', 'swelling', 'redness'],
    'general': ['test', 'diagnosis', 'doctor', 'hospital', 'appointment']
}

class ConversationContext:
    def __init__(self):
        self.current_topic = None
        self.previous_topics = []
        self.follow_up_count = 0
        self.last_medical_query = None
        self.last_medical_response = None
        self.conversation_history = []
    
    def update_context(self, query, response, is_medical=True):
        """Update conversation context based on current interaction"""
        if is_medical:
            current_topic = self.classify_medical_topic(query)
            
            if (current_topic == self.current_topic and 
                self.follow_up_count < 5 and
                len(query.split()) < 15):
                self.follow_up_count += 1
            else:
                if self.current_topic:
                    self.previous_topics.append(self.current_topic)
                self.current_topic = current_topic
                self.follow_up_count = 0
            
            self.last_medical_query = query
            self.last_medical_response = response
        
        self.conversation_history.append({'query': query, 'response': response})
        if len(self.conversation_history) > 3:
            self.conversation_history.pop(0)
    
    def classify_medical_topic(self, query):
        """Classify the medical topic of the query"""
        query_lower = query.lower()
        
        for category, keywords in MEDICAL_CATEGORIES.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        return 'general_health'
    
    def get_context_prompt(self):
        """Generate context prompt for the AI"""
        if not self.current_topic or self.follow_up_count == 0:
            return ""
        
        context_prompt = f"\n\n**Conversation Context:** The user is asking a follow-up question about {self.current_topic}. "
        
        if self.follow_up_count > 0:
            context_prompt += f"This is follow-up number {self.follow_up_count + 1} on this topic. "
        
        if self.last_medical_query and self.follow_up_count < 3:
            context_prompt += f"The previous question was: '{self.last_medical_query}'. "
        
        return context_prompt
    
    def resolve_references(self, query):
        """Resolve pronouns and references in the query"""
        if not self.last_medical_query:
            return query
        
        resolved_query = query
        reference_patterns = {
            r'\bit\b': self.last_medical_query,
            r'\bthat\b': self.last_medical_query,
            r'\bthis\b': self.last_medical_query,
            r'\bthe condition\b': f"the {self.current_topic} condition mentioned previously",
        }
        
        for pattern, replacement in reference_patterns.items():
            if re.search(pattern, query.lower()):
                resolved_query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
                break
        
        return resolved_query

@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return None

def is_emergency_query(query):
    emergency_keywords = [
        'emergency', '911', 'urgent', 'heart attack', 'stroke', 'bleeding',
        'unconscious', 'chest pain', 'difficulty breathing', 'suicide',
        'severe pain', 'allergic reaction', 'poison', 'choking', 'dying',
        'can\'t breathe', 'broken bone', 'seizure', 'fainting', 'overdose'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in emergency_keywords)

def is_conversational_query(query):
    conversational_phrases = [
        'thanks', 'thank you', 'hello', 'hi', 'hey', 'goodbye', 'bye',
        'how are you', 'good morning', 'good afternoon', 'good evening',
        'appreciate it', 'nice', 'great', 'awesome', 'perfect', 'ok', 'okay'
    ]
    query_lower = query.lower().strip()
    return any(phrase == query_lower for phrase in conversational_phrases)

def get_conversational_response(query):
    query_lower = query.lower().strip()
    
    if any(phrase in query_lower for phrase in ['thanks', 'thank you', 'appreciate']):
        return "You're welcome! I'm glad I could help. If you have any other medical questions, feel free to ask."
    
    elif any(phrase in query_lower for phrase in ['hello', 'hi', 'hey']):
        return "Hello! I'm your medical assistant. How can I help you today?"
    
    elif any(phrase in query_lower for phrase in ['how are you']):
        return "I'm functioning well, thank you! I'm here to assist with your medical questions. What would you like to know?"
    
    elif any(phrase in query_lower for phrase in ['goodbye', 'bye', 'see you']):
        return "Goodbye! Take care and remember to consult healthcare professionals for personal medical advice."
    
    elif any(phrase in query_lower for phrase in ['good morning', 'good afternoon', 'good evening']):
        return f"{query.capitalize()}! How can I assist with medical questions today?"
    
    else:
        return "I'm here to help with medical questions. Is there something specific you'd like to know about healthcare?"

def set_custom_prompt(context_prompt=""):
    prompt_template = f"""
    You are a medical assistant with expertise in diabetic foot ulcers and general medical knowledge.

    **CRITICAL INSTRUCTIONS:**
    1. ALWAYS prioritize accuracy and safety in medical information
    2. If the question is about diabetic foot ulcers, provide detailed, expert-level information
    3. For other medical topics, provide helpful but conservative information
    4. NEVER diagnose, prescribe treatments, or provide personal medical advice
    5. ALWAYS recommend consulting healthcare professionals
    6. If information is not in the context, explicitly state this
    7. Structure responses clearly with appropriate medical terminology
    8. Use bullet points for clarity when appropriate
    9. Explain complex medical terms in simple language
    10. Always maintain a professional, compassionate tone
    {context_prompt}

    **Context:** {{context}}
    **Question:** {{question}}

    Provide a comprehensive, accurate medical response:
    """
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = {}
    if 'current_session' not in st.session_state:
        st.session_state.current_session = "session_1"
    if 'session_counter' not in st.session_state:
        st.session_state.session_counter = 1
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = ConversationContext()
    if 'sugar_data' not in st.session_state:
        st.session_state.sugar_data = pd.DataFrame(columns=['Date', 'Time', 'Sugar_Level', 'Meal_Type'])
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "AI Chatbot"

def create_new_chat_session():
    session_id = f"session_{st.session_state.session_counter + 1}"
    st.session_state.session_counter += 1
    st.session_state.chat_sessions[session_id] = []
    st.session_state.current_session = session_id
    st.session_state.messages = []
    st.session_state.conversation_context = ConversationContext()

def switch_chat_session(session_id):
    st.session_state.current_session = session_id
    if session_id in st.session_state.chat_sessions:
        st.session_state.messages = st.session_state.chat_sessions[session_id]
    else:
        st.session_state.messages = []
    st.session_state.conversation_context = ConversationContext()

def clear_current_conversation():
    st.session_state.messages = []
    if st.session_state.current_session in st.session_state.chat_sessions:
        st.session_state.chat_sessions[st.session_state.current_session] = []
    st.session_state.conversation_context = ConversationContext()

def render_sidebar():
    with st.sidebar:
        st.title("Chat History")
        
        if st.button("+ New Chat", use_container_width=True):
            create_new_chat_session()
            st.rerun()
        
        st.divider()
        
        session_ids = list(st.session_state.chat_sessions.keys())
        for session_id in sorted(session_ids, reverse=True):
            session_messages = st.session_state.chat_sessions[session_id]
            
            preview = "New chat"
            for msg in session_messages:
                if msg['role'] == 'user':
                    preview = msg['content'][:30] + "..." if len(msg['content']) > 30 else msg['content']
                    break
            
            is_active = st.session_state.current_session == session_id
            button_style = "primary" if is_active else "secondary"
            
            if st.button(preview, key=f"btn_{session_id}", use_container_width=True):
                switch_chat_session(session_id)
                st.rerun()

def render_sugar_tracker():
    st.header("Blood Sugar Level Tracker")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Add New Reading")
        with st.form("sugar_form"):
            date = st.date_input("Date", datetime.now())
            time = st.time_input("Time", datetime.now().time())
            sugar_level = st.number_input("Blood Sugar Level (mg/dL)", min_value=50, max_value=500, value=120)
            meal_type = st.selectbox("Meal Context", ["Fasting", "Before Breakfast", "After Breakfast", 
                                                    "Before Lunch", "After Lunch", "Before Dinner", "After Dinner", "Bedtime"])
            
            if st.form_submit_button("Add Reading"):
                new_data = pd.DataFrame({
                    'Date': [date],
                    'Time': [time],
                    'Sugar_Level': [sugar_level],
                    'Meal_Type': [meal_type]
                })
                st.session_state.sugar_data = pd.concat([st.session_state.sugar_data, new_data], ignore_index=True)
                st.success("Reading added successfully!")
    
    with col2:
        st.subheader("Recent Readings")
        if not st.session_state.sugar_data.empty:
            recent_data = st.session_state.sugar_data.tail(5)
            st.dataframe(recent_data, use_container_width=True)
        else:
            st.info("No sugar level readings recorded yet.")
    
    st.divider()
    
    # Sugar Level Chart
    if not st.session_state.sugar_data.empty:
        st.subheader("Sugar Level Trends")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert Date to datetime and sort
        chart_data = st.session_state.sugar_data.copy()
        chart_data['DateTime'] = pd.to_datetime(chart_data['Date'].astype(str) + ' ' + chart_data['Time'].astype(str))
        chart_data = chart_data.sort_values('DateTime')
        
        ax.plot(chart_data['DateTime'], chart_data['Sugar_Level'], marker='o', linewidth=2, markersize=6)
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Hypoglycemia Risk (70 mg/dL)')
        ax.axhline(y=180, color='orange', linestyle='--', alpha=0.7, label='Hyperglycemia Risk (180 mg/dL)')
        ax.axhline(y=130, color='green', linestyle='--', alpha=0.7, label='Target Range (70-130 mg/dL)')
        
        ax.set_xlabel('Date and Time')
        ax.set_ylabel('Blood Sugar Level (mg/dL)')
        ax.set_title('Blood Sugar Level Trends')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Analysis and recommendations
        st.subheader("Analysis & Recommendations")
        latest_reading = chart_data.iloc[-1]
        sugar_level = latest_reading['Sugar_Level']
        
        if sugar_level < 70:
            st.error("**Warning: Hypoglycemia Risk** - Your latest reading indicates low blood sugar. Consider consuming fast-acting carbohydrates and consult your doctor.")
        elif sugar_level > 180:
            st.warning("**Warning: Hyperglycemia Risk** - Your latest reading indicates high blood sugar. Monitor carefully and consult your healthcare provider.")
        elif 70 <= sugar_level <= 130:
            st.success("**Good Control** - Your blood sugar is within the target range. Maintain your current management strategy.")
        else:
            st.info("**Moderate Control** - Your blood sugar is slightly elevated. Continue monitoring and follow your healthcare provider's advice.")
    
    # Image upload for sugar readings
    st.subheader("Upload Sugar Test Results")
    uploaded_image = st.file_uploader("Upload an image of your glucose meter reading", 
                                    type=['png', 'jpg', 'jpeg'], 
                                    help="Upload a clear photo of your glucose meter display")
    
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Glucose Reading", width=300)
        st.info("Image uploaded successfully. Please also enter the numerical value in the form above for accurate tracking.")

def render_chatbot():
    st.header("Medical AI Chatbot")
    
    # Display chat messages
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><strong>Assistant:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    
    # Welcome message if no messages yet
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Hello! I'm your medical assistant. I specialize in diabetic foot ulcers and general medical information. How can I assist you today?\n\n*Please remember: I provide educational information only. Always consult healthcare professionals for personal medical advice.*"
        })
    
    # Chat input
    prompt = st.chat_input("Type your medical question here...")
    
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Emergency check
        if is_emergency_query(prompt):
            st.session_state.messages.append({"role": "assistant", "content": EMERGENCY_RESPONSE})
            st.rerun()
            return
        
        # Conversational check
        if is_conversational_query(prompt):
            response = get_conversational_response(prompt)
            st.session_state.conversation_context.update_context(prompt, response, is_medical=False)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
            return

        # Process medical queries
        with st.spinner("Analyzing your question..."):
            try:
                resolved_query = st.session_state.conversation_context.resolve_references(prompt)
                context_prompt = st.session_state.conversation_context.get_context_prompt()
                
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    error_msg = "Medical knowledge base not available."
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.rerun()
                    return

                groq_api_key = os.environ.get("GROQ_API_KEY")
                if not groq_api_key:
                    error_msg = "API configuration error."
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.rerun()
                    return

                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatGroq(
                        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                        temperature=0.1,
                        groq_api_key=groq_api_key,
                    ),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            'k': 6,
                            'fetch_k': 10,
                            'lambda_mult': 0.7
                        }
                    ),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(context_prompt)}
                )

                response = qa_chain.invoke({'query': resolved_query})
                result = response["result"].strip()

                if (not result or 
                    any(phrase in result.lower() for phrase in [
                        "don't know", "not in", "no information", 
                        "i cannot", "unable to", "not found"
                    ])):
                    result = UNCERTAIN_RESPONSE_TEMPLATE

                st.session_state.conversation_context.update_context(prompt, result, is_medical=True)
                st.session_state.messages.append({"role": "assistant", "content": result})
                
                st.session_state.chat_sessions[st.session_state.current_session] = st.session_state.messages
                st.rerun()

            except Exception as e:
                error_msg = f"I apologize, I'm experiencing technical difficulties. Please try again in a moment."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.session_state.chat_sessions[st.session_state.current_session] = st.session_state.messages
                st.rerun()

def main():
    st.set_page_config(
        page_title="Medical Assistant Pro",
        page_icon="⚕️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    if st.session_state.current_session not in st.session_state.chat_sessions:
        st.session_state.chat_sessions[st.session_state.current_session] = st.session_state.messages
    
    # Header with tabs
    st.markdown('<div class="header-tabs">', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🤖 AI Chatbot", "📊 Sugar Level Tracker"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Render sidebar
    render_sidebar()
    
    # Main content based on selected tab
    with tab1:
        render_chatbot()
    
    with tab2:
        render_sugar_tracker()

if __name__ == "__main__":
    main()