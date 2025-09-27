import os 
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class EnhancedRAGSystem:
    def __init__(self, model_name="llama-3.1-8b-instant"):
        self.model_name = model_name
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        self.vector_store = None
        self.rag_chain = None
        self.setup_complete = False
        
    def setup_llm(self):
        """Enhanced LLM setup with medical optimization"""
        return ChatGroq(
            model=self.model_name,
            temperature=0.3,  # Lower temperature for medical accuracy
            max_tokens=1024,  # More tokens for detailed medical explanations
            api_key=self.groq_api_key,
        )
    
    def load_enhanced_vector_store(self):
        """Load vector store with enhanced configuration"""
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        try:
            self.vector_store = FAISS.load_local(
                "vectorstore/db_faiss", 
                embedding_model, 
                allow_dangerous_deserialization=True
            )
            return True
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            return False
    
    def create_medical_prompt(self):
        """Create enhanced medical-specific prompt"""
        medical_prompt = ChatPromptTemplate.from_template("""
        You are a medical AI assistant specializing in diabetic foot ulcers and general medical information.

        CONTEXT INFORMATION:
        {context}

        USER QUESTION: {input}

        MEDICAL GUIDELINES:
        1. Provide accurate, evidence-based medical information
        2. Focus on diabetic foot ulcers when relevant
        3. Always recommend consulting healthcare professionals
        4. Do not provide personal medical advice or diagnoses
        5. Use clear, understandable medical terminology
        6. Include relevant precautions and warnings

        Please provide a comprehensive yet concise response:
        """)
        return medical_prompt
    
    def build_enhanced_rag_chain(self):
        """Build enhanced RAG chain with medical optimization"""
        llm = self.setup_llm()
        
        # Use medical-specific prompt
        medical_prompt = self.create_medical_prompt()
        
        # Enhanced document combiner
        combine_docs_chain = create_stuff_documents_chain(llm, medical_prompt)
        
        # Enhanced retriever with medical optimization
        retriever = self.vector_store.as_retriever(
            search_type="mmr",  # Max Marginal Relevance for diversity
            search_kwargs={
                'k': 5,  # More documents for comprehensive medical answers
                'fetch_k': 10,
                'lambda_mult': 0.7
            }
        )
        
        # Build retrieval chain
        self.rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
        self.setup_complete = True
    
    def initialize_system(self):
        """Initialize the complete RAG system"""
        print("🔧 Initializing Enhanced RAG System...")
        
        if not self.groq_api_key:
            print("❌ GROQ_API_KEY not found in environment variables")
            return False
        
        if not self.load_enhanced_vector_store():
            return False
        
        self.build_enhanced_rag_chain()
        print("✅ Enhanced RAG system initialized successfully")
        return True
    
    def query_system(self, user_query):
        """Enhanced query processing with better error handling"""
        if not self.setup_complete:
            print("❌ System not initialized. Call initialize_system() first.")
            return None
        
        try:
            print(f"🔍 Processing query: '{user_query}'")
            response = self.rag_chain.invoke({'input': user_query})
            
            # Enhanced response formatting
            result = {
                'answer': response["answer"],
                'source_documents': response["context"],
                'confidence': self.calculate_confidence(response),
                'query_time': 'N/A'  # Could add timing here
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            return None
    
    def calculate_confidence(self, response):
        """Calculate confidence score based on response quality"""
        answer = response["answer"]
        sources = response["context"]
        
        # Simple confidence calculation
        if len(sources) == 0:
            return "Low"
        elif "I don't know" in answer or "not in the context" in answer:
            return "Low"
        elif len(sources) >= 3:
            return "High"
        else:
            return "Medium"
    
    def display_results(self, result):
        """Enhanced result display"""
        if not result:
            print("❌ No results to display")
            return
        
        print("\n" + "="*60)
        print("🎯 MEDICAL AI RESPONSE")
        print("="*60)
        print(f"📝 Answer: {result['answer']}")
        print(f"📊 Confidence: {result['confidence']}")
        
        if result['source_documents']:
            print(f"\n📚 Source Documents ({len(result['source_documents'])} found):")
            for i, doc in enumerate(result['source_documents']):
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                print(f"   {i+1}. {source} (Page {page})")
                print(f"      Preview: {doc.page_content[:100]}...")
        else:
            print("\n⚠️ No specific sources found for this query")
        
        print("="*60)

# Enhanced main execution
if __name__ == "__main__":
    # Initialize the enhanced system
    rag_system = EnhancedRAGSystem(model_name="llama-3.1-8b-instant")
    
    if rag_system.initialize_system():
        print("\n💬 Medical AI Assistant Ready! (Type 'quit' to exit)")
        
        while True:
            user_query = input("\n🩺 Write your medical query here: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'bye']:
                print("👋 Thank you for using the Medical AI Assistant!")
                break
            
            if not user_query:
                print("⚠️ Please enter a valid query")
                continue
            
            # Process query
            result = rag_system.query_system(user_query)
            
            if result:
                rag_system.display_results(result)
            else:
                print("❌ Failed to process your query. Please try again.")