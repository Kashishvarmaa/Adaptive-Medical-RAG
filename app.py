import streamlit as st

# Set page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Adaptive Med-RAG",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import sys
import os

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# Import our models
from models.rag_model import RAGModel
from models.report_generator import ReportGenerator

# Custom CSS for a more MERN-stack look
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }
    
    /* Header styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Card styling */
    .css-1r6slb0 {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Styling for the chat messages */
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        background-color: #022c4f;
        border-left: 5px solid #4285f4;
    }
    
    .system-message {
        background-color: #022c4f;
        border-left: 5px solid #34a853;
    }
    
    /* Title and header styling */
    h1, h2, h3 {
        color: #4285f4;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #4285f4;
        color: white;
        border-radius: 5px;
        font-weight: 500;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background-color: #2b6cb0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Chat input styling */
    .stTextInput input {
        border-radius: 20px;
        border: 1px solid #e0e0e0;
        padding: 10px 15px;
    }
    
    /* Reasoning steps styling */
    .reasoning-step {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    
    .identification-step {
        background-color: #052763;
        border-left: 3px solid #4285f4;
    }
    
    .association-step {
        background-color: #052763;
        border-left: 3px solid #34a853;
    }
    
    .refinement-step {
        background-color: #052763;
        border-left: 3px solid #fbbc04;
    }
    
    /* Conclusion styling */
    .conclusion-card {
        background-color: #052763;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .diagnosis-header {
        background-color: #052763;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    
    .diagnosis-high {
        color: #34a853;
        font-weight: bold;
    }
    
    .diagnosis-moderate {
        color: #fbbc04;
        font-weight: bold;
    }
    
    .diagnosis-low {
        color: #b81c0f;
        font-weight: bold;
    }
    
    .disclaimer {
        background-color: #052763;
        padding: 10px;
        border-radius: 5px;
        font-size: 0.9em;
        margin-top: 15px;
    }
    
    .treatment-item {
        background-color: #09de46;
        padding: 8px;
        margin: 5px 0;
        border-radius: 4px;
    }
    
    .report-info {
        background-color: #e8f0fe;
        padding: 10px;
        border-radius: 5px;
        margin-top: 15px;
        border-left: 3px solid #4285f4;
    }
    
    .llm-enhanced {
        background-color: #f0f8ff;
        padding: 12px;
        border-radius: 6px;
        margin: 10px 0;
        border-left: 4px solid #4285f4;
    }
    
    .llm-enhanced h4 {
        margin-top: 0;
        color: #1a73e8;
        font-size: 1.1em;
    }
</style>
""", unsafe_allow_html=True)

# Function to handle model responses
def handle_response(response):
    if not response.get('success', False):
        # Handle error
        error_message = response.get('error', 'An unknown error occurred')
        st.session_state.chat_history.append({
            'type': 'system',
            'content': f"⚠️ {error_message}"
        })
        return
    
    if response.get('status') == 'questioning':
        # Handle questioning state
        next_question = response.get('next_question')
        next_question_id = response.get('next_question_id')
        
        if next_question:
            st.session_state.current_question_id = next_question_id
            st.session_state.chat_history.append({
                'type': 'system',
                'content': next_question
            })
    elif response.get('status') == 'concluded':
        # Handle conclusion state
        conclusion = response.get('conclusion', {})
        
        if conclusion:
            # Format the conclusion for display
            diagnosis = conclusion.get('diagnosis', 'Unknown')
            diagnosis_status = conclusion.get('diagnosis_status', 'Inconclusive')
            confidence = conclusion.get('confidence', 'low')
            explanation = conclusion.get('explanation', '')
            disclaimer = conclusion.get('disclaimer', '')
            treatment = conclusion.get('treatment_approaches', [])
            urgency = conclusion.get('urgency', 'Unknown')
            
            # Generate report
            if hasattr(st.session_state, 'initial_query'):
                # Get identified symptoms
                identified_symptoms = [
                    st.session_state.rag_model.get_symptom_name(sym_id) 
                    for sym_id in st.session_state.rag_model.session["identified_symptoms"]
                ]
                
                # Generate and save report
                report_path = st.session_state.report_generator.generate_report(
                    st.session_state.initial_query,
                    identified_symptoms,
                    st.session_state.questions_answers,
                    st.session_state.rag_model.session["reasoning"]["steps"],
                    st.session_state.rag_model.session["reasoning"]["condition_probabilities"],
                    conclusion
                )
                
                # Add report path to session
                st.session_state.report_path = report_path
            
            # Create the conclusion message
            conclusion_message = f"""
            <div class="conclusion-card">
                <div class="diagnosis-header">
                    <h3>Diagnosis: {diagnosis}</h3>
                    <p>{diagnosis_status} with <span class="diagnosis-{confidence}">{confidence} confidence</span></p>
                    <p><strong>Urgency:</strong> {urgency}</p>
                </div>
                <p><strong>Explanation:</strong> {explanation}</p>
            """
            
            # Add enhanced LLM content if available
            if "enhanced_explanation" in conclusion:
                conclusion_message += f"""
                <div class="llm-enhanced">
                    <h4>Detailed Analysis:</h4>
                    <p>{conclusion["enhanced_explanation"]}</p>
                </div>
                """
            
            if "key_factors" in conclusion:
                conclusion_message += f"""
                <div class="llm-enhanced">
                    <h4>Key Diagnostic Factors:</h4>
                    <p>{conclusion["key_factors"]}</p>
                </div>
                """
            
            conclusion_message += """
                <h4>Possible Treatment Approaches:</h4>
                <ul>
            """
            for item in treatment:
                conclusion_message += f'<li class="treatment-item">{item}</li>'
                
            conclusion_message += "</ul>"
            # Add enhanced next steps if available
            if "next_steps" in conclusion:
                conclusion_message += f"""
                <div class="llm-enhanced">
                    <h4>Recommended Next Steps:</h4>
                    <p>{conclusion["next_steps"]}</p>
                </div>
                """
                
            conclusion_message += f"""
                <div class="disclaimer">
                    <strong>Important:</strong> {disclaimer}
                </div>
            """
            
            # Add report information if available
            if hasattr(st.session_state, 'report_path'):
                report_rel_path = os.path.relpath(st.session_state.report_path)
                conclusion_message += f"""
                <div class="report-info"><p><strong>Detailed Report:</strong> A complete consultation report has been saved to {report_rel_path}</p>
                </div>
                """
                
            conclusion_message += "</div>"
            
            st.session_state.chat_history.append({
                'type': 'system',
                'content': conclusion_message
            })
            
            st.session_state.diagnosis_complete = True
            st.session_state.current_question_id = None
    else:
        # Handle initial response with identified symptoms
        identified_symptoms = response.get('identified_symptoms', [])
        
        if identified_symptoms:
            symptoms_list = ", ".join(identified_symptoms)
            initial_message = f"I've identified these symptoms: {symptoms_list}. Let me ask you some questions to better understand your condition."
            
            st.session_state.chat_history.append({
                'type': 'system',
                'content': initial_message
            })
            
            # If there's a next question available, add it
            next_question = response.get('next_question')
            next_question_id = response.get('next_question_id')
            
            if next_question:
                st.session_state.current_question_id = next_question_id
                st.session_state.chat_history.append({
                    'type': 'system',
                    'content': next_question
                })

# Initialize session state
if 'rag_model' not in st.session_state:
    st.session_state.rag_model = RAGModel()
    
if 'report_generator' not in st.session_state:
    st.session_state.report_generator = ReportGenerator()
    
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
if 'current_question_id' not in st.session_state:
    st.session_state.current_question_id = None
    
if 'diagnosis_complete' not in st.session_state:
    st.session_state.diagnosis_complete = False
    
if 'questions_answers' not in st.session_state:
    st.session_state.questions_answers = {}

# Header
st.title("🩺 Adaptive Med-RAG")
st.markdown("A medical question-answering system powered by Chain-of-Thought reasoning and Graph of Thought visualization")

# Create layout with columns
col1, col2 = st.columns([3, 2])

with col1:
    # Chat interface
    st.subheader("Medical Consultation")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['type'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <div><strong>You:</strong> {message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message system-message">
                <div><strong>Med-RAG:</strong> {message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Input for user message
    user_input = st.text_input("Describe your symptoms", key="user_input")
    
    # Add Yes/No buttons for questions
    if st.session_state.current_question_id and not st.session_state.diagnosis_complete:
        col1_1, col1_2, col1_3 = st.columns([1, 1, 3])
        with col1_1:
            if st.button("Yes"):
                # Process 'Yes' answer
                response = st.session_state.rag_model.process_answer(
                    st.session_state.current_question_id, 
                    "yes"
                )
                
                # Store question and answer for report
                for q in st.session_state.rag_model.questions_data["questions"]:
                    if q["id"] == st.session_state.current_question_id:
                        st.session_state.questions_answers[q["text"]] = "Yes"
                        break
                
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': "Yes"
                })
                handle_response(response)
                st.rerun()
                
        with col1_2:
            if st.button("No"):
                # Process 'No' answer
                response = st.session_state.rag_model.process_answer(
                    st.session_state.current_question_id, 
                    "no"
                )
                
                # Store question and answer for report
                for q in st.session_state.rag_model.questions_data["questions"]:
                    if q["id"] == st.session_state.current_question_id:
                        st.session_state.questions_answers[q["text"]] = "No"
                        break
                
                st.session_state.chat_history.append({
                    'type': 'user',
                    'content': "No"
                })
                handle_response(response)
                st.rerun()
    
    # Submit button for initial symptoms
    if not st.session_state.current_question_id and not st.session_state.diagnosis_complete:
        if st.button("Submit") and user_input:
            # Process the initial query
            response = st.session_state.rag_model.process_initial_query(user_input)
            
            # Add message to chat history
            st.session_state.chat_history.append({
                'type': 'user',
                'content': user_input
            })
            
            # Store the initial query for the report
            st.session_state.initial_query = user_input
            
            # Handle the response
            handle_response(response)
            st.rerun()
    
    # Reset button
    if len(st.session_state.chat_history) > 0:
        if st.button("Start New Consultation"):
            # Reset the session
            st.session_state.rag_model.reset_session()
            st.session_state.chat_history = []
            st.session_state.current_question_id = None
            st.session_state.diagnosis_complete = False
            st.session_state.questions_answers = {}
            if 'initial_query' in st.session_state:
                del st.session_state.initial_query
            if 'report_path' in st.session_state:
                del st.session_state.report_path
            st.rerun()
    
with col2:
    # Graph visualization and reasoning steps
    st.subheader("Diagnostic Process")
    
    # Show graph if available
    if hasattr(st.session_state.rag_model, 'session') and 'graph' in st.session_state.rag_model.session:
        graph_data = st.session_state.rag_model.session['graph']
        
        if graph_data['nodes']:
            # Create networkx graph from our data
            G = nx.Graph()
            
            # Add nodes
            node_colors_dict = {}
            node_sizes_dict = {}
            labels = {}
            
            # First, add all nodes from the graph data
            for node in graph_data['nodes']:
                G.add_node(node['id'])
                labels[node['id']] = node['name']
                
                # Node color based on type
                if node['type'] == 'symptom':
                    color = '#A6CEE3'  # Light blue
                    size = 300
                elif node['type'] == 'condition':
                    # Color condition based on probability if available
                    if 'probability' in node:
                        # Gradient from red (low) to green (high)
                        prob = node['probability']
                        if prob < 0.3:
                            color = '#FB9A99'  # Light red
                        elif prob < 0.6:
                            color = '#FDBF6F'  # Orange
                        else:
                            color = '#B2DF8A'  # Light green
                    else:
                        color = '#B2DF8A'  # Default light green
                    size = 400
                else:  # question
                    color = '#CAB2D6'  # Light purple
                    size = 250
                
                node_colors_dict[node['id']] = color
                node_sizes_dict[node['id']] = size
            
            # Add edges
            for link in graph_data['links']:
                source = link['source']
                target = link['target']
                
                # Ensure nodes referenced in links are in the graph
                if source not in G:
                    G.add_node(source)
                    labels[source] = f"Node {source}"
                    node_colors_dict[source] = '#CCCCCC'  # Default gray
                    node_sizes_dict[source] = 200
                
                if target not in G:
                    G.add_node(target)
                    labels[target] = f"Node {target}"
                    node_colors_dict[target] = '#CCCCCC'  # Default gray
                    node_sizes_dict[target] = 200
                
                G.add_edge(source, target)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Use spring layout for natural spacing
            pos = nx.spring_layout(G, seed=42)
            
            # Create ordered lists of colors and sizes matching the order of nodes in G
            node_list = list(G.nodes())
            node_colors = [node_colors_dict.get(node, '#CCCCCC') for node in node_list]  # Use get with default
            node_sizes = [node_sizes_dict.get(node, 200) for node in node_list]  # Use get with default
            
            # Draw the network
            nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=node_colors, node_size=node_sizes, alpha=0.8)
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
            nx.draw_networkx_labels(G, pos, labels, font_size=8)
            
            # Create a legend
            symptom_patch = mpatches.Patch(color='#A6CEE3', label='Symptom')
            condition_patch = mpatches.Patch(color='#B2DF8A', label='Condition')
            question_patch = mpatches.Patch(color='#CAB2D6', label='Question')
            
            plt.legend(handles=[symptom_patch, condition_patch, question_patch],
                     loc='upper right')
            
            plt.axis('off')
            plt.title('Diagnostic Graph of Thought')
            
            # Display the plot
            st.pyplot(fig)
    
    # Show reasoning steps
    if hasattr(st.session_state.rag_model, 'session') and 'reasoning' in st.session_state.rag_model.session:
        reasoning = st.session_state.rag_model.session['reasoning']
        
        if 'steps' in reasoning and reasoning['steps']:
            st.subheader("Chain-of-Thought Reasoning")
            
            for step in reasoning['steps']:
                step_type = step.get('step_type', 'general')
                content = step.get('content', '')
                
                if step_type == 'identification':
                    st.markdown(f"""
                    <div class="reasoning-step identification-step">
                        <strong>Identification:</strong> {content}
                    </div>
                    """, unsafe_allow_html=True)
                elif step_type == 'association':
                    st.markdown(f"""
                    <div class="reasoning-step association-step">
                        <strong>Association:</strong> {content}
                    </div>
                    """, unsafe_allow_html=True)
                elif step_type == 'refinement':
                    st.markdown(f"""
                    <div class="reasoning-step refinement-step">
                        <strong>Refinement:</strong> {content}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="reasoning-step general-step">
                        {content}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Show probability table if available
    if (hasattr(st.session_state.rag_model, 'session') and 
        'reasoning' in st.session_state.rag_model.session and 
        'condition_probabilities' in st.session_state.rag_model.session['reasoning']):
        
        probabilities = st.session_state.rag_model.session['reasoning']['condition_probabilities']
        
        if probabilities:
            st.subheader("Condition Probabilities")
            
            # Convert probabilities to list for display
            prob_list = []
            for cond_id, prob in probabilities.items():
                condition_name = st.session_state.rag_model.get_condition_name(cond_id)
                prob_list.append({
                    "Condition": condition_name,
                    "Probability": f"{prob:.2f}"
                })
            
            # Sort by probability
            prob_list = sorted(prob_list, key=lambda x: float(x["Probability"]), reverse=True)
            
            # Display as table
            st.table(prob_list)

# Sidebar with app info
with st.sidebar:
    st.header("About Adaptive Med-RAG")
    st.markdown("""
    This application demonstrates an adaptive retrieval-augmented generation system for medical diagnostics.
    
    **Key Components:**
    
    * **Chain-of-Thought Reasoning**: Step-by-step diagnostic reasoning
    * **Self-Optimizing Adaptive Retrieval (SOAR)**: Dynamic question selection
    * **Graph of Thought**: Visual representation of the diagnostic process
    
    **How to Use:**
    1. Enter your symptoms in the text box
    2. Answer the follow-up questions with Yes/No
    3. Review the diagnostic reasoning and graph visualization
    4. Get a potential diagnosis with explanation
    
    **Important Disclaimer**: This is a research prototype and not a real medical diagnostic tool. Any diagnoses provided are for demonstration purposes only and should not be used for actual medical decisions. Always consult with qualified healthcare professionals.
    """)
    
    st.markdown("---")
    st.markdown("© 2025 Adaptive Med-RAG Research Project")

# Main app function
def main():
    # Everything is handled through Streamlit's session state
    pass

if __name__ == "__main__":
    main()