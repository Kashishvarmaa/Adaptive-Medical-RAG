import json
from typing import List, Dict, Any, Optional
from adaptive_retrieval import AdaptiveRetrieval
from chain_of_thought import ChainOfThought
from graph_builder import GraphBuilder

class RAGModel:
    """
    Main RAG model that coordinates the retrieval and generation components
    """
    
    def __init__(self):
        """Initialize the RAG model with its components"""
        self.adaptive_retrieval = AdaptiveRetrieval()
        self.chain_of_thought = ChainOfThought()
        self.graph_builder = GraphBuilder()
        
        # Load symptom and condition data
        with open('data/symptoms_conditions.json', 'r') as f:
            self.symptoms_conditions_data = json.load(f)
            
        # Load questions data
        with open('data/questions_bank.json', 'r') as f:
            self.questions_data = json.load(f)
        
        # Session state
        self.reset_session()
        
    def reset_session(self):
        """Reset the session state for a new conversation"""
        self.session = {
            "identified_symptoms": [],
            "potential_conditions": [],
            "asked_questions": [],
            "answers": {},
            "reasoning": {
                "steps": [],
                "condition_probabilities": {}
            },
            "graph": self.graph_builder.initialize_graph(),
            "conclusion": None,
            "status": "initial"  # initial, questioning, concluded
        }
        
    def process_initial_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process the initial query from the user
        
        Args:
            user_query: The user's initial complaint
            
        Returns:
            Dictionary with identified symptoms and next question
        """
        # Reset session for new query
        self.reset_session()
        
        # Identify symptoms in the query
        self.session["identified_symptoms"] = self.adaptive_retrieval.identify_symptoms(user_query)
        
        # If no symptoms identified, return early
        if not self.session["identified_symptoms"]:
            return {
                "success": False,
                "error": "No symptoms identified in your description. Please provide more details about your symptoms."
            }
        
        # Get potential conditions based on symptoms
        self.session["potential_conditions"] = self.adaptive_retrieval.get_initial_conditions(
            self.session["identified_symptoms"]
        )
        
        # Update graph with symptoms
        self.session["graph"] = self.graph_builder.add_symptom_nodes(
            self.session["graph"], 
            self.session["identified_symptoms"]
        )
        
        # Update graph with potential conditions
        self.session["graph"] = self.graph_builder.add_condition_nodes(
            self.session["graph"], 
            self.session["potential_conditions"]
        )
        
        # Connect symptoms to conditions in the graph
        self.session["graph"] = self.graph_builder.connect_symptoms_to_conditions(
            self.session["graph"],
            self.session["identified_symptoms"],
            self.session["potential_conditions"]
        )
        
        # Generate initial reasoning
        self.session["reasoning"] = self.chain_of_thought.generate_reasoning(
            self.session["identified_symptoms"],
            self.session["potential_conditions"]
        )
        
        # Update condition probabilities in graph
        self.session["graph"] = self.graph_builder.update_condition_probabilities(
            self.session["graph"],
            self.session["reasoning"]["condition_probabilities"]
        )
        
        # Get next question to ask
        next_question = self.adaptive_retrieval.select_next_question(
            self.session["identified_symptoms"],
            self.session["potential_conditions"],
            self.session["reasoning"]["condition_probabilities"],
            self.session["asked_questions"]
        )
        
        # Update session status
        self.session["status"] = "questioning"
        
        # Return the response
        return {
            "success": True,
            "identified_symptoms": [self.get_symptom_name(sym_id) for sym_id in self.session["identified_symptoms"]],
            "potential_conditions": [self.get_condition_name(cond_id) for cond_id in self.session["potential_conditions"]],
            "next_question": next_question["text"] if next_question else None,
            "next_question_id": next_question["id"] if next_question else None,
            "graph": self.session["graph"],
            "reasoning_steps": self.session["reasoning"]["steps"]
        }
        
    def process_answer(self, question_id: str, answer: str) -> Dict[str, Any]:
        """
        Process the user's answer to a question
        
        Args:
            question_id: ID of the question that was asked
            answer: User's answer (usually 'yes' or 'no')
            
        Returns:
            Dictionary with updated state and next question
        """
        # Validate input
        if self.session["status"] != "questioning":
            return {
                "success": False,
                "error": "No active questioning session. Please start with a new query."
            }
            
        if question_id in self.session["asked_questions"]:
            return {
                "success": False,
                "error": "This question has already been answered."
            }
            
        # Standardize answer format
        std_answer = answer.lower()
        if std_answer not in ["yes", "no"]:
            std_answer = "yes" if "yes" in std_answer.lower() else "no"
            
        # Record the question and answer
        self.session["asked_questions"].append(question_id)
        self.session["answers"][question_id] = std_answer
        
        # Find the question object
        question = None
        for q in self.questions_data["questions"]:
            if q["id"] == question_id:
                question = q
                break
                
        if not question:
            return {
                "success": False,
                "error": f"Question with ID {question_id} not found."
            }
            
        # Update the graph with the question
        self.session["graph"] = self.graph_builder.add_question_node(
            self.session["graph"],
            question,
            std_answer
        )
        
        # Update the reasoning based on the answer
        self.session["reasoning"] = self.chain_of_thought.update_reasoning(
            self.session["reasoning"],
            question_id,
            std_answer,
            self.questions_data,
            self.symptoms_conditions_data
        )
        
        # Update condition probabilities in the graph
        self.session["graph"] = self.graph_builder.update_condition_probabilities(
            self.session["graph"],
            self.session["reasoning"]["condition_probabilities"]
        )
        
        # Calculate information gain from this question
        # (simplified as the absolute change in probabilities)
        probabilities = self.session["reasoning"]["condition_probabilities"]
        max_prob_change = 0.0
        
        for cond_id in self.session["potential_conditions"]:
            if cond_id in probabilities:
                # Information gain is higher when probabilities change significantly
                max_prob_change = max(max_prob_change, abs(probabilities[cond_id] - 0.5) * 2)
                
        # Update question performance tracking
        self.adaptive_retrieval.update_question_performance(
            question_id, 
            max_prob_change
        )
        
        # Check if we should conclude the diagnosis
        should_conclude = (
            len(self.session["asked_questions"]) >= 3 and  # At least 3 questions asked
            (
                max(probabilities.values()) >= 0.6 or  # High confidence in top condition
                len(self.session["asked_questions"]) >= 5  # Or maximum questions reached
            )
        )
        
        if should_conclude:
            # Generate conclusion
            self.session["conclusion"] = self.chain_of_thought.get_conclusion(
                self.session["reasoning"],
                self.symptoms_conditions_data
            )
            self.session["status"] = "concluded"
            
            return {
                "success": True,
                "conclusion": self.session["conclusion"],
                "graph": self.session["graph"],
                "reasoning_steps": self.session["reasoning"]["steps"],
                "status": "concluded"
            }
        else:
            # Get next question
            next_question = self.adaptive_retrieval.select_next_question(
                self.session["identified_symptoms"],
                self.session["potential_conditions"],
                self.session["reasoning"]["condition_probabilities"],
                self.session["asked_questions"]
            )
            
            return {
                "success": True,
                "next_question": next_question["text"] if next_question else None,
                "next_question_id": next_question["id"] if next_question else None,
                "graph": self.session["graph"],
                "reasoning_steps": self.session["reasoning"]["steps"],
                "status": "questioning"
            }
    
    def get_conclusion(self) -> Dict[str, Any]:
        """
        Get the final diagnostic conclusion
        
        Returns:
            The conclusion dictionary
        """
        if self.session["status"] != "concluded" or not self.session["conclusion"]:
            # Generate conclusion from current state
            self.session["conclusion"] = self.chain_of_thought.get_conclusion(
                self.session["reasoning"],
                self.symptoms_conditions_data
            )
            self.session["status"] = "concluded"
            
        return {
            "success": True,
            "conclusion": self.session["conclusion"],
            "graph": self.session["graph"],
            "reasoning_steps": self.session["reasoning"]["steps"],
            "status": "concluded"
        }
    
    def get_symptom_name(self, symptom_id: str) -> str:
        """Get symptom name from ID"""
        for symptom in self.symptoms_conditions_data["symptoms"]:
            if symptom["id"] == symptom_id:
                return symptom["name"]
        return f"Unknown symptom ({symptom_id})"
    
    def get_condition_name(self, condition_id: str) -> str:
        """Get condition name from ID"""
        for condition in self.symptoms_conditions_data["conditions"]:
            if condition["id"] == condition_id:
                return condition["name"]
        return f"Unknown condition ({condition_id})"