import json
import numpy as np
from typing import List, Dict, Any, Tuple

class AdaptiveRetrieval:
    """
    Implements the Self-Optimizing Adaptive Retrieval (SOAR) component
    """
    
    def __init__(self, 
                 symptoms_conditions_path: str = 'data/symptoms_conditions.json',
                 questions_bank_path: str = 'data/questions_bank.json'):
        """
        Initialize the adaptive retrieval system
        
        Args:
            symptoms_conditions_path: Path to symptoms and conditions data
            questions_bank_path: Path to questions bank data
        """
        # Load symptom and condition data
        with open(symptoms_conditions_path, 'r') as f:
            self.symptoms_conditions_data = json.load(f)
            
        # Load questions data
        with open(questions_bank_path, 'r') as f:
            self.questions_data = json.load(f)
        
        # Create symptom-to-condition index
        self.symptom_to_conditions = {}
        for symptom in self.symptoms_conditions_data["symptoms"]:
            self.symptom_to_conditions[symptom["id"]] = symptom["related_conditions"]
            
        # Create question effectiveness tracking
        self.question_performance = {q["id"]: 0.5 for q in self.questions_data["questions"]}
    
    def identify_symptoms(self, user_query: str) -> List[str]:
        """
        Identify symptoms from user query using keyword matching
        
        Args:
            user_query: The user's initial complaint
            
        Returns:
            List of symptom IDs identified in the query
        """
        # Convert query to lowercase for case-insensitive matching
        query_lower = user_query.lower()
        
        # Find symptoms mentioned in the query
        identified_symptoms = []
        
        for symptom in self.symptoms_conditions_data["symptoms"]:
            # Check if symptom name or aliases appear in query
            if symptom["name"].lower() in query_lower:
                identified_symptoms.append(symptom["id"])
                continue
                
            # Check for related keywords/descriptions
            if "description" in symptom and symptom["description"].lower() in query_lower:
                identified_symptoms.append(symptom["id"])
                
        return identified_symptoms
    
    def get_initial_conditions(self, symptom_ids: List[str]) -> List[str]:
        """
        Get initial set of potential conditions based on identified symptoms
        
        Args:
            symptom_ids: List of identified symptom IDs
            
        Returns:
            List of condition IDs that might explain the symptoms
        """
        # Create a set of condition IDs that match any of the symptoms
        condition_ids = set()
        
        for symptom_id in symptom_ids:
            # Add all conditions related to this symptom
            if symptom_id in self.symptom_to_conditions:
                condition_ids.update(self.symptom_to_conditions[symptom_id])
                
        return list(condition_ids)
    
    def select_next_question(self, 
                            identified_symptoms: List[str],
                            potential_conditions: List[str],
                            condition_probabilities: Dict[str, float],
                            asked_questions: List[str]) -> Dict:
        """
        Select the most informative next question to ask based on current state
        
        Args:
            identified_symptoms: Symptoms identified so far
            potential_conditions: Current list of potential conditions
            condition_probabilities: Current probabilities for each condition
            asked_questions: List of questions already asked
            
        Returns:
            Selected question object
        """
        if not potential_conditions:
            return None
            
        # Filter questions not already asked
        available_questions = [q for q in self.questions_data["questions"] 
                              if q["id"] not in asked_questions]
        
        if not available_questions:
            return None
            
        # Calculate information gain for each question
        question_scores = {}
        
        for question in available_questions:
            # Skip if question not related to any potential condition
            relevant_conditions = set()
            if "yes" in question["condition_indication"]:
                relevant_conditions.update(question["condition_indication"]["yes"])
            if "no" in question["condition_indication"]:
                relevant_conditions.update(question["condition_indication"]["no"])
                
            relevant_conditions = relevant_conditions.intersection(set(potential_conditions))
            
            if not relevant_conditions:
                continue
                
            # Calculate average probability of affected conditions
            avg_probability = sum(condition_probabilities.get(cond, 0) for cond in relevant_conditions) / len(relevant_conditions)
            
            # Calculate differentiation potential (how well it separates top conditions)
            if "yes" in question["condition_indication"] and "no" in question["condition_indication"]:
                yes_conditions = set(question["condition_indication"]["yes"]).intersection(set(potential_conditions))
                no_conditions = set(question["condition_indication"]["no"]).intersection(set(potential_conditions))
                
                # If question can differentiate between conditions, give it higher score
                differentiation_score = len(yes_conditions) * len(no_conditions)
            else:
                differentiation_score = 0
                
            # Factor in question performance history
            performance_factor = self.question_performance.get(question["id"], 0.5)
            
            # Calculate final score
            question_scores[question["id"]] = (
                avg_probability * 0.4 +          # Condition relevance (40%)
                differentiation_score * 0.4 +     # Differentiation power (40%)
                performance_factor * 0.2          # Historical performance (20%)
            )
            
        # If no relevant questions found
        if not question_scores:
            return None
            
        # Get question with highest score
        best_question_id = max(question_scores, key=question_scores.get)
        
        for q in self.questions_data["questions"]:
            if q["id"] == best_question_id:
                return q
                
        return None
    
    def update_question_performance(self, question_id: str, information_gain: float):
        """
        Update question performance tracking based on information gain
        
        Args:
            question_id: ID of the question asked
            information_gain: How much information was gained (0.0-1.0)
        """
        # Current performance score
        current_score = self.question_performance.get(question_id, 0.5)
        
        # Update using exponential moving average
        alpha = 0.3  # Learning rate
        self.question_performance[question_id] = (1 - alpha) * current_score + alpha * information_gain