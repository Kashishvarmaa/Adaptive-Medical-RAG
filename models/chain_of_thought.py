import json
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Set

class ChainOfThought:
    """
    Implements Chain-of-Thought reasoning for medical diagnosis
    """
    
    def __init__(self, knowledge_base_path: str = 'data/medical_knowledge_base.json'):
        """
        Initialize the Chain-of-Thought reasoning module
        
        Args:
            knowledge_base_path: Path to medical knowledge base
        """
        self.knowledge_base = self._load_json(knowledge_base_path)
        
    def _load_json(self, file_path: str) -> Dict:
        """Load JSON data from file"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def generate_reasoning(self, symptoms: List[str], potential_conditions: List[str]) -> Dict[str, Any]:
        """
        Generate step-by-step reasoning for diagnosis based on symptoms
        
        Args:
            symptoms: List of symptom IDs
            potential_conditions: List of condition IDs to consider
            
        Returns:
            Dictionary containing reasoning steps and initial probabilities
        """
        reasoning = {
            "steps": [],
            "condition_probabilities": {}
        }
        
        # Initial step: identify symptoms
        reasoning["steps"].append({
            "step_type": "identification",
            "content": f"Identified {len(symptoms)} symptoms in the patient's description."
        })
        
        # Map symptoms to potential conditions
        symptom_to_condition_map = {}
        for condition in self.knowledge_base["medical_knowledge"]:
            if condition["condition_id"] in potential_conditions:
                # Get primary and secondary symptoms for this condition
                primary_symptoms = set([s for s in condition.get("primary_symptoms", [])])
                secondary_symptoms = set([s for s in condition.get("secondary_symptoms", [])])
                
                # Count matching symptoms
                matching_primary = primary_symptoms.intersection(set(symptoms))
                matching_secondary = secondary_symptoms.intersection(set(symptoms))
                
                # Calculate initial probability based on symptom match
                primary_weight = 2.0  # Primary symptoms carry more weight
                secondary_weight = 1.0
                
                total_possible_score = (len(primary_symptoms) * primary_weight + 
                                        len(secondary_symptoms) * secondary_weight)
                
                if total_possible_score > 0:
                    match_score = (len(matching_primary) * primary_weight + 
                                   len(matching_secondary) * secondary_weight)
                    probability = match_score / total_possible_score
                else:
                    probability = 0.0
                
                # Apply minimum threshold
                probability = max(0.1, probability)
                
                reasoning["condition_probabilities"][condition["condition_id"]] = probability
                
                # Record the reasoning step
                matching_symptoms = list(matching_primary) + list(matching_secondary)
                if matching_symptoms:
                    reasoning["steps"].append({
                        "step_type": "association",
                        "condition": condition["name"],
                        "content": f"The reported symptoms align with {condition['name']} (initial probability: {probability:.2f})."
                    })
        
        # Normalize probabilities to sum to 1
        total_prob = sum(reasoning["condition_probabilities"].values())
        if total_prob > 0:
            for cond in reasoning["condition_probabilities"]:
                reasoning["condition_probabilities"][cond] /= total_prob
        
        return reasoning
    
    def update_reasoning(self, reasoning: Dict, question_id: str, answer: str, 
                         questions_data: Dict, conditions_data: Dict) -> Dict:
        """
        Update reasoning based on patient's answer to a question
        
        Args:
            reasoning: Current reasoning state
            question_id: ID of the question that was asked
            answer: Patient's answer ('yes' or 'no')
            questions_data: Questions bank data
            conditions_data: Conditions data
            
        Returns:
            Updated reasoning dictionary
        """
        # Find the question in the questions bank
        question = None
        for q in questions_data["questions"]:
            if q["id"] == question_id:
                question = q
                break
        
        if not question:
            reasoning["steps"].append({
                "step_type": "error",
                "content": f"Question with ID {question_id} not found."
            })
            return reasoning
        
        # Extract which conditions this answer indicates
        condition_indication = question["condition_indication"].get(answer.lower(), [])
        
        # Update probabilities based on the answer
        probabilities = reasoning["condition_probabilities"]
        
        # Get condition names for logging
        condition_names = {c["id"]: c["name"] for c in conditions_data["conditions"]}
        
        if answer.lower() == "yes":
            # If yes, increase probability for indicated conditions
            indicated_conditions = ", ".join([condition_names.get(c_id, c_id) for c_id in condition_indication])
            if indicated_conditions:
                reasoning["steps"].append({
                    "step_type": "refinement",
                    "content": f"The positive response to '{question['text']}' increases likelihood of: {indicated_conditions}"
                })
            
            # Boost indicated conditions
            boost_factor = 1.5
            for cond_id in condition_indication:
                if cond_id in probabilities:
                    probabilities[cond_id] *= boost_factor
        else:  # answer is "no"
            # If no, decrease probability for indicated conditions
            indicated_conditions = ", ".join([condition_names.get(c_id, c_id) for c_id in condition_indication])
            if indicated_conditions:
                reasoning["steps"].append({
                    "step_type": "refinement",
                    "content": f"The negative response to '{question['text']}' decreases likelihood of: {indicated_conditions}"
                })
            
            # Reduce probability for indicated conditions
            reduction_factor = 0.5
            for cond_id in condition_indication:
                if cond_id in probabilities:
                    probabilities[cond_id] *= reduction_factor
        
        # Renormalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            for cond_id in probabilities:
                probabilities[cond_id] /= total_prob
        
        return reasoning
    
    def get_conclusion(self, reasoning: Dict, conditions_data: Dict, 
                       threshold: float = 0.5) -> Dict[str, Any]:
        """
        Generate a diagnostic conclusion based on reasoning
        
        Args:
            reasoning: Reasoning dictionary with updated probabilities
            conditions_data: Conditions data
            threshold: Probability threshold for definitive diagnosis
            
        Returns:
            Conclusion dictionary with diagnosis and recommendations
        """
        probabilities = reasoning["condition_probabilities"]
        
        # Get list of conditions sorted by probability
        sorted_conditions = sorted(
            probabilities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Get the top condition
        top_condition_id, top_prob = sorted_conditions[0]
        
        # Get condition data
        condition_data = None
        for c in conditions_data["conditions"]:
            if c["id"] == top_condition_id:
                condition_data = c
                break
                
        if not condition_data:
            return {
                "diagnosis": "Inconclusive",
                "confidence": 0,
                "explanation": "Could not find matching condition data."
            }
        
        # Get condition name
        condition_name = condition_data["name"]
        
        # Find matching knowledge base entry
        knowledge_entry = None
        for entry in self.knowledge_base["medical_knowledge"]:
            if entry["condition_id"] == top_condition_id:
                knowledge_entry = entry
                break
        
        # Generate conclusion based on probability
        if top_prob >= threshold:
            diagnosis_status = "Likely diagnosis"
            confidence = "high"
        elif top_prob >= 0.3:
            diagnosis_status = "Possible diagnosis"
            confidence = "moderate"
        else:
            diagnosis_status = "Differential diagnosis"
            confidence = "low"
        
        # Generate explanation
        if knowledge_entry:
            explanation = knowledge_entry["medical_description"]
            treatment = knowledge_entry["treatment_approaches"]
        else:
            explanation = condition_data["description"]
            treatment = []
        
        # Add disclaimer
        disclaimer = ("This is not a definitive medical diagnosis. Please consult with a healthcare "
                      "professional for proper evaluation and treatment.")
        
        return {
            "diagnosis": condition_name,
            "diagnosis_status": diagnosis_status,
            "confidence": confidence,
            "probability": top_prob,
            "explanation": explanation,
            "treatment_approaches": treatment,
            "urgency": condition_data.get("urgency", "Unknown"),
            "disclaimer": disclaimer,
            "differential_diagnoses": [
                {"condition": conditions_data["conditions"][int(cond_id[1:])-1]["name"], "probability": prob}
                for cond_id, prob in sorted_conditions[1:4]  # Include next 3 most likely conditions
            ]
        }