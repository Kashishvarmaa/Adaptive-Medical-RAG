import json
from typing import List, Dict, Any, Tuple, Optional

class GraphBuilder:
    """
    Implements the Graph of Thought visualization component
    """
    
    def __init__(self, symptoms_conditions_path: str = 'data/symptoms_conditions.json'):
        """
        Initialize the graph builder
        
        Args:
            symptoms_conditions_path: Path to symptoms and conditions data
        """
        # Load symptom and condition data
        with open(symptoms_conditions_path, 'r') as f:
            self.data = json.load(f)
            
        # Create symptom and condition lookups
        self.symptoms = {s["id"]: s for s in self.data["symptoms"]}
        self.conditions = {c["id"]: c for c in self.data["conditions"]}
        
    def initialize_graph(self) -> Dict[str, Any]:
        """
        Create initial empty graph structure
        
        Returns:
            Graph structure dictionary
        """
        return {
            "nodes": [],
            "links": [],
            "nodeTypes": {
                "symptom": {"color": "#A6CEE3", "radius": 10},
                "condition": {"color": "#B2DF8A", "radius": 12},
                "question": {"color": "#FB9A99", "radius": 8}
            }
        }
    
    def add_symptom_nodes(self, graph: Dict[str, Any], symptom_ids: List[str]) -> Dict[str, Any]:
        """
        Add symptom nodes to the graph
        
        Args:
            graph: Current graph structure
            symptom_ids: List of symptom IDs to add
            
        Returns:
            Updated graph
        """
        # Add each symptom as a node
        for sym_id in symptom_ids:
            if sym_id in self.symptoms:
                symptom = self.symptoms[sym_id]
                
                # Check if node already exists
                if not any(n["id"] == sym_id for n in graph["nodes"]):
                    graph["nodes"].append({
                        "id": sym_id,
                        "name": symptom["name"],
                        "type": "symptom",
                        "status": "confirmed"
                    })
            
        return graph
    
    def add_condition_nodes(self, 
                           graph: Dict[str, Any], 
                           condition_ids: List[str],
                           probabilities: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Add condition nodes to the graph
        
        Args:
            graph: Current graph structure
            condition_ids: List of condition IDs to add
            probabilities: Optional dictionary of condition probabilities
            
        Returns:
            Updated graph
        """
        # Add each condition as a node
        for cond_id in condition_ids:
            if cond_id in self.conditions:
                condition = self.conditions[cond_id]
                
                # Check if node already exists
                existing_node = next((n for n in graph["nodes"] if n["id"] == cond_id), None)
                
                if existing_node:
                    # Update existing node if probabilities provided
                    if probabilities and cond_id in probabilities:
                        existing_node["probability"] = probabilities[cond_id]
                else:
                    # Create new node
                    node = {
                        "id": cond_id,
                        "name": condition["name"],
                        "type": "condition"
                    }
                    
                    # Add probability if provided
                    if probabilities and cond_id in probabilities:
                        node["probability"] = probabilities[cond_id]
                        
                    graph["nodes"].append(node)
            
        return graph
    
    def connect_symptoms_to_conditions(self, 
                                     graph: Dict[str, Any], 
                                     symptom_ids: List[str],
                                     condition_ids: List[str]) -> Dict[str, Any]:
        """
        Create links between symptoms and their related conditions
        
        Args:
            graph: Current graph structure
            symptom_ids: List of symptom IDs in the graph
            condition_ids: List of condition IDs in the graph
            
        Returns:
            Updated graph with links
        """
        # For each symptom-condition pair, check relationship and add link
        for sym_id in symptom_ids:
            if sym_id not in self.symptoms:
                continue
                
            for cond_id in condition_ids:
                if cond_id not in self.conditions:
                    continue
                    
                condition = self.conditions[cond_id]
                
                # Check if symptom is primary or secondary for this condition
                link_strength = "weak"
                if "primary_symptoms" in condition and sym_id in condition["primary_symptoms"]:
                    link_strength = "strong"
                elif "secondary_symptoms" in condition and sym_id in condition["secondary_symptoms"]:
                    link_strength = "medium"
                else:
                    # No direct relationship
                    continue
                
                # Create unique link ID
                link_id = f"{sym_id}-{cond_id}"
                
                # Check if link already exists
                if not any(l["id"] == link_id for l in graph["links"]):
                    graph["links"].append({
                        "id": link_id,
                        "source": sym_id,
                        "target": cond_id,
                        "strength": link_strength
                    })
            
        return graph
    
    def add_question_node(self, 
                         graph: Dict[str, Any], 
                         question: Dict[str, Any],
                         answer: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a question node to the graph
        
        Args:
            graph: Current graph structure
            question: Question object to add
            answer: Optional answer to the question
            
        Returns:
            Updated graph
        """
        question_id = question["id"]
        
        # Check if node already exists
        existing_node = next((n for n in graph["nodes"] if n["id"] == question_id), None)
        
        if existing_node:
            # Update existing node if answer provided
            if answer:
                existing_node["answer"] = answer
        else:
            # Create new node
            node = {
                "id": question_id,
                "name": question["text"],
                "type": "question"
            }
            
            # Add answer if provided
            if answer:
                node["answer"] = answer
                
            graph["nodes"].append(node)
        
        # Connect question to related conditions
        if answer and answer.lower() in ["yes", "no"]:
            condition_ids = question["condition_indication"].get(answer.lower(), [])
            
            for cond_id in condition_ids:
                # Create unique link ID
                link_id = f"{question_id}-{cond_id}"
                
                # Set link type based on answer
                link_type = "positive" if answer.lower() == "yes" else "negative"
                
                # Check if link already exists
                if not any(l["id"] == link_id for l in graph["links"]):
                    graph["links"].append({
                        "id": link_id,
                        "source": question_id,
                        "target": cond_id,
                        "type": link_type
                    })
        
        # Connect question to target symptom if applicable
        target_symptom = question.get("target_symptom")
        if target_symptom:
            # Create unique link ID
            link_id = f"{question_id}-{target_symptom}"
            
            # Check if link already exists
            if not any(l["id"] == link_id for l in graph["links"]):
                graph["links"].append({
                    "id": link_id,
                    "source": question_id,
                    "target": target_symptom,
                    "type": "explores"
                })
        
        return graph
    
    def update_condition_probabilities(self, 
                                     graph: Dict[str, Any],
                                     probabilities: Dict[str, float]) -> Dict[str, Any]:
        """
        Update condition node probabilities
        
        Args:
            graph: Current graph structure
            probabilities: Dictionary of condition probabilities
            
        Returns:
            Updated graph
        """
        # Update probability for each condition node
        for node in graph["nodes"]:
            if node["type"] == "condition" and node["id"] in probabilities:
                node["probability"] = probabilities[node["id"]]
                
                # Add visual indicators based on probability
                if probabilities[node["id"]] >= 0.5:
                    node["status"] = "likely"
                elif probabilities[node["id"]] >= 0.3:
                    node["status"] = "possible"
                else:
                    node["status"] = "unlikely"
        
        return graph