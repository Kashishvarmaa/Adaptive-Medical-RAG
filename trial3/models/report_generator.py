import os
import datetime
import json
from typing import Dict, Any

class ReportGenerator:
    """
    Generates and saves detailed medical consultation reports
    """
    
    def __init__(self, report_dir: str = 'reports'):
        """
        Initialize the report generator
        
        Args:
            report_dir: Directory where reports will be saved
        """
        self.report_dir = report_dir
        
        # Create the reports directory if it doesn't exist
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
    
    def generate_report(self, 
                       user_input: str,
                       identified_symptoms: list,
                       questions_answers: Dict[str, str],
                       reasoning_steps: list,
                       condition_probabilities: Dict[str, float],
                       conclusion: Dict[str, Any]) -> str:
        """
        Generate a detailed medical consultation report
        
        Args:
            user_input: Original user symptoms description
            identified_symptoms: List of identified symptoms
            questions_answers: Dictionary of questions and their answers
            reasoning_steps: List of reasoning steps from the diagnostic process
            condition_probabilities: Dictionary of conditions and their probabilities
            conclusion: Conclusion dictionary with diagnosis and recommendations
            
        Returns:
            Path to the saved report file
        """
        # Generate timestamp for the report
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a unique filename
        filename = f"medical_report_{timestamp}.txt"
        filepath = os.path.join(self.report_dir, filename)
        
        # Generate report content
        report_content = self._format_report(
            user_input,
            identified_symptoms,
            questions_answers,
            reasoning_steps,
            condition_probabilities,
            conclusion
        )
        
        # Save the report
        with open(filepath, 'w') as f:
            f.write(report_content)
            
        return filepath
    
    def _format_report(self,
                      user_input: str,
                      identified_symptoms: list,
                      questions_answers: Dict[str, str],
                      reasoning_steps: list,
                      condition_probabilities: Dict[str, float],
                      conclusion: Dict[str, Any]) -> str:
        """
        Format the report content
        
        Args:
            user_input: Original user symptoms description
            identified_symptoms: List of identified symptoms
            questions_answers: Dictionary of questions and their answers
            reasoning_steps: List of reasoning steps from the diagnostic process
            condition_probabilities: Dictionary of conditions and their probabilities
            conclusion: Conclusion dictionary with diagnosis and recommendations
            
        Returns:
            Formatted report content
        """
        # Create report header
        report = "="*80 + "\n"
        report += " "*30 + "MEDICAL CONSULTATION REPORT\n"
        report += "="*80 + "\n\n"
        
        # Add timestamp
        report += f"Date and Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add original symptoms description
        report += "ORIGINAL DESCRIPTION\n"
        report += "-"*80 + "\n"
        report += f"{user_input}\n\n"
        
        # Add identified symptoms
        report += "IDENTIFIED SYMPTOMS\n"
        report += "-"*80 + "\n"
        for symptom in identified_symptoms:
            report += f"- {symptom}\n"
        report += "\n"
        
        # Add questions and answers
        report += "CONSULTATION DIALOGUE\n"
        report += "-"*80 + "\n"
        for question, answer in questions_answers.items():
            report += f"Q: {question}\n"
            report += f"A: {answer}\n\n"
        
        # Add reasoning steps
        report += "DIAGNOSTIC REASONING\n"
        report += "-"*80 + "\n"
        for step in reasoning_steps:
            step_type = step.get('step_type', 'general')
            content = step.get('content', '')
            
            if step_type == 'identification':
                report += f"[Identification] {content}\n"
            elif step_type == 'association':
                report += f"[Association] {content}\n"
            elif step_type == 'refinement':
                report += f"[Refinement] {content}\n"
            else:
                report += f"{content}\n"
        report += "\n"
        
        # Add condition probabilities
        report += "DIFFERENTIAL DIAGNOSIS\n"
        report += "-"*80 + "\n"
        
        # Sort conditions by probability (descending)
        sorted_conditions = sorted(condition_probabilities.items(), key=lambda x: x[1], reverse=True)
        
        for condition_id, probability in sorted_conditions:
            report += f"- {condition_id}: {probability:.2f} probability\n"
        report += "\n"
        
        # Add conclusion
        report += "DIAGNOSIS AND RECOMMENDATIONS\n"
        report += "-"*80 + "\n"
        report += f"Diagnosis: {conclusion.get('diagnosis', 'Unknown')}\n"
        report += f"Confidence: {conclusion.get('confidence', 'low')}\n"
        report += f"Urgency: {conclusion.get('urgency', 'Unknown')}\n\n"
        
        report += "Explanation:\n"
        report += f"{conclusion.get('explanation', 'No explanation available.')}\n\n"
        
        report += "Treatment Approaches:\n"
        for treatment in conclusion.get('treatment_approaches', []):
            report += f"- {treatment}\n"
        report += "\n"
        
        # Add differential diagnoses
        report += "Alternative Diagnoses:\n"
        for diff_diag in conclusion.get('differential_diagnoses', []):
            report += f"- {diff_diag.get('condition', 'Unknown')}: {diff_diag.get('probability', 0):.2f} probability\n"
        report += "\n"
        
        # Add disclaimer
        report += "DISCLAIMER\n"
        report += "-"*80 + "\n"
        report += f"{conclusion.get('disclaimer', 'This is not a definitive medical diagnosis. Please consult with a healthcare professional.')}\n\n"
        
        report += "="*80 + "\n"
        report += " "*15 + "GENERATED BY ADAPTIVE MED-RAG SYSTEM - FOR RESEARCH PURPOSES ONLY\n"
        report += "="*80 + "\n"
        
        return report