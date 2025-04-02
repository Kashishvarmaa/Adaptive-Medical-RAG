"""
Adaptive Med-RAG Demo Script
This script demonstrates how to use the Adaptive Med-RAG system with sample medical queries.
"""

import os
import json
from typing import List, Dict
from tabulate import tabulate
import time
import torch

# Import the Adaptive Med-RAG system
from model import AdaptiveMedRAG, MedicalRAGEvaluator

def demo_setup():
    """Set up the demo environment"""
    print("Setting up Adaptive Med-RAG demo...")
    
    # Create sample data directory if it doesn't exist
    os.makedirs("./medical_data", exist_ok=True)
    
    # Create sample medical text if none exists
    if not os.path.exists("./medical_data/sample_medical_info.txt"):
        with open("./medical_data/sample_medical_info.txt", "w") as f:
            f.write("""
# Diabetes Type 2 Treatment Guidelines

Type 2 diabetes is characterized by insulin resistance and relative insulin deficiency. Treatment typically involves lifestyle modifications, oral medications, and sometimes insulin therapy.

## First-line treatments:
- Lifestyle modifications (diet, exercise, weight loss)
- Metformin (biguanide) - Reduces hepatic glucose production and improves insulin sensitivity

## Second-line treatments (add-ons to metformin):
- Sulfonylureas - Increase insulin secretion
- DPP-4 inhibitors - Increase incretin levels
- SGLT2 inhibitors - Reduce glucose reabsorption in kidneys
- GLP-1 receptor agonists - Increase insulin secretion, reduce glucagon, slow gastric emptying
- Thiazolidinediones - Improve insulin sensitivity

## Insulin therapy:
- Usually reserved for when oral medications fail to achieve glycemic targets
- Basal insulin: Long-acting insulins like glargine, detemir
- Bolus insulin: Rapid-acting insulins like lispro, aspart

## Monitoring:
- HbA1c every 3-6 months
- Blood glucose self-monitoring
- Regular screening for complications (retinopathy, nephropathy, neuropathy)

# COVID-19 Treatment Overview

COVID-19 treatment approaches have evolved significantly since the beginning of the pandemic. Current strategies focus on severity-based management.

## Mild to moderate disease:
- Symptomatic treatment: antipyretics, analgesics, hydration
- Monoclonal antibodies for high-risk patients
- Antivirals (remdesivir, molnupiravir, Paxlovid) within 5 days of symptom onset for eligible patients

## Severe disease:
- Oxygen therapy
- Dexamethasone or other corticosteroids
- Remdesivir
- Immunomodulators (tocilizumab, baricitinib) in selected patients
- Anticoagulation for thromboprophylaxis

## Prevention:
- Vaccination remains the most effective strategy
- Updated boosters targeting circulating variants
- Masking and social distancing in high-risk settings

# Hypertension Management

Hypertension, defined as blood pressure ≥130/80 mmHg, is a major risk factor for cardiovascular disease. Management includes both non-pharmacological and pharmacological approaches.

## Non-pharmacological interventions:
- DASH diet (rich in fruits, vegetables, whole grains, low-fat dairy)
- Sodium restriction (<2300 mg/day)
- Regular physical activity (150 minutes/week)
- Weight management
- Alcohol limitation
- Smoking cessation

## First-line medications:
- Thiazide diuretics
- ACE inhibitors
- Angiotensin II receptor blockers (ARBs)
- Calcium channel blockers

## Treatment targets:
- General population: <130/80 mmHg
- Older adults (>65): individualized targets
- Comorbid conditions may alter targets

## Monitoring:
- Regular blood pressure measurements
- Assessment of medication adherence
- Monitoring for adverse effects
- Periodic laboratory evaluations
            """)
    
    # Create sample test queries file
    if not os.path.exists("./test_queries.txt"):
        with open("./test_queries.txt", "w") as f:
            f.write("""
What are the first-line treatments for type 2 diabetes?
How does COVID-19 treatment differ between mild and severe cases?
What medications are commonly prescribed for hypertension?
Can you explain the mechanism of action of metformin in diabetes?
What are the treatment targets for hypertension in older adults?
How do SGLT2 inhibitors work in diabetes management?
            """.strip())
    
    print("Demo setup complete!")
    return True

def run_sample_queries(med_rag: AdaptiveMedRAG, queries: List[str]):
    """Run sample queries through different strategies"""
    results = []
    
    strategies = ["standard", "soar", "cot", "contrastive"]
    
    print("Running sample queries with different strategies...")
    for query in queries:
        row = {"Query": query}
        
        for strategy in strategies:
            print(f"Processing query with {strategy} strategy: {query[:40]}...")
            response = med_rag.answer_medical_query(query, strategy=strategy)
            row[strategy] = response["answer"]
        
        results.append(row)
    
    # Display results in a tabular format
    for i, row in enumerate(results, 1):
        print(f"\n\n--- QUERY {i}: {row['Query']} ---")
        for strategy in strategies:
            print(f"\n[{strategy.upper()}]:")
            print(row[strategy])
        print("\n" + "="*80)
    
    return results

def compare_strategies_detailed(med_rag: AdaptiveMedRAG, query: str):
    """Detailed comparison of all strategies on a single query"""
    print(f"\nDetailed comparison for query: {query}")
    
    strategies = ["standard", "soar", "cot", "contrastive", "consistency"]
    results = {}
    
    for strategy in strategies:
        print(f"Processing with {strategy} strategy...")
        start_time = time.time()
        response = med_rag.answer_medical_query(query, strategy=strategy)
        elapsed = time.time() - start_time
        
        results[strategy] = {
            "answer": response["answer"],
            "time": elapsed,
            "retrieval_info": response["retrieval"],
            "sources": response["sources"]
        }
    
    # Display comparison
    print("\n" + "="*100)
    print(f"QUERY: {query}")
    print("="*100)
    
    for strategy, data in results.items():
        print(f"\n--- {strategy.upper()} STRATEGY ---")
        print(f"Time: {data['time']:.2f} seconds")
        print(f"Retrieval: {data['retrieval_info']}")
        print(f"Sources: {', '.join(data['sources'][:3])}{'...' if len(data['sources']) > 3 else ''}")
        print(f"\nAnswer:\n{data['answer']}")
        print("-"*100)
    
    return results

def personalization_demo(med_rag: AdaptiveMedRAG):
    """Demonstrate personalization capabilities"""
    print("\nDemonstrating personalization capabilities...")
    
    # Create two test users
    users = ["researcher_user", "clinician_user"]
    
    # Sample queries for each user to establish preferences
    researcher_queries = [
        "What are the latest research findings on SGLT2 inhibitors?",
        "Explain the molecular pathway of insulin resistance in type 2 diabetes",
        "What evidence supports the use of dexamethasone in severe COVID-19?"
    ]
    
    clinician_queries = [
        "What is the recommended dosage of metformin for an elderly patient?",
        "How should hypertension treatment be adjusted for patients with CKD?",
        "What are the contraindications for GLP-1 agonists in diabetes management?"
    ]
    
    # Build user profiles
    print("Building user profiles...")
    for query in researcher_queries:
        response = med_rag.personalized_medical_query(query, users[0])
        print(f"Processed researcher query: {query[:40]}...")
    
    for query in clinician_queries:
        response = med_rag.personalized_medical_query(query, users[1])
        print(f"Processed clinician query: {query[:40]}...")
    
    # Test query for both users
    test_query = "What should be considered when prescribing SGLT2 inhibitors?"
    
    print("\nTesting personalization with the same query for both users...")
    researcher_response = med_rag.personalized_medical_query(test_query, users[0])
    clinician_response = med_rag.personalized_medical_query(test_query, users[1])
    
    print("\n" + "="*100)
    print("PERSONALIZATION RESULTS - SAME QUERY, DIFFERENT USERS")
    print("="*100)
    print(f"\nQuery: {test_query}")
    
    print(f"\n--- RESEARCHER USER RESPONSE ---")
    print(researcher_response["answer"])
    print(f"\nUser topics: {researcher_response['topics']}")
    
    print(f"\n--- CLINICIAN USER RESPONSE ---")
    print(clinician_response["answer"])
    print(f"\nUser topics: {clinician_response['topics']}")
    
    return {"researcher": researcher_response, "clinician": clinician_response}

def main():
    """Main demo function"""
    # Setup
    demo_setup()
    
    # Initialize the system
    config = {
        "base_model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "medical_lm": "epfl-llm/meditron-7b",  # Using a medical LLM
        "embedding_model": "pritamdeka/S-PubMedBert-MS-MARCO",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./medical_data",
        "index_path": "./vector_indexes",
        "cache_dir": "./retrieval_cache",
        "max_length": 512,
        "chunk_size": 384,
        "chunk_overlap": 128,
        "top_k_default": 5
    }
    
    print("\nInitializing Adaptive Med-RAG system...")
    med_rag = AdaptiveMedRAG(config)
    
    # Index documents
    print("\nIndexing medical documents...")
    med_rag.index_documents()
    
    # Load test queries
    with open("./test_queries.txt", 'r') as f:
        queries = [line.strip() for line in f if line.strip()]
    
    # Demo 1: Run sample queries with different strategies
    results = run_sample_queries(med_rag, queries[:3])  # Use first 3 queries
    
    # Demo 2: Compare all strategies on a specific query
    detailed_results = compare_strategies_detailed(
        med_rag, 
        "What is the mechanism of action of SGLT2 inhibitors and when should they be prescribed?"
    )
    
    # Demo 3: Personalization capabilities
    personalization_results = personalization_demo(med_rag)
    
    print("\n" + "="*100)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*100)
    
    return {
        "sample_queries": results,
        "detailed_comparison": detailed_results,
        "personalization": personalization_results
    }

if __name__ == "__main__":
    main()