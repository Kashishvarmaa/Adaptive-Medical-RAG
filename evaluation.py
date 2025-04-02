"""
Adaptive Med-RAG Evaluation Script
This script evaluates the Adaptive Med-RAG system on public medical question answering datasets.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from typing import Dict, List, Tuple, Any
from tqdm.auto import tqdm
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# Import the Adaptive Med-RAG system
from model import AdaptiveMedRAG, MedicalRAGEvaluator

class MedicalQAEvaluator:
    """Comprehensive evaluator for medical QA systems"""
    
    def __init__(self, med_rag: AdaptiveMedRAG, output_dir: str = "./evaluation_results"):
        self.med_rag = med_rag
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.rouge = Rouge()
        
    def prepare_dataset(self, name: str, path: str = None):
        """Prepare evaluation dataset"""
        # If path not provided, use sample datasets
        if not path:
            if name.lower() == "pubmedqa":
                return self._prepare_pubmedqa_sample()
            elif name.lower() == "medical_qa":
                return self._prepare_medical_qa_sample()
            elif name.lower() == "emrqa":
                return self._prepare_emrqa_sample()
            else:
                raise ValueError(f"Unknown dataset name: {name}")
        
        # Load dataset from file
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        elif path.endswith('.json'):
            df = pd.read_json(path)
        else:
            raise ValueError("Dataset file must be CSV or JSON")
            
        return df
    
    def _prepare_pubmedqa_sample(self):
        """Create a sample of PubMedQA format for testing"""
        # Create directory if it doesn't exist
        os.makedirs("./sample_datasets", exist_ok=True)
        
        sample_path = "./sample_datasets/pubmedqa_sample.csv"
        
        # If sample already exists, load it
        if os.path.exists(sample_path):
            return pd.read_csv(sample_path)
            
        # Create sample dataset
        data = {
            "question": [
                "Does metformin reduce cardiovascular events in type 2 diabetes?",
                "Is dexamethasone effective in treating severe COVID-19?",
                "Can SGLT2 inhibitors improve heart failure outcomes?",
                "Are ACE inhibitors beneficial in diabetic nephropathy?",
                "Does aspirin prevent colorectal cancer?"
            ],
            "answer": [
                "yes",
                "yes",
                "yes",
                "yes",
                "yes"
            ],
            "context": [
                "Metformin has been shown to reduce cardiovascular events in patients with type 2 diabetes in multiple clinical trials.",
                "Dexamethasone significantly reduced mortality in COVID-19 patients requiring oxygen or ventilatory support.",
                "Recent clinical trials demonstrate that SGLT2 inhibitors reduce hospitalization for heart failure and cardiovascular death.",
                "ACE inhibitors slow the progression of diabetic nephropathy and reduce albuminuria in patients with diabetes.",
                "Regular aspirin use is associated with reduced incidence of colorectal cancer and mortality."
            ]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(sample_path, index=False)
        return df
    
    def _prepare_medical_qa_sample(self):
        """Create a sample of general medical QA format for testing"""
        sample_path = "./sample_datasets/medical_qa_sample.csv"
        
        # If sample already exists, load it
        if os.path.exists(sample_path):
            return pd.read_csv(sample_path)
            
        # Create sample dataset
        data = {
            "question": [
                "What is the first-line treatment for type 2 diabetes?",
                "How do SGLT2 inhibitors work in diabetes management?",
                "What are the major side effects of metformin?",
                "When should insulin therapy be initiated in type 2 diabetes?",
                "What dietary changes are recommended for hypertension management?"
            ],
            "answer": [
                "Metformin combined with lifestyle modifications including diet, exercise, and weight loss is the first-line treatment for type 2 diabetes.",
                "SGLT2 inhibitors work by preventing glucose reabsorption in the kidneys, leading to increased glucose excretion in urine and lowering blood glucose levels.",
                "The major side effects of metformin include gastrointestinal issues such as diarrhea, nausea, abdominal discomfort, and in rare cases, lactic acidosis.",
                "Insulin therapy should be initiated in type 2 diabetes when oral medications fail to achieve glycemic targets or during periods of acute illness, surgery, or pregnancy.",
                "Dietary recommendations for hypertension include the DASH diet (rich in fruits, vegetables, whole grains, low-fat dairy), sodium restriction, and limiting alcohol consumption."
            ]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(sample_path, index=False)
        return df
    
    def _prepare_emrqa_sample(self):
        """Create a sample of EMR-based QA format for testing"""
        sample_path = "./sample_datasets/emrqa_sample.csv"
        
        # If sample already exists, load it
        if os.path.exists(sample_path):
            return pd.read_csv(sample_path)
            
        # Create sample dataset
        data = {
            "question": [
                "What medications is the patient currently taking for diabetes?",
                "When was the patient's last HbA1c test performed and what was the result?",
                "Does the patient have any history of diabetic retinopathy?",
                "What was the patient's blood pressure at the last visit?",
                "Has the patient experienced any hypoglycemic episodes?"
            ],
            "context": [
                "Patient is a 62-year-old male with type 2 diabetes diagnosed 8 years ago. Current medications include metformin 1000mg BID, empagliflozin 10mg daily, and lisinopril 20mg daily for hypertension.",
                "Patient's last HbA1c was 7.2% performed on January 15, 2023. Previous result from October 2022 was 7.8%.",
                "Patient underwent diabetic retinopathy screening on March 2, 2023. Report indicates mild non-proliferative diabetic retinopathy in both eyes. Ophthalmology follow-up scheduled in 6 months.",
                "Vital signs at visit on April 10, 2023: BP 138/82, HR 72, RR 16, Temp 98.2°F, SpO2 98% on room air.",
                "Patient reports one episode of hypoglycemia (blood glucose 65 mg/dL) last month after missing a meal. No severe hypoglycemic events requiring assistance."
            ],
            "answer": [
                "metformin 1000mg BID, empagliflozin 10mg daily",
                "January 15, 2023, result was 7.2%",
                "yes, mild non-proliferative diabetic retinopathy in both eyes",
                "138/82",
                "yes, one episode with blood glucose 65 mg/dL after missing a meal"
            ]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(sample_path, index=False)
        return df
    
    def calculate_metrics(self, actual: List[str], predicted: List[str]) -> Dict[str, float]:
        """Calculate evaluation metrics for yes/no or categorical answers"""
        # For yes/no questions in datasets like PubMedQA
        if all(a.lower() in ['yes', 'no', 'maybe'] for a in actual):
            y_true = [1 if a.lower() == 'yes' else 0 for a in actual]
            y_pred = [1 if p.lower() == 'yes' else 0 for p in predicted]
            
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0)
            }
        else:
            # For longer, text-based answers, use semantic metrics
            metrics = self.calculate_semantic_metrics(actual, predicted)
            
        return metrics
    
    def calculate_semantic_metrics(self, actual: List[str], predicted: List[str]) -> Dict[str, float]:
        """Calculate semantic similarity metrics for text answers"""
        rouge_scores = {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}
        bleu_scores = 0
        
        for i, (act, pred) in enumerate(zip(actual, predicted)):
            # Calculate ROUGE scores
            try:
                rouge_result = self.rouge.get_scores(pred, act)[0]
                for k in rouge_scores:
                    rouge_scores[k] += rouge_result[k]['f']
            except:
                # Handle very short answers where ROUGE might fail
                pass
            
            # Calculate BLEU score
            try:
                reference = [act.lower().split()]
                candidate = pred.lower().split()
                bleu = sentence_bleu(reference, candidate)
                bleu_scores += bleu
            except:
                pass
        
        # Average scores
        n = len(actual)
        metrics = {
            'rouge-1': rouge_scores['rouge-1'] / n if n > 0 else 0,
            'rouge-2': rouge_scores['rouge-2'] / n if n > 0 else 0,
            'rouge-l': rouge_scores['rouge-l'] / n if n > 0 else 0,
            'bleu': bleu_scores / n if n > 0 else 0
        }
        
        return metrics
    
    def evaluate_strategy(self, dataset: pd.DataFrame, strategy: str) -> Dict[str, Any]:
        """Evaluate a specific retrieval strategy on the dataset"""
        results = {
            'strategy': strategy,
            'predictions': [],
            'retrieval_info': [],
            'timings': [],
            'sources': []
        }
        
        for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Evaluating {strategy}"):
            query = row['question']
            
            # Time the response
            start_time = time.time()
            response = self.med_rag.answer_medical_query(query, strategy=strategy)
            elapsed = time.time() - start_time
            
            # Store results
            results['predictions'].append(response['answer'])
            results['retrieval_info'].append(response['retrieval'])
            results['timings'].append(elapsed)
            results['sources'].append(response['sources'])
        
        # Calculate evaluation metrics
        actual_answers = dataset['answer'].tolist()
        metrics = self.calculate_metrics(actual_answers, results['predictions'])
        results['metrics'] = metrics
        
        return results
    
    def compare_strategies(self, dataset: pd.DataFrame, strategies: List[str] = None) -> Dict[str, Any]:
        """Compare different retrieval strategies on the same dataset"""
        if strategies is None:
            strategies = ["standard", "soar", "cot", "contrastive", "consistency"]
        
        comparison = {
            'dataset_size': len(dataset),
            'strategies': {}
        }
        
        for strategy in strategies:
            print(f"\nEvaluating {strategy} strategy...")
            results = self.evaluate_strategy(dataset, strategy)
            comparison['strategies'][strategy] = results
            
            # Print preliminary results
            metrics = results['metrics']
            print(f"  - Average response time: {sum(results['timings']) / len(results['timings']):.2f} seconds")
            for metric, value in metrics.items():
                print(f"  - {metric}: {value:.4f}")
        
        # Save detailed comparison results
        output_file = os.path.join(self.output_dir, "strategy_comparison.json")
        with open(output_file, 'w') as f:
            # Convert NumPy values to native Python types for JSON serialization
            json_comparison = self._prepare_for_json(comparison)
            json.dump(json_comparison, f, indent=2)
        
        print(f"\nDetailed comparison saved to {output_file}")
        
        # Generate visualizations
        self._generate_comparison_visualizations(comparison)
        
        return comparison
    
    def _prepare_for_json(self, obj):
        """Convert NumPy values to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def _generate_comparison_visualizations(self, comparison: Dict):
        """Generate visualizations comparing strategies"""
        strategies = list(comparison['strategies'].keys())
        
        # Set up plotting style
        plt.style.use('ggplot')
        sns.set_palette("deep")
        
        # 1. Performance metrics comparison
        metrics_fig, ax = plt.subplots(figsize=(10, 6))
        metrics_data = []
        
        for strategy in strategies:
            for metric, value in comparison['strategies'][strategy]['metrics'].items():
                metrics_data.append({
                    'Strategy': strategy,
                    'Metric': metric,
                    'Score': value
                })
        
        metrics_df = pd.DataFrame(metrics_data)
        sns.barplot(x='Metric', y='Score', hue='Strategy', data=metrics_df, ax=ax)
        ax.set_title('Performance Metrics by Strategy')
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'metrics_comparison.png'))
        
        # 2. Response time comparison
        time_fig, ax = plt.subplots(figsize=(8, 5))
        time_data = {strategy: comparison['strategies'][strategy]['timings'] for strategy in strategies}
        sns.boxplot(data=pd.DataFrame(time_data), ax=ax)
        ax.set_title('Response Time by Strategy')
        ax.set_ylabel('Time (seconds)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'response_time_comparison.png'))
        
        # 3. Heatmap of strategy performance by metric
        heatmap_data = []
        for strategy in strategies:
            metrics = comparison['strategies'][strategy]['metrics']
            heatmap_data.append([metrics.get(m, 0) for m in ['accuracy', 'f1', 'rouge-1', 'rouge-l', 'bleu']])
        
        heatmap_fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            heatmap_data, 
            annot=True, 
            fmt='.3f',
            xticklabels=['Accuracy', 'F1', 'ROUGE-1', 'ROUGE-L', 'BLEU'],
            yticklabels=strategies,
            cmap='YlGnBu',
            ax=ax
        )
        ax.set_title('Strategy Performance Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'strategy_heatmap.png'))
        
        plt.close('all')
    
    def analyze_retrieval_patterns(self, comparison: Dict) -> Dict:
        """Analyze patterns in retrieval behavior across strategies"""
        analysis = {
            'source_usage': {},
            'retrieval_depth': {},
            'time_vs_performance': {}
        }
        
        for strategy, results in comparison['strategies'].items():
            # Analyze source usage
            all_sources = []
            for sources in results['sources']:
                all_sources.extend(sources)
            
            source_counts = {}
            for source in all_sources:
                if source not in source_counts:
                    source_counts[source] = 0
                source_counts[source] += 1
            
            analysis['source_usage'][strategy] = source_counts
            
            # Analyze retrieval depth
            if 'docs_retrieved' in results['retrieval_info'][0]:
                depths = [info['docs_retrieved'] for info in results['retrieval_info']]
                analysis['retrieval_depth'][strategy] = {
                    'mean': np.mean(depths),
                    'median': np.median(depths),
                    'min': np.min(depths),
                    'max': np.max(depths)
                }
            
            # Time vs performance correlation
            if len(results['timings']) > 1:
                performance_metric = results['metrics'].get('f1', results['metrics'].get('rouge-l', 0))
                analysis['time_vs_performance'][strategy] = {
                    'avg_time': np.mean(results['timings']),
                    'performance': performance_metric
                }
        
        # Save analysis results
        output_file = os.path.join(self.output_dir, "retrieval_analysis.json")
        with open(output_file, 'w') as f:
            json.dump(self._prepare_for_json(analysis), f, indent=2)
        
        return analysis
    
    def evaluate_cross_domain(self, datasets: Dict[str, pd.DataFrame], strategy: str = "soar") -> Dict:
        """Evaluate performance across different medical domains"""
        cross_domain = {
            'strategy': strategy,
            'domains': {}
        }
        
        for domain, dataset in datasets.items():
            print(f"\nEvaluating {domain} domain...")
            results = self.evaluate_strategy(dataset, strategy)
            cross_domain['domains'][domain] = results
            
            # Print domain results
            metrics = results['metrics']
            print(f"  - Domain: {domain}")
            for metric, value in metrics.items():
                print(f"  - {metric}: {value:.4f}")
        
        # Generate cross-domain visualization
        self._generate_cross_domain_visualization(cross_domain)
        
        return cross_domain
    
    def _generate_cross_domain_visualization(self, cross_domain: Dict):
        """Generate visualizations for cross-domain performance"""
        domains = list(cross_domain['domains'].keys())
        metrics = ['accuracy', 'f1', 'rouge-1', 'rouge-l', 'bleu']
        
        # Set up the figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        data = []
        for domain in domains:
            domain_metrics = cross_domain['domains'][domain]['metrics']
            for metric in metrics:
                if metric in domain_metrics:
                    data.append({
                        'Domain': domain,
                        'Metric': metric,
                        'Score': domain_metrics[metric]
                    })
        
        # Create the grouped bar chart
        df = pd.DataFrame(data)
        sns.barplot(x='Domain', y='Score', hue='Metric', data=df, ax=ax)
        
        ax.set_title(f'Performance Across Medical Domains ({cross_domain["strategy"]} strategy)')
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cross_domain_performance.png'))
        plt.close()
    
    def evaluate_complex_queries(self, complex_queries: List[Dict], strategy: str = "cot") -> Dict:
        """Evaluate performance on especially complex medical queries"""
        results = {
            'strategy': strategy,
            'queries': [],
            'success_rate': 0.0
        }
        
        for query_item in tqdm(complex_queries, desc="Evaluating complex queries"):
            query = query_item['question']
            expected = query_item['answer']
            
            # Get response using the specified strategy
            response = self.med_rag.answer_medical_query(query, strategy=strategy, explain=True)
            
            # Evaluate correctness - using human judgment criteria if available
            if 'criteria' in query_item:
                criteria = query_item['criteria']
                correctness = self._evaluate_against_criteria(response['answer'], criteria)
            else:
                # Fall back to semantic metrics
                metrics = self.calculate_semantic_metrics([expected], [response['answer']])
                correctness = metrics['rouge-l'] > 0.5  # Simplified threshold
            
            # Store results
            results['queries'].append({
                'question': query,
                'expected': expected,
                'actual': response['answer'],
                'correct': correctness,
                'retrieval_info': response['retrieval']
            })
        
        # Calculate success rate
        if results['queries']:
            results['success_rate'] = sum(q['correct'] for q in results['queries']) / len(results['queries'])
        
        return results
    
    def _evaluate_against_criteria(self, answer: str, criteria: List[str]) -> bool:
        """Evaluate if an answer meets all specified criteria"""
        # Simple implementation - for a production system, use a more sophisticated approach
        for criterion in criteria:
            if criterion.lower() not in answer.lower():
                return False
        return True
    
    def run_comprehensive_evaluation(self):
        """Run a comprehensive evaluation suite"""
        print("Starting comprehensive evaluation of Adaptive Med-RAG...")
        
        # 1. Basic strategy comparison on PubMedQA
        pubmedqa = self.prepare_dataset("pubmedqa")
        pubmedqa_results = self.compare_strategies(pubmedqa)
        
        # 2. General medical QA evaluation
        medical_qa = self.prepare_dataset("medical_qa")
        medical_qa_results = self.compare_strategies(medical_qa, strategies=["standard", "soar", "cot"])
        
        # 3. Cross-domain evaluation
        domains = {
            "general": medical_qa,
            "pubmed": pubmedqa,
            "emr": self.prepare_dataset("emrqa")
        }
        cross_domain_results = self.evaluate_cross_domain(domains)
        
        # 4. Complex query evaluation
        complex_queries = [
            {
                "question": "What is the pathophysiological mechanism linking SGLT2 inhibitors to reduced cardiovascular events in diabetic patients with heart failure?",
                "answer": "SGLT2 inhibitors reduce cardiovascular events through multiple mechanisms including improved cardiac metabolism, reduced cardiac preload and afterload, inhibition of sodium-hydrogen exchange, reduced inflammation, and improved cardiac energy metabolism.",
                "criteria": ["cardiac", "metabolism", "preload", "inflammation"]
            },
            {
                "question": "Compare and contrast the benefits and risks of GLP-1 receptor agonists versus SGLT2 inhibitors in patients with type 2 diabetes and chronic kidney disease.",
                "answer": "GLP-1 receptor agonists provide superior glycemic control and weight loss but may cause GI side effects, while SGLT2 inhibitors offer cardiovascular and renal protection with risk of genital infections and DKA. For diabetic CKD patients, both offer renal protection but SGLT2 inhibitors have stronger evidence for slowing CKD progression.",
                "criteria": ["glycemic", "weight", "renal", "protection"]
            }
        ]
        complex_results = self.evaluate_complex_queries(complex_queries)
        
        # 5. Analyze retrieval patterns
        retrieval_analysis = self.analyze_retrieval_patterns(pubmedqa_results)
        
        # Compile comprehensive report
        comprehensive_results = {
            "pubmedqa": pubmedqa_results,
            "medical_qa": medical_qa_results,
            "cross_domain": cross_domain_results,
            "complex_queries": complex_results,
            "retrieval_analysis": retrieval_analysis
        }
        
        # Save comprehensive results
        output_file = os.path.join(self.output_dir, "comprehensive_evaluation.json")
        with open(output_file, 'w') as f:
            json.dump(self._prepare_for_json(comprehensive_results), f, indent=2)
        
        print(f"\nComprehensive evaluation complete. Results saved to {output_file}")
        return comprehensive_results


def main():
    """Main evaluation function"""
    # Initialize the Med-RAG system
    config = {
        "base_model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "medical_lm": "epfl-llm/meditron-7b",
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
    
    print("Initializing Adaptive Med-RAG system...")
    med_rag = AdaptiveMedRAG(config)
    
    # Make sure documents are indexed
    print("Indexing medical documents...")
    med_rag.index_documents()
    
    # Initialize evaluator
    evaluator = MedicalQAEvaluator(med_rag)
    
    # Run comprehensive evaluation
    comprehensive_results = evaluator.run_comprehensive_evaluation()
    
    # Generate summary report
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    # Strategy comparison summary
    print("\nStrategy Performance on PubMedQA:")
    for strategy, results in comprehensive_results["pubmedqa"]["strategies"].items():
        metrics = results["metrics"]
        print(f"  - {strategy.upper()}: F1={metrics.get('f1', 0):.4f}, Rouge-L={metrics.get('rouge-l', 0):.4f}")
    
    # Cross-domain summary
    print("\nCross-Domain Performance:")
    for domain, results in comprehensive_results["cross_domain"]["domains"].items():
        metrics = results["metrics"]
        print(f"  - {domain.upper()}: F1={metrics.get('f1', 0):.4f}, Rouge-L={metrics.get('rouge-l', 0):.4f}")
    
    # Complex query performance
    print(f"\nComplex Query Success Rate: {comprehensive_results['complex_queries']['success_rate']:.2f}")
    
    print("\nDetailed results and visualizations are available in the evaluation_results directory.")
    print("="*80)
    
    return comprehensive_results


if __name__ == "__main__":
    import time
    import torch
    main()