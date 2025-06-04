from sklearn.metrics import precision_score, recall_score, f1_score
import re
from typing import List, Dict, Any
from langchain_core.documents import Document

class ModelEvaluator:
            def __init__(self):
                self.sample_data = self._create_sample_dataset()
                self.metrics = {
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                    "mrr": 0
                }
            
            def _create_sample_dataset(self):
                """Create a small evaluation dataset with questions, contexts, and answers"""
                return [
                    {
                        "question": "What is the capital of France?",
                        "context": "France is a country in Europe. Its capital is Paris, which is known for its art, fashion, and culture.",
                        "answer": "Paris",
                        "page": 1
                    },
                    {
                        "question": "When was the Declaration of Independence signed?",
                        "context": "The Declaration of Independence was signed on August 2, 1776, though it was adopted on July 4th.",
                        "answer": "August 2, 1776",
                        "page": 2
                    },
                    {
                        "question": "What is the chemical symbol for gold?",
                        "context": "Gold is a chemical element with the symbol Au (from Latin: aurum) and atomic number 79.",
                        "answer": "Au",
                        "page": 3
                    },
                    {
                        "question": "Who wrote 'Romeo and Juliet'?",
                        "context": "William Shakespeare, the famous English playwright, wrote 'Romeo and Juliet' in the late 16th century.",
                        "answer": "William Shakespeare",
                        "page": 4
                    },
                    {
                        "question": "What is the largest planet in our solar system?",
                        "context": "Jupiter is the largest planet in our solar system, with a mass more than two and a half times that of all other planets combined.",
                        "answer": "Jupiter",
                        "page": 5
                    }
                ]
            
            def _normalize_text(self, text):
                """Normalize text for comparison"""
                text = text.lower()
                text = re.sub(r'[^\w\s]', '', text)
                return text.strip()
            
            def _calculate_mrr(self, results):
                """Calculate Mean Reciprocal Rank"""
                reciprocal_ranks = []
                for res in results:
                    correct_answer = self._normalize_text(res["correct_answer"])
                    generated_answers = [self._normalize_text(ans) for ans in res["generated_answers"]]
                    
                    for rank, ans in enumerate(generated_answers, start=1):
                        if correct_answer in ans or ans in correct_answer:
                            reciprocal_ranks.append(1.0 / rank)
                            break
                    else:
                        reciprocal_ranks.append(0)
                
                return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
            
            def evaluate_model(self, query_processor, response_gen):
                """Evaluate the model on the sample dataset"""
                results = []
                all_true = []
                all_pred = []
                
                for item in self.sample_data:
                    # Simulate document retrieval by creating a mock document
                    mock_doc = Document(
                        page_content=item["context"],
                        metadata={"page": item["page"], "source": "evaluation"}
                    )
                    
                    # Generate response using the model
                    response = response_gen.generate_response(item["question"], [mock_doc])
                    
                    # Extract the direct answer part (assuming it's marked with 🎯)
                    direct_answer = ""
                    if "🎯" in response:
                        direct_answer = response.split("🎯")[1].split("\n")[0].strip("*").strip()
                    
                    # For evaluation, we'll consider both direct answer and full response
                    generated_answers = [direct_answer, response]
                    
                    results.append({
                        "question": item["question"],
                        "correct_answer": item["answer"],
                        "generated_answers": generated_answers,
                        "context": item["context"],
                        "full_response": response
                    })
                    
                    # For precision/recall/F1, check if correct answer appears in any generated answer
                    correct_normalized = self._normalize_text(item["answer"])
                    found = any(
                        correct_normalized in self._normalize_text(ans) or 
                        self._normalize_text(ans) in correct_normalized
                        for ans in generated_answers
                    )
                    
                    all_true.append(1)
                    all_pred.append(1 if found else 0)
                
                # Calculate metrics
                precision = precision_score(all_true, all_pred)
                recall = recall_score(all_true, all_pred)
                f1 = f1_score(all_true, all_pred)
                mrr = self._calculate_mrr(results)
                
                self.metrics = {
                    "precision": round(precision, 3),
                    "recall": round(recall, 3),
                    "f1": round(f1, 3),
                    "mrr": round(mrr, 3),
                    "results": results
                }
                
                return self.metrics