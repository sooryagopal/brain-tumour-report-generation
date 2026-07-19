"""
Evaluation Metrics: BLEU-1, ROUGE-L, BERTScore
Used to evaluate quality of generated reports vs reference reports.
"""


def compute_metrics(generated: str, reference: str) -> dict:
    """
    Compute BLEU-1, ROUGE-L, BERTScore for generated vs reference report.
    Gracefully handles missing optional dependencies.
    """
    result = {}

    # --- BLEU-1 ---
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tokens = [reference.lower().split()]
        gen_tokens = generated.lower().split()
        smoothie = SmoothingFunction().method4
        bleu1 = sentence_bleu(ref_tokens, gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        result["bleu_1"] = round(float(bleu1), 4)
    except ImportError:
        result["bleu_1"] = "nltk not installed"
    except Exception as e:
        result["bleu_1"] = f"Error: {e}"

    # --- ROUGE-L ---
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_l = scorer.score(reference, generated)["rougeL"].fmeasure
        result["rouge_l"] = round(float(rouge_l), 4)
    except ImportError:
        result["rouge_l"] = "rouge-score not installed"
    except Exception as e:
        result["rouge_l"] = f"Error: {e}"

    # --- BERTScore ---
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn([generated], [reference], lang="en", verbose=False)
        result["bertscore"] = round(float(F1.mean()), 4)
    except ImportError:
        result["bertscore"] = "bert-score not installed"
    except Exception as e:
        result["bertscore"] = f"Error: {e}"

    return result
