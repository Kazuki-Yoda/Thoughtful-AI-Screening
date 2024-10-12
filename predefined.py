"""This entire file was solely written by the applicant, Kazuki Yoda."""

import json
from typing import Optional

# # For Debugging only
# from scipy.spatial import distance_matrix
# from sklearn.metrics.pairwise import cosine_similarity

from huggingface_hub import InferenceClient

zero_shot_classification_client = InferenceClient("facebook/bart-large-mnli")


def load_predefined_questions_to_answers_as_dict(path="predefined.json"
) -> dict[str, str]:
    """Load the predefined question-answer pairs as dict of.
    key: question (str), value: answer (str)"""
    
    with open(path) as file:
        data = json.load(file)
    
    if "questions" not in data:
        raise ValueError("`questions` key is expected but missing.")
    
    question_to_answer = dict()
    
    for item in data.get("questions"):
        question = item.get("question")
        answer = item.get("answer")
        
        # Skip if either "question" or "answer" key not found
        if question and answer:
            question_to_answer[question] = answer
    
    return question_to_answer


def get_embeddings(texts: list[str]):
    client = InferenceClient("efederici/sentence-bert-base")

    return [client.feature_extraction(text) for text in texts]


def get_predefined_answer_for_closest_predefined_question(
    question: str,
    cutoff=0.5,  # Minimum classification score to use the predefined answer 
) -> Optional[str]:

    question_to_answer = load_predefined_questions_to_answers_as_dict()
    labels = list(question_to_answer.keys())

    zero_shot_classification_result = zero_shot_classification_client.zero_shot_classification(
        text=question,
        labels=labels,
        multi_label=True,
    )
    max_score_result = max(zero_shot_classification_result,
                           key=lambda x: x.score)

    if max_score_result.score > cutoff:
        closest_predefined_question = max_score_result.label
        return question_to_answer[closest_predefined_question]
    else:
        # Switch back to the normal LLM response
        return None


if __name__ == "__main__":
    """Run some print debugs. Not executed from the Gradio app."""
    
    question_to_answer = load_predefined_questions_to_answers_as_dict()
    print(question_to_answer)
    
    additional_questions = [
        "What does EVA do?",
        "How does PHIL work?",
        "Thoughtful AI",
        ### Irrelevant but confusing questions ###
        "Who is the CEO of Thoughtful AI?",
        "How much does Thoughtful AI pay for its ML engineers?",
        "What's Evangelion (EVA)?"
    ]
    predefined_questions = list(question_to_answer.keys())
    questions = predefined_questions + additional_questions
    
    embeddings = get_embeddings(questions)
    
    for embedding in embeddings:
        print(embedding.shape)
    
    # For DEBUG, check the embeddings
    # print(distance_matrix(embeddings, embeddings[:len(predefined_questions)]))
    # print(cosine_similarity(embeddings, embeddings[:len(predefined_questions)]))

    for question in questions:
        closest_question = get_predefined_answer_for_closest_predefined_question(question)
        print(f"question: {question}")
        print(f"closest_question: {closest_question}")
        print()
