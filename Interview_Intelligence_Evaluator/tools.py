from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import spacy
from textblob import TextBlob
import numpy as np

# Input Schema
class TranscriptInput(BaseModel):
    transcript: str = Field(..., description="Interview transcript")
    job_title: str = Field(..., description="Job title for context")

# Communication Scoring Tool
class CommunicationScorer(BaseTool):
    name: str = "CommunicationAnalyzer"
    description: str = "Scores communication based on clarity, fluency, and filler words."
    args_schema: Type[BaseModel] = TranscriptInput

    def _run(self, transcript: str, job_title: str) -> str:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(transcript.lower())
        
        # Filler word detection
        filler_words = {"um", "uh", "like", "you know", "so"}
        filler_count = sum(1 for token in doc if token.text in filler_words)
        total_words = len(doc)
        filler_score = max(0, 100 - (filler_count / total_words * 100) * 5)  # Penalize fillers
        
        # Sentence complexity (average sentence length)
        sentences = list(doc.sents)
        avg_sentence_length = total_words / len(sentences) if sentences else 0
        complexity_score = min(100, avg_sentence_length * 4)  # Ideal length ~25 words
        
        # Combine scores
        final_score = (filler_score * 0.6 + complexity_score * 0.4)
        return f"Communication Score: {final_score:.0f}/100\n- Filler Words: {filler_count} ({filler_score:.0f}/100)\n- Clarity (Sentence Length): {complexity_score:.0f}/100"

# Content Evaluation Tool
class ContentEvaluator(BaseTool):
    name: str = "ContentEvaluator"
    description: str = "Evaluates relevance and depth of responses."
    args_schema: Type[BaseModel] = TranscriptInput

    def _run(self, transcript: str, job_title: str) -> str:
        # Simulated keyword matching for job title
        job_keywords = {
            "ai engineer": {"python", "machine learning", "deep learning", "nlp", "model"},
            "data scientist": {"statistics", "python", "sql", "data analysis", "modeling"}
        }
        keywords = job_keywords.get(job_title.lower(), set())
        
        doc = spacy.load("en_core_web_sm")(transcript.lower())
        found_keywords = [token.text for token in doc if token.text in keywords]
        relevance_score = min(100, len(found_keywords) / len(keywords) * 100) if keywords else 50
        
        return f"Content Score: {relevance_score:.0f}/100\n- Relevant Terms: {', '.join(found_keywords)}"

# Emotion Analysis Tool
class EmotionAnalyzer(BaseTool):
    name: str = "EmotionAnalyzer"
    description: str = "Analyzes emotional tone in transcript."
    args_schema: Type[BaseModel] = TranscriptInput

    def _run(self, transcript: str, job_title: str) -> str:
        blob = TextBlob(transcript)
        sentiment = blob.sentiment
        polarity = sentiment.polarity  # -1 (negative) to 1 (positive)
        subjectivity = sentiment.subjectivity  # 0 (objective) to 1 (subjective)
        
        # Map sentiment to confidence score
        confidence_score = max(0, min(100, (polarity + 1) * 50))  # Normalize to 0-100
        return f"Emotion Score: {confidence_score:.0f}/100\n- Polarity: {polarity:.2f}\n- Subjectivity: {subjectivity:.2f}"