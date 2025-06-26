from crewai import Task
from agents import communication_analyzer, content_evaluator_agent, emotion_analyzer_agent, manager_agent

def create_tasks(transcript: str, job_title: str):
    tasks = []

    # Communication Analysis Task
    task1 = Task(
        description=f"Analyze the transcript for clarity, fluency, and filler words.\n\nTranscript: {transcript}\nJob Title: {job_title}",
        expected_output="Communication score with detailed breakdown.",
        agent=communication_analyzer
    )
    tasks.append(task1)

    # Content Evaluation Task
    task2 = Task(
        description=f"Evaluate the relevance and depth of responses in the transcript.\n\nTranscript: {transcript}\nJob Title: {job_title}",
        expected_output="Content score with relevant terms identified.",
        agent=content_evaluator_agent
    )
    tasks.append(task2)

    # Emotion Analysis Task
    task3 = Task(
        description=f"Analyze the emotional tone of the transcript.\n\nTranscript: {transcript}\nJob Title: {job_title}",
        expected_output="Emotion score with polarity and subjectivity.",
        agent=emotion_analyzer_agent
    )
    tasks.append(task3)

    # Manager Task
    task4 = Task(
        description="Compile all outputs into a final evaluation report with scores and feedback.",
        expected_output="Comprehensive report with scores and improvement suggestions.",
        agent=manager_agent
    )
    tasks.append(task4)

    return tasks