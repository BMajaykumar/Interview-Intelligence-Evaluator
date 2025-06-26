from crewai import Agent, LLM
from tools import CommunicationScorer, ContentEvaluator, EmotionAnalyzer

# LLM Setup (Replace with your API key in production)
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
    api_key="AIzaSyBIUOFIcJgObfWmNybDifPtxZ9x1rw2SGc"  # Use Streamlit secrets
)

# Tool Instances
communication_scorer = CommunicationScorer()
content_evaluator = ContentEvaluator()
emotion_analyzer = EmotionAnalyzer()

# Agents
communication_analyzer = Agent(
    role="Communication Analyzer",
    goal="Evaluate clarity, fluency, and pacing of responses",
    backstory="Linguist specialized in spoken communication analysis.",
    tools=[communication_scorer],
    llm=llm
)

content_evaluator_agent = Agent(
    role="Content Evaluator",
    goal="Assess relevance and depth of interview responses",
    backstory="HR expert with deep knowledge of technical interviews.",
    tools=[content_evaluator],
    llm=llm
)

emotion_analyzer_agent = Agent(
    role="Emotion Analyzer",
    goal="Detect emotional tone and confidence in responses",
    backstory="Psychologist skilled in sentiment and tone analysis.",
    tools=[emotion_analyzer],
    llm=llm
)

manager_agent = Agent(
    role="Evaluation Manager",
    goal="Coordinate agents and compile a final evaluation report",
    backstory="Senior HR manager overseeing candidate evaluations.",
    llm=llm,
    allow_delegation=True
)