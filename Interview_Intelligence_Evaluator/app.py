import streamlit as st
from crewai import Crew, Process
from agents import communication_analyzer, content_evaluator_agent, emotion_analyzer_agent, manager_agent
from tasks import create_tasks

st.set_page_config(page_title="Interview Intelligence Evaluator", layout="wide")
st.title("🧠 Interview Intelligence Evaluator")

# Input Fields
job_title = st.text_input("🔤 Enter Job Title (e.g., AI Engineer)")
transcript_file = st.file_uploader("📝 Upload Interview Transcript (.txt)", type=["txt"])
transcript_text = st.text_area("📝 Or Paste Interview Transcript", height=200)

run_button = st.button("🚀 Evaluate Interview")

# Process Inputs
if run_button:
    # Read transcript from file or text area
    transcript = ""
    if transcript_file:
        transcript = transcript_file.read().decode("utf-8")
    elif transcript_text:
        transcript = transcript_text
    else:
        st.warning("Please upload a transcript file or paste a transcript.")
        st.stop()

    if not job_title:
        st.warning("Please provide a job title.")
        st.stop()

    with st.spinner("🛠️ Agents are evaluating the interview..."):
        # Create tasks
        tasks = create_tasks(transcript, job_title)

        # Initialize Crew
        crew = Crew(
            agents=[communication_analyzer, content_evaluator_agent, emotion_analyzer_agent],
            tasks=tasks,
            manager_agent=manager_agent,
            process=Process.hierarchical,
            verbose=True
        )

        # Run Workflow
        result = crew.kickoff()

    st.success("✅ Evaluation Complete!")
    st.subheader("📊 Interview Evaluation Report")
    st.text_area("🔍 Output", result, height=500)