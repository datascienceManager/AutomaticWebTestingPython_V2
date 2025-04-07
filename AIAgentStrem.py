import streamlit as st
import asyncio
from browser_use.agent.service import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import os
import config
import time
import html

# Set page configuration
st.set_page_config(page_title="Browser Automation", layout="wide")

# Page header
st.title("Browser Automation Tool")
st.markdown("Use this app to automate browser tasks with natural language instructions.")

# API key input (could also be loaded from config)
api_key = st.sidebar.text_input("API Key", value=config.GoogleAPIKey if hasattr(config, 'GoogleAPIKey') else "", type="password")
model = st.sidebar.text_input("Model", value=config.Model if hasattr(config, 'Model') else "gemini-pro")

# Task input
task_input = st.text_area("Enter your automation task", 
                         height=150, 
                         placeholder="Example: Open website youtube.com, search for Ollama, play the first video for 2 minutes and close the browser")

# Use vision checkbox
use_vision = st.checkbox("Use vision", value=True)

# Function to run the automation task
async def run_automation_task(task, api_key, model, use_vision):
    os.environ["API_KEY"] = api_key
    
    # Create LLM
    llm = ChatGoogleGenerativeAI(model=model, api_key=SecretStr(api_key))
    
    # Create agent
    agent = Agent(task, llm, use_vision=use_vision)
    
    # Run the task with progress
    st.write("Running automation...")
    progress_bar = st.progress(0)
    
    # Create a placeholder for logs
    log_container = st.empty()
    logs = []
    
    # Start time
    start_time = time.time()
    
    try:
        # Run automation and capture steps
        history = await agent.run(
            callback=lambda step: logs.append(f"Step {step.number}: {step.action}")
        )
        
        # Format the logs and update progress
        for i, log in enumerate(logs):
            log_container.markdown("\n".join(logs))
            progress_bar.progress((i + 1) / len(logs))
            time.sleep(0.1)  # Small delay for visual effect
        
        # Get final result
        result = history.final_result()
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Return results and stats
        return {
            "result": result,
            "success": history.is_successful(),
            "steps": logs,
            "time": execution_time,
            "history": history
        }
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return {
            "result": f"Error: {str(e)}",
            "success": False,
            "steps": logs,
            "time": time.time() - start_time,
            "history": None
        }

# Function to generate HTML report
def generate_html_report(task_result):
    # Create HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Browser Automation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #4CAF50; color: white; padding: 10px; }}
            .result {{ margin: 20px 0; padding: 15px; background-color: #f2f2f2; }}
            .success {{ color: green; }}
            .failure {{ color: red; }}
            .steps {{ margin: 20px 0; }}
            .step {{ padding: 5px; border-bottom: 1px solid #ddd; }}
            .stats {{ margin: 20px 0; background-color: #e6f7ff; padding: 10px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Browser Automation Report</h1>
        </div>
        
        <div class="result">
            <h2>Result:</h2>
            <p>{html.escape(task_result['result'])}</p>
            <p class="{'success' if task_result['success'] else 'failure'}">
                Status: {"Success" if task_result['success'] else "Failed"}
            </p>
        </div>
        
        <div class="steps">
            <h2>Steps:</h2>
            {"".join(f'<div class="step">{html.escape(step)}</div>' for step in task_result['steps'])}
        </div>
        
        <div class="stats">
            <h2>Statistics:</h2>
            <p>Execution Time: {task_result['time']:.2f} seconds</p>
            <p>Total Steps: {len(task_result['steps'])}</p>
        </div>
    </body>
    </html>
    """
    return html_report

# Run button
if st.button("Run Automation"):
    if not api_key or not task_input:
        st.error("Please provide both API key and task description")
    else:
        # Create a spinner while task is running
        with st.spinner("Running automation task..."):
            # Run the task
            task_result = asyncio.run(run_automation_task(task_input, api_key, model, use_vision))
        
        # Display tabs for different output formats
        tab1, tab2 = st.tabs(["Results", "HTML Report"])
        
        with tab1:
            st.subheader("Task Result")
            if task_result["success"]:
                st.success(task_result["result"])
            else:
                st.error(task_result["result"])
            
            # Display steps
            st.subheader("Steps")
            for step in task_result["steps"]:
                st.text(step)
            
            # Display stats
            st.subheader("Statistics")
            st.info(f"Execution Time: {task_result['time']:.2f} seconds")
            st.info(f"Total Steps: {len(task_result['steps'])}")
            
        with tab2:
            # Generate and display HTML report
            html_report = generate_html_report(task_result)
            st.download_button(
                label="Download HTML Report",
                data=html_report,
                file_name="automation_report.html",
                mime="text/html"
            )
            st.subheader("HTML Preview")
            st.components.v1.html(html_report, height=600)

# Footer
st.markdown("---")
st.markdown("Browser Automation Tool powered by browser_use agent")