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
import logging
import io
import sys
import re
import base64
from datetime import datetime
import tempfile
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Browser Automation", layout="wide")

# Page header
st.title("Browser Automation Tool")
st.markdown("Use this app to automate browser tasks with natural language instructions.")

# API key input (could also be loaded from config)
api_key = st.sidebar.text_input("API Key", value=config.GoogleAPIKey if hasattr(config, 'GoogleAPIKey') else "", type="password")
model = st.sidebar.text_input("Model", value=config.Model if hasattr(config, 'Model') else "gemini-pro")

# Screenshot options
st.sidebar.subheader("Screenshot Options")
take_screenshots = st.sidebar.checkbox("Take Screenshots", value=True)
screenshot_format = st.sidebar.selectbox("Screenshot Format", ["JPEG", "PNG"], index=0)
screenshot_quality = st.sidebar.slider("JPEG Quality", min_value=10, max_value=100, value=85, step=5, 
                                     disabled=screenshot_format!="JPEG")

# Task input
task_input = st.text_area("Enter your automation task", 
                         height=150, 
                         placeholder="Example: Open website youtube.com, search for something, play the first video")

# Use vision checkbox
use_vision = st.checkbox("Use vision", value=True)

# Custom log handler to capture logs
class StringIOHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_output = io.StringIO()
        
    def emit(self, record):
        msg = self.format(record)
        self.log_output.write(msg + '\n')
        
    def get_value(self):
        return self.log_output.getvalue()

# Class to detect step changes in logs and trigger screenshot capture
class StepDetectorHandler(logging.Handler):
    def __init__(self, screenshot_callback):
        super().__init__()
        self.current_step = 0
        self.screenshot_callback = screenshot_callback
        self.step_pattern = re.compile(r'\[agent\] 🚩 Step (\d+)')
        
    def emit(self, record):
        msg = self.format(record)
        match = self.step_pattern.search(msg)
        if match:
            step_num = int(match.group(1))
            if step_num > self.current_step:
                self.current_step = step_num
                if self.screenshot_callback:
                    asyncio.create_task(self.screenshot_callback(step_num))

# Function to parse and extract formatted log steps
def extract_formatted_steps(log_content):
    # Extract step lines from logs
    step_lines = []
    
    lines = log_content.split('\n')
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
            
        # Extract formatted log lines
        if '[agent]' in line or '[controller]' in line:
            step_lines.append(line)
    
    return step_lines

# Function to run the automation task
async def run_automation_task(task, api_key, model, use_vision, take_screenshots=True, screenshot_format="JPEG", screenshot_quality=85):
    os.environ["API_KEY"] = api_key
    
    # Setup logging
    # Remove any existing handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set log level
    root_logger.setLevel(logging.INFO)
    
    # Create and add custom log handler
    string_handler = StringIOHandler()
    formatter = logging.Formatter('%(levelname)-8s [%(name)s] %(message)s')
    string_handler.setFormatter(formatter)
    root_logger.addHandler(string_handler)
    
    # Add console handler for debugging
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create directory for screenshots if it doesn't exist
    screenshots_dir = "screenshots"
    os.makedirs(screenshots_dir, exist_ok=True)
    screenshot_paths = []
    browser_controller = None
    
    # Start time
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Function to take screenshots at each step
    async def take_screenshot_at_step(step_num):
        if take_screenshots and browser_controller and browser_controller.page:
            try:
                # Take screenshot using Playwright page
                screenshot_data = await browser_controller.page.screenshot(
                    type=screenshot_format.lower(),
                    quality=screenshot_quality if screenshot_format.lower() == "jpeg" else None
                )
                
                # Save screenshot to file
                file_ext = "jpg" if screenshot_format.lower() == "jpeg" else "png"
                screenshot_filename = f"{screenshots_dir}/step_{step_num}_{timestamp}.{file_ext}"
                with open(screenshot_filename, "wb") as f:
                    f.write(screenshot_data)
                
                screenshot_paths.append({
                    "path": screenshot_filename,
                    "step": step_num,
                    "data": base64.b64encode(screenshot_data).decode('utf-8')
                })
                return screenshot_filename
            except Exception as e:
                logging.error(f"Failed to take screenshot: {str(e)}")
                return None
        return None
    
    # Add step detector handler if screenshots are enabled
    if take_screenshots:
        step_detector = StepDetectorHandler(take_screenshot_at_step)
        step_detector.setFormatter(formatter)
        root_logger.addHandler(step_detector)
    
    try:
        # Create LLM
        llm = ChatGoogleGenerativeAI(model=model, api_key=SecretStr(api_key))
        
        # Create agent
        agent = Agent(task, llm, use_vision=use_vision)
        
        # Run automation
        history = await agent.run()
        
        # Store the browser controller for screenshots
        if take_screenshots:
            try:
                # Extract browser controller from agent
                browser_controller = agent.controller
                
                # Take a final screenshot
                await take_screenshot_at_step(9999)  # Use a large number for final screenshot
            except Exception as e:
                logging.error(f"Failed to access browser controller: {str(e)}")
        
        # Get final result
        result = None
        success = False
        
        try:
            result = history.final_result()
            success = history.is_successful()
        except AttributeError:
            # In case history doesn't have these methods
            result = "Task completed but couldn't get final result"
            success = False
        
        # Get log output
        log_output = string_handler.get_value()
        
        # Extract formatted steps from logs
        formatted_steps = extract_formatted_steps(log_output)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        return {
            "result": result,
            "success": success,
            "steps": formatted_steps,
            "time": execution_time,
            "history": history,
            "log_output": log_output,
            "screenshots": screenshot_paths
        }
    except Exception as e:
        log_output = string_handler.get_value()
        st.error(f"Error: {str(e)}")
        return {
            "result": f"Error: {str(e)}",
            "success": False,
            "steps": extract_formatted_steps(log_output),
            "time": time.time() - start_time,
            "history": None,
            "log_output": log_output,
            "screenshots": screenshot_paths
        }

# Function to generate HTML report with screenshots
def generate_html_report(task_result):
    # Create HTML for screenshots
    screenshots_html = ""
    if task_result.get("screenshots"):
        screenshots_html = "<div class='screenshots'><h2>Screenshots:</h2>"
        for screenshot in task_result["screenshots"]:
            file_ext = "jpeg" if screenshot["path"].endswith("jpg") else "png"
            screenshots_html += f"""
            <div class="screenshot-container">
                <h3>Step {screenshot["step"]}</h3>
                <img src="data:image/{file_ext};base64,{screenshot['data']}" class="screenshot" />
            </div>
            """
        screenshots_html += "</div>"

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
            .step {{ padding: 5px; border-bottom: 1px solid #ddd; font-family: monospace; white-space: pre-wrap; }}
            .logs {{ margin: 20px 0; background-color: #f8f9fa; padding: 10px; font-family: monospace; white-space: pre-wrap; }}
            .stats {{ margin: 20px 0; background-color: #e6f7ff; padding: 10px; }}
            .screenshots {{ margin: 20px 0; }}
            .screenshot-container {{ margin-bottom: 20px; }}
            .screenshot {{ max-width: 100%; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Browser Automation Report</h1>
        </div>
        
        <div class="result">
            <h2>Result:</h2>
            <p>{html.escape(str(task_result['result']))}</p>
            <p class="{'success' if task_result['success'] else 'failure'}">
                Status: {"Success" if task_result['success'] else "Failed"}
            </p>
        </div>
        
        {screenshots_html}
        
        <div class="steps">
            <h2>Execution Steps:</h2>
            {"".join(f'<div class="step">{html.escape(step)}</div>' for step in task_result['steps'])}
        </div>
        
        <div class="logs">
            <h2>Full Log:</h2>
            <pre>{html.escape(task_result['log_output'])}</pre>
        </div>
        
        <div class="stats">
            <h2>Statistics:</h2>
            <p>Execution Time: {task_result['time']:.2f} seconds</p>
            <p>Total Steps: {len(task_result['steps'])}</p>
            <p>Screenshots Taken: {len(task_result.get('screenshots', []))}</p>
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
        # Create a progress bar placeholder
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create a spinner while task is running
        with st.spinner("Running automation task..."):
            status_text.text("Starting automation...")
            progress_bar.progress(10)
            
            # Run the task
            task_result = asyncio.run(run_automation_task(
                task_input, 
                api_key, 
                model, 
                use_vision,
                take_screenshots,
                screenshot_format,
                screenshot_quality
            ))
            
            status_text.text("Processing results...")
            progress_bar.progress(100)
        
        # Display tabs for different output formats
        tab1, tab2, tab3, tab4 = st.tabs(["Logs", "Steps", "Screenshots", "HTML Report"])
        
        with tab1:
            st.subheader("Task Result")
            if task_result["success"]:
                st.success(task_result["result"])
            else:
                st.error(task_result["result"])
            
            # Display full log output
            st.subheader("Full Log Output")
            st.code(task_result["log_output"], language="")
            
            # Display stats
            st.subheader("Statistics")
            st.info(f"Execution Time: {task_result['time']:.2f} seconds")
            
        with tab2:
            # Display steps
            st.subheader("Execution Steps")
            for step in task_result["steps"]:
                st.text(step)
                
            st.info(f"Total Steps: {len(task_result['steps'])}")
        
        with tab3:
            # Display screenshots
            st.subheader("Screenshots")
            if task_result.get("screenshots"):
                for screenshot in task_result["screenshots"]:
                    st.subheader(f"Step {screenshot['step']}")
                    
                    # Display image from base64 data
                    file_ext = "jpeg" if screenshot["path"].endswith("jpg") else "png"
                    st.image(f"data:image/{file_ext};base64,{screenshot['data']}")
                    
                    # Add download button for each screenshot
                    with open(screenshot["path"], "rb") as file:
                        st.download_button(
                            label=f"Download Screenshot {screenshot['step']}",
                            data=file,
                            file_name=os.path.basename(screenshot["path"]),
                            mime=f"image/{file_ext}"
                        )
            else:
                st.info("No screenshots captured during this run")
            
        with tab4:
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