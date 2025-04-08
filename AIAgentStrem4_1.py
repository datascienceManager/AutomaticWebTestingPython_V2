import streamlit as st
import asyncio
from browser_use.agent.service import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from datetime import datetime

import asyncio
import os
import time
from datetime import datetime
import pyautogui
import cv2
import numpy as np  # Added numpy import which was missing
import imageio
from pathlib import Path

import pdb
import logging

from dotenv import load_dotenv

load_dotenv()
import os
import glob
import asyncio
import argparse
import os

logger = logging.getLogger(__name__)

# import gradio as gr

from browser_use.agent.service import Agent
from playwright.async_api import async_playwright
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContextConfig,
    BrowserContextWindowSize,
)
from langchain_ollama import ChatOllama
from playwright.async_api import async_playwright
from src.utils.agent_state import AgentState

from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.browser.custom_context import BrowserContextConfig, CustomBrowserContext
from src.controller.custom_controller import CustomController
# from gradio.themes import Citrus, Default, Glass, Monochrome, Ocean, Origin, Soft, Base
from src.utils.default_config_settings import default_config, load_config_from_file, save_config_to_file, save_current_config, update_ui_from_config
from src.utils.utils import update_model_dropdown, get_latest_files, capture_screenshot

import os
import config
import time
import html
import logging
import io
import sys

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
                         placeholder="Example: Open website youtube.com, search for something, play the first video")

# Use vision checkbox
use_vision = st.checkbox("Use vision", value=True)

# Setup logging to capture output
def setup_logging():
    # Create a string buffer to capture log output
    log_capture_string = io.StringIO()
    
    # Configure root logger
    root_logger = logging.getLogger()
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set logging level
    root_logger.setLevel(logging.INFO)
    
    # Add handler that writes to the string buffer
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.INFO)
    
    # Create a formatter
    formatter = logging.Formatter('%(levelname)-8s [%(name)s] %(message)s')
    ch.setFormatter(formatter)
    
    # Add handler to the root logger
    root_logger.addHandler(ch)
    
    # Add a stream handler to also print to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return log_capture_string


# Dummy fallback if steps are not available
def extract_formatted_steps(steps_text: str) -> list:
    steps = []
    for i, line in enumerate(steps_text.split("\n")):
        line = line.strip()
        if line:
            steps.append({"step": f"Step {i + 1}: {line}"})
    return steps or [{"step": "No steps extracted"}]


# Screen recorder to capture and create GIF
def record_screen(output_path: str, duration: int = 10, fps: int = 5):
    st.write(f"Starting screen recording to {output_path}") 
    frames = []
    interval = 1 / fps
    start_time = time.time()

    while (time.time() - start_time) < duration:
        screenshot = pyautogui.screenshot()
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        frames.append(frame)
        time.sleep(interval)

    # Save video
    video_path = output_path + ".mp4"
    gif_path = output_path + ".gif"
    st.write(f"Saving video to {video_path}") 
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

    # Save gif
    imageio.mimsave(gif_path, [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=fps)
    st.write(f"Saved GIF to {gif_path}") 
    return video_path, gif_path

# Function to run the automation task
async def run_automation_task(task_input, api_key, model, use_vision):
    log_capture_string = setup_logging()
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = f"automation_record_{timestamp}"
        Path(base_path).mkdir(parents=True, exist_ok=True)

        screen_video_path, screen_gif_path = record_screen(os.path.join(base_path, "recording"), duration=10, fps=5)

        # [Insert actual automation logic here, call your custom_browser etc.]
        await asyncio.sleep(2)  # Simulated automation task

        return {
            "result": "Automation task completed successfully.",
            "success": True,
            "steps": extract_formatted_steps("Step A\nStep B\nStep C"),
            "video": screen_video_path,
            "gif": screen_gif_path,
            "time": time.time(),
            "history": None,
            "start_time": time.time(),
            "log_output": log_capture_string.getvalue()
        }
    except Exception as e:
        return {
            "result": f"Error: {str(e)}",
            "success": False,
            "steps": extract_formatted_steps(""),
            "time": time.time(),
            "history": None,
            "start_time": time.time(),
            "log_output": log_capture_string.getvalue() if 'log_capture_string' in locals() else ""
        }


# Function to generate HTML report
def generate_html_report(task_result):
    # Extract step text from step dictionaries
    step_html = ""
    for step_dict in task_result['steps']:
        step_text = step_dict.get('step', 'Unknown step')
        step_html += f'<div class="step">{html.escape(step_text)}</div>'
    
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
            .logs {{ margin: 20px 0; background-color: #f8f9fa; padding: 10px; font-family: monospace; white-space: pre-wrap; }}
            .stats {{ margin: 20px 0; background-color: #e6f7ff; padding: 10px; }}
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
        
        <div class="logs">
            <h2>Detailed Log:</h2>
            <pre>{html.escape(task_result['log_output'])}</pre>
        </div>
        
        <div class="steps">
            <h2>Steps Summary:</h2>
            {step_html}
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
        # Create a progress bar placeholder
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create a spinner while task is running
        with st.spinner("Running automation task..."):
            status_text.text("Starting automation...")
            progress_bar.progress(10)
            
            # Run the task
            task_result = asyncio.run(run_automation_task(task_input, api_key, model, use_vision))
            
            status_text.text("Processing results...")
            progress_bar.progress(100)
        
        # Display tabs for different output formats
        tab1, tab2, tab3 = st.tabs(["Logs", "Steps", "HTML Report"])
        
        with tab1:
            st.subheader("Task Result")
            if task_result["success"]:
                st.success(task_result["result"])
            else:
                st.error(task_result["result"])

            # Display videos if available
            if "video" in task_result and os.path.exists(task_result["video"]):
                st.subheader("Screen Recording")
                st.video(task_result["video"])
            
            if "gif" in task_result and os.path.exists(task_result["gif"]):
                st.subheader("Screen Recording GIF")
                st.image(task_result["gif"])
            
            # Display detailed logs
            st.subheader("Detailed Logs")
            st.code(task_result["log_output"], language="")
            
            # Display stats
            st.subheader("Statistics")
            st.info(f"Execution Time: {task_result['time']:.2f} seconds")
            
        with tab2:
            # Display steps
            st.subheader("Execution Steps")
            for step_dict in task_result["steps"]:
                step_text = step_dict.get('step', 'Unknown step')
                st.text(step_text)
                
            st.info(f"Total Steps: {len(task_result['steps'])}")
            
        with tab3:
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

