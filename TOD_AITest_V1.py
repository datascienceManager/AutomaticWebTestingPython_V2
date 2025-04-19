import streamlit as st
import asyncio
import cv2
import numpy as np  # Moved numpy import here
import pyautogui
import imageio
from datetime import datetime
import time
from pathlib import Path
from browser_use.agent.service import Agent  # Assuming this is the correct import path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI # Imported but not used - consider adding logic or removing
from pydantic import SecretStr
import os
# import config # Using config, ensure this file exists or handle absence
try:
    import config
except ImportError:
    st.warning("config.py not found. API key and Model must be entered manually.")
    class config: # Create a dummy config class if file is missing
        GoogleAPIKey = ""
        Model = "gemini-pro"

import time
import html
import logging
import io
import sys
import threading # Added for potential concurrent recording

# Set page configuration
st.set_page_config(page_title="Browser Automation", layout="wide")

# Page header
st.title("Browser Automation Tool")
st.markdown("Use this app to automate browser tasks with natural language instructions.")

# API key input (could also be loaded from config)
# Use session state to preserve API key across reruns if needed

# ******************** Commented out for now , if you want side bar with information ********************

# if 'api_key' not in st.session_state:
#     st.session_state.api_key = config.GoogleAPIKey if hasattr(config, 'GoogleAPIKey') else ""

# api_key = st.sidebar.text_input("API Key", value=st.session_state.api_key, type="password")
# st.session_state.api_key = api_key # Store changes back to session state

# model = st.sidebar.text_input("Model", value=config.Model if hasattr(config, 'Model') else "gemini-pro")

# *******************  Incorporated to hide the sidebar *******************

api_key = config.GoogleAPIKey if hasattr(config, 'GoogleAPIKey') else ""
model = config.Model if hasattr(config, 'Model') else "gemini-pro"


# Task input
task_input = st.text_area("Enter your automation task",
                         height=150,
                         placeholder="Example: Open google.com, search for 'best Streamlit apps', click the first result.")

# Use vision checkbox
use_vision = st.checkbox("Use vision", value=True)

# === Video Recording & GIF Generation Code Start ===

def start_screen_recording(output_path, duration_seconds=10, fps=10, stop_event=None):
    """
    Records the screen for a specified duration or until stop_event is set.

    Args:
        output_path (str or Path): Base path for saving files (without extension).
        duration_seconds (int): Maximum recording duration if stop_event is None.
        fps (int): Frames per second for the recording.
        stop_event (threading.Event, optional): Event to signal stopping the recording externally.
                                                If provided, duration_seconds is ignored.

    Returns:
        tuple: (video_path, gif_path) or (None, None) if recording failed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    video_path = str(output_path.with_suffix(".mp4"))
    gif_path = str(output_path.with_suffix(".gif"))

    st.write(f"Starting screen recording...") # Use st.info or st.status for less prominent messages
    frames = []
    interval = 1.0 / fps
    start_time = time.time()
    screen_size = pyautogui.size() # Get screen size

    try:
        # Determine screen size for the writer
        s_width, s_height = screen_size
        # Initialize VideoWriter - ensure correct dimensions
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (s_width, s_height))

        while True:
            current_time = time.time()
            # Stop condition check
            if stop_event and stop_event.is_set():
                st.write("Recording stopped by signal.")
                break
            if not stop_event and (current_time - start_time) > duration_seconds:
                st.write(f"Recording finished after {duration_seconds} seconds.")
                break

            screenshot = pyautogui.screenshot()
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            # Resize frame ONLY if screenshot size mismatches VideoWriter size (unlikely with pyautogui.screenshot)
            f_height, f_width, _ = frame.shape
            if f_width != s_width or f_height != s_height:
                 st.warning(f"Frame size ({f_width}x{f_height}) differs from screen size ({s_width}x{s_height}). Check screen scaling.")
                 # You might need to resize here if issues occur, but usually not needed.
                 # frame = cv2.resize(frame, (s_width, s_height)) # Example resize

            out.write(frame) # Write frame to video
            frames.append(frame) # Keep frame for GIF
            time.sleep(max(0, interval - (time.time() - current_time))) # Adjust sleep for processing time

        st.write(f"Saving video to {video_path}")
        out.release() # Release the video writer

        if frames: # Only save GIF if frames were captured
            st.write(f"Saving GIF to {gif_path}")
            # Convert frames BGR to RGB for imageio
            rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
            imageio.mimsave(gif_path, rgb_frames, fps=fps)
            st.write(f"Saved GIF to {gif_path}")
            return video_path, gif_path
        else:
            st.warning("No frames captured for GIF.")
            # Clean up empty video file if needed
            if Path(video_path).exists():
                 Path(video_path).unlink()
            return None, None

    except Exception as e:
        st.error(f"Screen recording failed: {e}")
        if 'out' in locals() and out.isOpened():
            out.release() # Ensure writer is released on error
        # Clean up partial files if needed
        if Path(video_path).exists():
            Path(video_path).unlink()
        if Path(gif_path).exists():
            Path(gif_path).unlink()
        return None, None

# === Video Recording & GIF Generation Code End ===

# Setup logging to capture output
def setup_logging():
    log_capture_string = io.StringIO()
    # Consider using a specific logger instead of root
    # logger = logging.getLogger("BrowserAutomationApp")
    logger = logging.getLogger() # Using root for now as per original code

    # Remove existing handlers (be cautious with this)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)-8s [%(name)s] %(message)s') # Added timestamp

    # Handler for capturing logs to string
    string_handler = logging.StreamHandler(log_capture_string)
    string_handler.setFormatter(formatter)
    logger.addHandler(string_handler)

    # Handler for printing logs to console (Streamlit terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return log_capture_string, logger # Return logger if needed elsewhere

# Function to run the automation task
async def run_automation_task(task, api_key, model, use_vision):
    # Ensure API key is set for the agent's environment if needed
    # os.environ["GOOGLE_API_KEY"] = api_key # Or specific key name agent expects

    log_capture_string, logger = setup_logging()
    start_time = time.time()
    logger.info("Starting automation task...")
    logger.info(f"Task: {task}")
    # logger.info(f"Model: {model}, Vision Enabled: {use_vision}")

    try:
        # Create LLM
        # TODO: Add logic to choose between Google/OpenAI based on UI selection
        llm = ChatGoogleGenerativeAI(model=model, google_api_key=SecretStr(api_key))
        # Or: llm = ChatOpenAI(openai_api_key=SecretStr(api_key), model="gpt-4-vision-preview")

        # Create agent
        agent = Agent(task, llm, use_vision=use_vision)
        logger.info("Agent initialized.")

        # Run automation
        logger.info("Running agent...")
        history = await agent.run()
        logger.info("Agent run completed.")

        # Get final result
        result = history.final_result() if hasattr(history, 'final_result') else "No final result method found."
        is_success = history.is_successful() if hasattr(history, 'is_successful') else False
        logger.info(f"Task Success Status: {is_success}")
        logger.info(f"Final Result: {result}")


        execution_time = time.time() - start_time
        logger.info(f"Total execution time: {execution_time:.2f} seconds")

        # Extract log output AFTER execution finishes
        log_output = log_capture_string.getvalue()

        # Extract steps from history
        steps_extracted = []
        try:
            # Attempt to access steps attribute directly (adjust attribute name if needed)
            if hasattr(history, 'steps') and history.steps:
                 for i, step_obj in enumerate(history.steps):
                     # Try to get a meaningful string representation of the step
                     step_str = str(step_obj.action) if hasattr(step_obj, 'action') else str(step_obj)
                     steps_extracted.append(f"Step {i+1}: {step_str}")
                 logger.info(f"Extracted {len(steps_extracted)} steps from history.steps")
            # Fallback: Iterate through history if it's iterable and doesn't have 'steps'
            elif isinstance(history, (list, tuple)):
                 for i, item in enumerate(history):
                      steps_extracted.append(f"Step {i+1}: {str(item)}")
                 logger.info(f"Extracted {len(steps_extracted)} items by iterating over history")
            else:
                 logger.warning("Could not extract detailed steps from history object.")
                 steps_extracted = ["Detailed step extraction not supported for this history type."]
        except Exception as step_ex:
            logger.error(f"Error extracting steps from history: {step_ex}")
            steps_extracted = ["Error occurred during step extraction."]


        return {
            "result": result,
            "success": is_success,
            "steps": steps_extracted,
            "time": execution_time,
            # "history": history, # History object might be large, maybe exclude from direct return
            "log_output": log_output
        }
    except Exception as e:
        logger.error(f"An error occurred during automation: {e}", exc_info=True) # Log traceback
        execution_time = time.time() - start_time
        log_output = log_capture_string.getvalue() # Get logs captured before error
        return {
            "result": f"Error: {str(e)}",
            "success": False,
            "steps": [],
            "time": execution_time,
            # "history": None,
            "log_output": log_output
        }
    finally:
        # Clean up logging handlers if necessary (depends on approach)
        pass


# Function to generate HTML report
def generate_html_report(task_result):
    status_class = "success" if task_result.get('success', False) else "failure"
    status_text = "Success" if task_result.get('success', False) else "Failed"
    steps_html = "".join(f'<div class="step">{html.escape(step)}</div>' for step in task_result.get('steps', []))
    log_output_html = f"<pre>{html.escape(task_result.get('log_output', 'No logs available.'))}</pre>"

    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Browser Automation Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; line-height: 1.6; }}
            .header {{ background-color: #007bff; color: white; padding: 15px 20px; text-align: center; border-radius: 5px; }}
            h1, h2 {{ color: #333; }}
            .section {{ margin: 25px 0; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .result {{ background-color: #e9ecef; }}
            .success {{ color: #28a745; font-weight: bold; }}
            .failure {{ color: #dc3545; font-weight: bold; }}
            .steps {{ background-color: #f8f9fa; }}
            .step {{ padding: 8px 0; border-bottom: 1px solid #eee; }}
            .step:last-child {{ border-bottom: none; }}
            .logs {{ background-color: #343a40; color: #f8f9fa; padding: 15px; font-family: 'Courier New', Courier, monospace; white-space: pre-wrap; word-wrap: break-word; border-radius: 5px; font-size: 0.9em; }}
            .stats {{ background-color: #e2f0ff; }}
            pre {{ margin: 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Browser Automation Report</h1>
        </div>

        <div class="section result">
            <h2>Result Summary</h2>
            <p class="{status_class}">Status: {status_text}</p>
            <p><strong>Details:</strong> {html.escape(str(task_result.get('result', 'No result message.')))}</p>
        </div>

        <div class="section stats">
            <h2>Statistics</h2>
            <p>Execution Time: {task_result.get('time', 0):.2f} seconds</p>
            <p>Total Steps Extracted: {len(task_result.get('steps', []))}</p>
        </div>

        <div class="section steps">
            <h2>Steps Summary</h2>
            {steps_html if task_result.get('steps', []) else "<p>No steps were extracted.</p>"}
        </div>

        <div class="section">
             <h2>Detailed Log</h2>
            <div class="logs">
                {log_output_html}
            </div>
        </div>

    </body>
    </html>
    """
    return html_report


# --- Main App Logic ---
if st.button("Run Automation"):
    if not api_key:
        st.error("❌ Please provide the API key in the sidebar.")
    elif not task_input:
        st.error("❌ Please enter the automation task description.")
    else:
        # # --- Optional: Add concurrent screen recording ---
        # recording_stop_event = threading.Event()
        # output_filename = f"automation_recording_{time.strftime('%Y%m%d_%H%M%S')}"
        # recording_thread = threading.Thread(
        #     target=start_screen_recording,
        #     args=(output_filename, None, 10, recording_stop_event), # Pass None for duration to use stop_event
        #     daemon=True # Allows main thread to exit even if this thread is running
        # )
        # recording_thread.start()
        # st.info("📹 Screen recording started...")
        # ------------------------------------------------

        # Progress indication
        status_text = st.empty()
        status_text.info("🚀 Starting automation task...")
        progress_bar = st.progress(0)

        task_result = None # Initialize task_result

        try:
            # Run the async task
            task_result = asyncio.run(run_automation_task(task_input, api_key, model, use_vision))
            progress_bar.progress(100) # Mark as complete
            if task_result and task_result.get("success"):
                 status_text.success("✅ Automation completed successfully!")
            else:
                 status_text.error("⚠️ Automation finished with errors or failed.")

        except Exception as e:
            # Catch errors during asyncio.run or if run_automation_task fails unexpectedly
            st.error(f"🚨 Critical error during task execution: {e}")
            status_text.error("🚨 Critical error occurred.")
            task_result = None # Ensure result is None or an error dict
        finally:
            # # --- Optional: Stop concurrent screen recording ---
            # if 'recording_stop_event' in locals():
            #     recording_stop_event.set()
            #     st.info("⏹️ Stopping screen recording...")
            #     if 'recording_thread' in locals():
            #         recording_thread.join(timeout=10) # Wait for thread to finish (with timeout)
            #         if recording_thread.is_alive():
            #              st.warning("Recording thread did not stop gracefully.")
            #     st.info("📹 Recording stopped.")
            # -------------------------------------------------
            progress_bar.progress(100) # Ensure progress bar is full


        # Display results only if task_result is available
        if task_result:
            st.success("Processing results complete.") # Change status text

            tab1, tab2, tab3 = st.tabs(["📊 Results & Logs", "🪜 Steps", "📄 HTML Report"])

            with tab1:
                st.subheader("Task Result")
                if task_result.get("success", False):
                    st.success(f"Status: Success\n\nDetails: {task_result.get('result', 'N/A')}")
                else:
                    st.error(f"Status: Failed\n\nDetails: {task_result.get('result', 'N/A')}")

                st.subheader("Statistics")
                st.info(f"Execution Time: {task_result.get('time', 0):.2f} seconds")

                st.subheader("Detailed Logs")
                st.code(task_result.get("log_output", "No logs captured."), language="log")


            with tab2:
                st.subheader("Execution Steps")
                if task_result.get("steps"):
                    for step in task_result.get("steps", []):
                        st.text(step)
                    st.info(f"Total Steps Extracted: {len(task_result.get('steps', []))}")
                else:
                    st.warning("No steps were extracted or available.")


            with tab3:
                st.subheader("HTML Report")
                try:
                    html_report = generate_html_report(task_result)
                    st.download_button(
                        label="📥 Download HTML Report",
                        data=html_report,
                        file_name=f"automation_report_{time.strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                    )
                    st.markdown("### Report Preview")
                    st.components.v1.html(html_report, height=600, scrolling=True)
                except Exception as html_err:
                    st.error(f"Could not generate or display HTML report: {html_err}")

            # --- Optional: Display Recording ---
            # You would need the paths returned from start_screen_recording
            # Example:
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # # base_path = f"recording{timestamp}"
            # base_path = f"recording{timestamp}"
            # Path(base_path).mkdir(parents=True, exist_ok=True)

            # os.path.join(base_path, "recording")

            # output_filename="recording"



            # video_file_path = f"{output_filename}.mp4"
            # gif_file_path = f"{output_filename}.gif"
            # # video_file_path = "path_to_your_video.mp4" # Replace with actual path
            # # gif_file_path = "path_to_your_gif.gif"     # Replace with actual path
            # # 
            # if Path(video_file_path).exists():
            #     st.subheader("Screen Recording (Video)")
            #     try:
            #         video_file = open(video_file_path, 'rb')
            #         video_bytes = video_file.read()
            #         st.video(video_bytes)
            #         video_file.close()
            #     except Exception as vid_err:
            #         st.error(f"Error displaying video: {vid_err}")
            # if Path(gif_file_path).exists():
            #     st.subheader("Screen Recording (GIF)")
            #     st.image(gif_file_path)
            # ----------------------------------

        else:
             st.error("Task execution failed to produce results.")


# Footer
st.markdown("---")
st.markdown("Browser Automation Tool powered by AI agent")