"""
Runner script for GlycoTwin Streamlit Dashboard
Starts the clinical portal directly in headless mode.
"""
import os
import sys

# Ensure headless settings
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_PORT"] = "8501"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "localhost"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

from streamlit.web import bootstrap

def main():
    script_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
    flag_options = {
        "server.port": 8501,
        "server.headless": True,
        "server.address": "localhost",
        "browser.gatherUsageStats": False,
    }
    print("Launching GlycoTwin Clinical Dashboard on http://localhost:8501 ...")
    bootstrap.run(script_path, False, [], flag_options)

if __name__ == "__main__":
    main()
