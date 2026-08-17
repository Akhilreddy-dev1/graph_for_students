import streamlit as st
import plotly.express as px
import pandas as pd
import json
import base64
import re
from groq import Groq

# 1. Page Configuration
st.set_page_config(page_title="AI Graph Finder", layout="wide")
st.title("📊 Smart Graph Camera & AI Assistant")
st.write("Snap a photo of any graph to recreate it dynamically and get the code!")

# 2. Sidebar Configuration
st.sidebar.header("🔧 Setup")
st.sidebar.write("Get a 100% free API key at [console.groq.com](https://console.groq.com/)")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Initialize Groq Client
client = Groq(api_key=api_key) if api_key else None

# 3. Layout Configuration
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Step 1: Capture Graph")
    camera_img = st.camera_input("Take a photo of a graph or chart")
    
    if camera_img and not api_key:
        st.warning("Please enter your free Groq API key in the sidebar to process the image.")

with col2:
    st.subheader("🤖 Step 2: AI Rendering & Code")
    
    if camera_img and client:
        with st.spinner("AI is analyzing the graph structure..."):
            try:
                # Base64 encode the binary camera data stream
                image_bytes = camera_img.getvalue()
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                
                # Query Groq via the active multimodal channel
                completion = client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text", 
                                    "text": "Identify the data points in this graph image. Output a single JSON string object containing exactly three keys: 'x' (a list of numbers), 'y' (a list of matching numbers), and 'label' (a string title name). Example structure: {\"x\":, \"y\":, \"label\": \"Growth Curve\"}. Do not output any markdown code blocks, backticks, summaries, or pre-text conversation."
                                },
                                {
                                    "type": "image_url", 
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                }
                            ]
                        }
                    ]
                )
                
                # Isolate raw output string
                raw_output = completion.choices.message.content.strip()
                
                # FAIL-SAFE UNWRAPPING: Extract only the content inside braces {} to bypass conversational noise
                match = re.search(r"(\{.*?\})", raw_output, re.DOTALL)
                if match:
                    raw_output = match.group(1).strip()
                
                # Safely convert extracted clean string into an active dictionary tree
                data = json.loads(raw_output)
                
                # Check data format structures
                if "x" in data and "y" in data:
                    # Construct matching DataFrame arrays
                    df = pd.DataFrame({"X-Axis": data["x"], "Y-Axis": data["y"]})
                    
                    # Graph Generation Layer
                    fig = px.line(df, x="X-Axis", y="Y-Axis", title=data.get("label", "Detected Graph"), markers=True)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Clean Code Assembly Component
                    st.write("### 🐍 Python Code to Replicate:")
                    generated_code = f"""import plotly.express as px
import pandas as pd

# Data points extracted by your AI Assistant
data = {{
    'x': {data['x']},
    'y': {data['y']}
}}

df = pd.DataFrame(data)
fig = px.line(df, x='x', y='y', title='{data.get("label", "My Graph")}', markers=True)
fig.show()"""
                    st.code(generated_code, language="python")
                else:
                    st.error("The AI failed to format the coordinates properly. Please take a clearer photo and try again.")
                
            except json.JSONDecodeError:
                st.error("AI response formatting error. Try snapping the picture again with clearer alignment.")
            except Exception as e:
                st.error(f"System failed to evaluate visualization query. Details: {e}")

# 4. Interactive Chat Assistant Feature at the bottom
st.divider()
st.subheader("💬 Ask the AI Graph Assistant")
user_question = st.chat_input("Ask a question about your data or math equations...")

if user_question:
    if client:
        with st.chat_message("user"):
            st.write(user_question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # FIX: Swapped out decommissioned text model name for active vision model text handler
                chat_completion = client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[{"role": "user", "content": user_question}]
                )
                st.write(chat_completion.choices.message.content)
    else:
        st.info("Provide your free Groq API key in the sidebar to chat with the assistant.")
