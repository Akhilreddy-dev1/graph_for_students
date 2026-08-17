import streamlit as st
import plotly.express as px
import pandas as pd
import json
import base64
from groq import Groq

# 1. Page Configuration
st.set_page_config(page_title="AI Graph Finder", layout="wide")
st.title("📊 Smart Graph Camera & AI Assistant")
st.write("Snap a photo of any graph to recreate it dynamically and get the code!")

# 2. Sidebar for Free API Configuration
st.sidebar.header("🔧 Setup Setup")
st.sidebar.write("Get a 100% free API key at [://groq.com](https://://groq.com/)")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Initialize Groq Client if key is provided
client = Groq(api_key=api_key) if api_key else None

# 3. Layout: Split screen into Camera and Results
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
                # FIX: Convert the camera image bytes into a Base64 encoded string format for Groq API compatibility
                image_bytes = camera_img.getvalue()
                base64_image = base64.b64encode(image_bytes).decode("utf-8")
                
                # Ask Llama-Vision to read the graph data points
                completion = client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Analyze this graph. Extract 5-10 key approximate (x, y) data points. Return ONLY a valid, clean JSON object like this: {\"x\":, \"y\":, \"label\": \"Graph Title\"}. Do not write any conversational intro or markdown backticks."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }
                    ]
                )
                
                # Parse the AI response safely
                raw_response = completion.choices.message.content.strip()
                
                # Safety check to strip accidental markdown code blocks if the AI includes them
                if raw_response.startswith("```json"):
                    raw_response = raw_response.replace("```json", "").replace("```", "").strip()
                elif raw_response.startswith("```"):
                    raw_response = raw_response.replace("```", "").strip()
                    
                data = json.loads(raw_response)
                
                # Plot the interactive graph using Plotly
                df = pd.DataFrame({"X-Axis": data["x"], "Y-Axis": data["y"]})
                fig = px.line(df, x="X-Axis", y="Y-Axis", title=data.get("label", "Detected Graph"), markers=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Generate and show clean Python code to replicate it
                st.write("### 🐍 Python Code to Replicate:")
                generated_code = f"""import plotly.express as px
import pandas as pd

# Data extracted by AI Assistant
data = {{
    'x': {data['x']},
    'y': {data['y']}
}}

df = pd.DataFrame(data)
fig = px.line(df, x='x', y='y', title='{data.get("label", "My Graph")}', markers=True)
fig.show()"""
                st.code(generated_code, language="python")
                
            except Exception as e:
                st.error(f"Error processing the graph. Please ensure the image is clear. Details: {e}")

# 4. Interactive Chat Assistant Feature at the bottom
# FIX: Swapped broken st.hr() with correct official Streamlit divider system
st.divider()
st.subheader("💬 Ask the AI Graph Assistant")
user_question = st.chat_input("Ask a question about your data or math equations...")

if user_question:
    if client:
        with st.chat_message("user"):
            st.write(user_question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chat_completion = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": user_question}]
                )
                st.write(chat_completion.choices.message.content)
    else:
        st.info("Provide your free Groq API key in the sidebar to chat with the assistant.")
