import os
import json
import pandas as pd
from PIL import Image
import google.generativeai as genai
from google.oauth2 import service_account

def main():
    # 1. Configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_PATH = os.path.join(BASE_DIR, 'ga4_multi_property_visual.png')
    CSV_PATH = os.path.join(BASE_DIR, 'ga4_all_accounts_properties_bubbly.csv')
    JSON_PATH = os.path.join(BASE_DIR, 'bubbly-cascade-398303-5f3dd0a21703.json')
    
    # 2. Authentication with Gemini API Key
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    if API_KEY:
        print(f"Configuring Gemini API with key ending in {API_KEY[-4:]}...")
    else:
        print("Configuring Gemini API (No API Key found in environment)...")
    
    try:
        genai.configure(api_key=API_KEY)
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        return

    # 3. Load CSV Data for Context
    print(f"Loading context data from {CSV_PATH}...")
    try:
        df = pd.read_csv(CSV_PATH)
        context_data = df[['Account Name', 'Property Name', 'Industry', 'License']].to_string()
    except Exception as e:
        print(f"Warning: Could not read CSV data: {e}")
        context_data = "No additional context available from CSV."

    # 4. Load Image
    print(f"Loading image from {IMAGE_PATH}...")
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        return
    
    img = Image.open(IMAGE_PATH)

    # 5. Initialize Gemini Flash
    print("Initializing Gemini-2.5-Flash model...")
    # Using 'gemini-2.5-flash' as requested by the user
    model = genai.GenerativeModel('gemini-2.5-flash')

    # 6. Construct Multimodal Prompt
    prompt = f"""
    You are an expert Google Analytics 4 (GA4) analyst. 
    Analyze the attached visualization image which shows 15 reports for three different properties:
    VNA (Vietnam Airlines), Vinpearl, and VinWonders.
    
    The properties have the following metadata:
    {context_data}
    
    Tasks:
    1. Provide a concise executive summary of the overall trends shown in the 15 charts.
    2. Deep dive into each property group:
       - VNA: What are the top performers and any significant patterns?
       - Vinpearl: How is the traffic and revenue looking?
       - VinWonders: Any specific insights from the events and hardware reports?
    3. Identify any major anomalies or outliers that require immediate attention.
    4. Provide 3 actionable recommendations based on the visual data.
    
    Format your response in professional Markdown.
    """

    # 7. Generate Content
    print("Generating context from Gemini Flash (this may take a minute)...")
    import time
    start_time = time.time()
    try:
        response = model.generate_content([prompt, img])
        duration = time.time() - start_time
        print(f"Content generated successfully in {duration:.2f} seconds.")
        
        print("\n" + "="*50)
        print("GEMINI ANALYSIS SUMMARY")
        print("="*50)
        print(response.text)
        print("="*50)
        
        # Save summary to file
        output_txt = os.path.join(BASE_DIR, 'ga4_gemini_analysis.md')
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"\nAnalysis saved to: {output_txt}")
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"Error after {duration:.2f} seconds: {e}")
        if "403" in str(e):
            print("\nIMPORTANT: Please ensure the 'Generative Language API' is enabled for project 170401117126 in the Google Cloud Console.")
        elif "429" in str(e):
            print("\nRate limit exceeded. Please wait a moment and try again.")

if __name__ == '__main__':
    main()
