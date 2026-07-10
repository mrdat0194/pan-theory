import os
import pandas as pd
from PIL import Image
import google.generativeai as genai

# Analysis script for March 2026 Batched GA4 Reports using Gemini


def main():
    # 1. Configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    IMAGE_PATH = os.path.join(BASE_DIR, "ga4_march_batched_visual.png")
    CSV_PATH = os.path.join(BASE_DIR, "ga4_all_accounts_properties_bubbly.csv")

    # 2. Authentication with Gemini API Key
    # Load API key from environment variable GEMINI_API_KEY
    API_KEY = os.environ.get("GEMINI_API_KEY")
    if not API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    print("Configuring Gemini API...")

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
        context_data = df[
            ["Account Name", "Property Name", "Industry", "License"]
        ].to_string()
    except Exception as e:
        print(f"Warning: Could not read CSV data: {e}")
        context_data = "No additional context available from CSV."

    # 4. Load Image
    print(f"Loading image from {IMAGE_PATH}...")
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        return

    img = Image.open(IMAGE_PATH)

    # 5. Initialize Gemini Model
    # Using gemini-3-flash-preview as requested/available
    MODEL_NAME = "gemini-3-flash-preview"
    print(f"Initializing {MODEL_NAME} model...")
    model = genai.GenerativeModel(MODEL_NAME)

    # 6. Construct Multimodal Prompt
    prompt = f"""
    You are an expert Google Analytics 4 (GA4) analyst. 
    Analyze the attached visualization image which shows 15 reports for three major 360 properties for March 2026:
    - VNA (Vietnam Airlines)
    - Vinpearl
    - VinWonders
    
    The properties have the following metadata:
    {context_data}
    
    Tasks:
    1. Provide a concise executive summary of the overall trends shown in the 15 charts for March 2026.
    2. Deep dive into each property group:
       - VNA: Analyze traffic, pages, and revenue patterns.
       - Vinpearl: Evaluate performance trends and user engagement.
       - VinWonders: Assess event volume and hardware/device preferences.
    3. Identify any major anomalies, spikes, or drops in the charts.
    4. Correlate trends across different report types (e.g., does high traffic lead to high revenue?).
    5. Provide 3 high-impact actionable recommendations based on this visualization.
    
    Format your response in professional Markdown.
    """

    # 7. Generate Content
    print("Generating analysis from Gemini (this may take a minute)...")
    import time

    start_time = time.time()
    try:
        response = model.generate_content([prompt, img])
        duration = time.time() - start_time
        print(f"Analysis generated successfully in {duration:.2f} seconds.")

        # Save summary to file
        output_md = os.path.join(BASE_DIR, "march_2026_ga4_analysis.md")
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(response.text)

        print("\n" + "=" * 50)
        print("GEMINI ANALYSIS SUMMARY")
        print("=" * 50)
        print(response.text[:1000] + "...")  # Print first 1000 chars
        print("=" * 50)
        print(f"\nFull analysis saved to: {output_md}")

    except Exception as e:
        duration = time.time() - start_time
        print(f"Error after {duration:.2f} seconds: {e}")


if __name__ == "__main__":
    main()
