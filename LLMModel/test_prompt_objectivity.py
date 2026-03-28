
import os
import sys
from unittest.mock import MagicMock, patch

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from LLMModel.query_questionandanswer_vector_db import gemini_answer

def test_prompt_objectivity():
    print("Testing prompt objectivity...")
    
    context = "100% dữ liệu lưu trong BigQuery của VNA. Vortex không giữ. Quyền sở hữu và kiểm soát hoàn toàn thuộc về VNA."
    question = "Data có an toàn không? Ai sở hữu?"
    
    # Mock genai and GOOGLE_API_KEY
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"}), \
         patch("google.generativeai.GenerativeModel") as mock_model_class:
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        # We want to see if the prompt sent to generate_content contains our instructions
        gemini_answer(question, context, "gemini-1.5-flash")
        
        # Check the call arguments
        args, kwargs = mock_model.generate_content.call_args
        prompt = args[0]
        
        print("\nConstructed Prompt:")
        print(prompt)
        
        assert "objective AI assistant" in prompt
        assert "generalize the answer" in prompt
        assert "white-label" in prompt
        print("\nPrompt correctly contains objectivity instructions!")

if __name__ == "__main__":
    try:
        import google.generativeai
        test_prompt_objectivity()
    except ImportError:
        print("google-generativeai not installed, skipping real mock test, just checking file content manually.")
        with open("LLMModel/query_questionandanswer_vector_db.py", "r", encoding="utf-8") as f:
            content = f.read()
            if "objective AI assistant" in content:
                print("Code change verified in file content.")
            else:
                print("Code change NOT found in file content!")
                sys.exit(1)
