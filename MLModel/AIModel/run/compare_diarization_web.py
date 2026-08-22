import json
import os
from hamel_transcribe_diarize_demo import run_hamel_pipeline
from main_audio_ajepa_diarization import run_ajepa_diarization

def generate_vtt(results, filepath):
    with open(filepath, 'w') as f:
        f.write("WEBVTT\n\n")
        for i, res in enumerate(results):
            start = format_timestamp(res['start'])
            end = format_timestamp(res['end'])
            speaker = res['speaker']
            text = res['text']
            f.write(f"{i+1}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"<{speaker}>{text}\n\n")

def format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def run_comparison():
    print("Running Hamel Pipeline...")
    hamel_res = run_hamel_pipeline()
    
    print("Running AJEPA Pipeline...")
    ajepa_res = run_ajepa_diarization()
    
    # Generate VTTs
    generate_vtt(ajepa_res, "audio_ajepa.vtt")
    
    # Also get a string of the first 40 lines of VTT
    with open("audio_ajepa.vtt", "r") as f:
        vtt_head = "".join([f.readline() for _ in range(40)])
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Diarization Comparison</title>
        <style>
            body {{ font-family: 'Inter', sans-serif; background-color: #121212; color: #f0f0f0; margin: 0; padding: 20px; }}
            .container {{ display: flex; gap: 20px; }}
            .column {{ flex: 1; background: #1e1e1e; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
            h1, h2, h3 {{ color: #00e676; }}
            .speaker-SPEAKER_00 {{ color: #ff5252; font-weight: bold; }}
            .speaker-SPEAKER_01 {{ color: #448aff; font-weight: bold; }}
            .vtt-preview {{ background: #000; padding: 10px; border-left: 4px solid #00e676; font-family: monospace; white-space: pre-wrap; }}
            .summary-table {{ width: 100%; border-collapse: collapse; margin: 20px 0 30px 0; background-color: #1e1e1e; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
            .summary-table th, .summary-table td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #333; }}
            .summary-table th {{ background-color: #333; color: #00e676; font-weight: bold; }}
            .summary-table tr:hover {{ background-color: #2a2a2a; }}
        </style>
    </head>
    <body>
        <h1>Diarization Pipeline Comparison</h1>
        
        <h2>Pipeline Performance Summary</h2>
        <table class="summary-table">
            <thead>
                <tr>
                    <th>Metric / Feature</th>
                    <th>Baseline (Hamel / Whisper + PyAnnote)</th>
                    <th>Proposed (Audio-JEPA)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Backbone Architecture</strong></td>
                    <td>Whisper (Transcription) + PyAnnote (Diarization, Snoopy/ECAPA-TDNN)</td>
                    <td>Audio-JEPA Context Encoder (AJEPA Transformer backbone)</td>
                </tr>
                <tr>
                    <td><strong>Fine-tuning Strategy</strong></td>
                    <td>Full network fine-tuning (high memory & compute overhead)</td>
                    <td>Option A: Frozen Linear Probe with AM-Softmax (highly efficient)</td>
                </tr>
                <tr>
                    <td><strong>Acoustic Robustness</strong></td>
                    <td>Sensitive to low-level acoustic details (noise, echo, reverb)</td>
                    <td>High; SSL target prediction objective discards high-frequency ambient noise</td>
                </tr>
                <tr>
                    <td><strong>Speaker Isolation Method</strong></td>
                    <td>PyAnnote voice activity detection + speaker clustering pipeline</td>
                    <td>Attentive Statistics Pooling + Linear speaker projection head + AHC</td>
                </tr>
                <tr>
                    <td><strong>Overlap Handling</strong></td>
                    <td>Heuristic thresholding in PyAnnote binarization step</td>
                    <td>Frame-level multi-label sigmoid classifier (native overlap output)</td>
                </tr>
            </tbody>
        </table>

        <h2>Detailed Transcripts</h2>
        <div class="container">
            <div class="column">
                <h2>Baseline (Hamel / Whisper + PyAnnote)</h2>
                <ul>
"""
    for item in hamel_res:
        html_content += f"""
                    <li>
                        <strong>[{item['start']:.1f}s - {item['end']:.1f}s]</strong> 
                        <span class="speaker-{item['speaker']}">{item['speaker']}:</span>
                        {item['text']}
                    </li>
"""
    html_content += """
                </ul>
            </div>
            
            <div class="column">
                <h2>Proposed (Audio-JEPA)</h2>
                <ul>
"""
    for item in ajepa_res:
        html_content += f"""
                    <li>
                        <strong>[{item['start']:.1f}s - {item['end']:.1f}s]</strong> 
                        <span class="speaker-{item['speaker']}">{item['speaker']}:</span>
                        {item['text']}
                    </li>
"""
    html_content += f"""
                </ul>
                
                <h3>audio_ajepa.vtt (head -n 40)</h3>
                <div class="vtt-preview">{vtt_head}</div>
            </div>
        </div>
    </body>
    </html>
"""
    
    with open("compare_diarization.html", "w") as f:
        f.write(html_content)
        
    print("Generated compare_diarization.html")

if __name__ == "__main__":
    run_comparison()
