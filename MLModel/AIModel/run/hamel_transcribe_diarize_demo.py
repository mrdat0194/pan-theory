import json

def run_hamel_pipeline(audio_path=None):
    """
    Simulates the standard Hamel Transcription & Diarization Pipeline
    using Whisper + PyAnnote.
    Returns a mock transcript with speaker labels.
    """
    return [
        {"start": 0.248, "end": 13.531, "speaker": "SPEAKER_01", "text": "Hi, this is Jeremy Howard, and you're listening to Coffee Time Data Science, a podcast for data science enthusiasts, where I interview practitioners, researchers, and Kagglers about their journey, experience, and talk all things data science."},
        {"start": 13.531, "end": 17.151, "speaker": "SPEAKER_01", "text": "And before we begin, I apologize for the change to our schedule."},
        {"start": 17.151, "end": 22.593, "speaker": "SPEAKER_01", "text": "Of course, usually you would be seeing Chai Time Data Science on this channel with Sanyam Bhutani."},
        {"start": 22.593, "end": 24.373, "speaker": "SPEAKER_01", "text": "Unfortunately, he's not available today."},
        {"start": 24.373, "end": 29.514, "speaker": "SPEAKER_01", "text": "He had a prior appointment on another podcast, and he was not able to join Chai Time Data Science."},
        {"start": 29.974, "end": 34.338, "speaker": "SPEAKER_01", "text": "We hope you enjoy this special episode of Coffee Time Data Science."},
        {"start": 34.338, "end": 45.148, "speaker": "SPEAKER_01", "text": "And without further ado, I would like to invite our very special VIP guest, newly anointed Kaggle Grand Master, Sanyam Bhutani."},
        {"start": 45.148, "end": 47.190, "speaker": "SPEAKER_01", "text": "Sanyam, welcome to Coffee Time Data Science."},
        {"start": 48.372, "end": 49.073, "speaker": "SPEAKER_00", "text": "Thank you, Jeremy."},
        {"start": 49.073, "end": 53.537, "speaker": "SPEAKER_00", "text": "Usually, I'm very anti coffee, but I'll have to allow that."},
        {"start": 53.537, "end": 55.678, "speaker": "SPEAKER_00", "text": "I still can't believe you weren't kidding."},
        {"start": 55.678, "end": 59.421, "speaker": "SPEAKER_00", "text": "And I mentioned in our message also, like I, I think I don't deserve this."},
        {"start": 59.421, "end": 60.042, "speaker": "SPEAKER_00", "text": "But thank you."}
    ]

if __name__ == "__main__":
    result = run_hamel_pipeline()
    print("Hamel Pipeline Result:", json.dumps(result, indent=2))
