# 🚀 Future Contributions & Feature Roadmap — Growmetrics AI Video Service

Welcome to the **Growmetrics AI Video Generation Service** contribution guide and feature roadmap! Growmetrics is designed as a modular, resilient microservice for generating educational vector animations and narrations. 

This document outlines planned architectural enhancements, upcoming feature tracks, and guidelines for contributors looking to extend the platform.

---

## 🗺️ Strategic Product Roadmap

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                               GROWMETRICS ROADMAP                                 │
├──────────────────────┬────────────────────────┬───────────────────────────────────┤
│ Phase 1: Core Engine │ Phase 2: A2A & RAG     │ Phase 3: Distributed Scale        │
│  (Current Version)   │   (Near-Term)          │   (Long-Term Vision)              │
├──────────────────────┼────────────────────────┼───────────────────────────────────┤
│ • Manim + Edge-TTS   │ • A2A Protocol         │ • Celery/Redis GPU Worker Pools   │
│ • 4 Quality Gates    │ • Service-Agent RAG    │ • Vision-LLM Gate 5 Validation    │
│ • Self-Correction    │ • Multi-Subject Support│ • S3/GCS Cloud Blob Storage       │
│ • Fallback Router    │ • Multi-Voice Narration│ • Blender / 3D Hybrid Renderer    │
└──────────────────────┴────────────────────────┴───────────────────────────────────┘
```

---

## 💡 Planned Technical Enhancements

### 1. 🤝 Agent-to-Agent (A2A) & RAG Integration
- **A2A Protocol Adapter**: Expose standardized A2A RPC endpoints (`/a2a/v1/tasks`) so knowledge agents (like **Service-Agent**) can delegate video production requests asynchronously.
- **RAG-Grounded Scene Specs**: Ingest verified domain context and citation metadata from upstream RAG agents to eliminate chemical/scientific hallucinations in LLM-generated scripts.
- **Bi-directional Feedback**: Transmit quality gate failure reports back to upstream agents via A2A event streaming to request updated domain facts or script expansions.

### 2. 🎨 Animation & Rendering Enhancements
- **Multi-Scene Storytelling**: Support multi-scene sequences (`ChemistrySceneIntro`, `ChemistrySceneMain`, `ChemistrySceneOutro`) stitched seamlessly with Manim transitions.
- **Multi-Subject Domain Expansion**: Extend beyond chemistry to physics (kinematics, electromagnetism), mathematics (calculus, linear algebra), and computer science algorithms.
- **3D & Hybrid Rendering**: Integrate 3D Manim scenes (`ThreeDScene`) and optional Blender / Remotion render engine drivers for motion graphic overlays.

### 3. 🎙️ Advanced Audio & Narration Pipeline
- **Multi-Speaker Dialogues**: Enable two-voice educational dialogues (e.g., teacher and student asking clarifying questions).
- **Background Music (BGM) & Audio Ducking**: Auto-mix royalty-free background music track with automatic volume ducking during Edge-TTS speech segments.
- **Multi-Language Localization**: Support automatic translation and multi-lingual voice synthesis (e.g., Vietnamese `vi-VN-HoaiMyNeural`, Spanish, French).

### 4. 🛡️ Quality Gate 5: Vision-LLM Visual Inspection
- **Frame-by-Frame Visual Audit**: Introduce a 5th Quality Gate powered by a Vision LLM (e.g., Gemini 2.0 Flash / GPT-4o vision) to inspect rendered video frames for text overlap, off-screen elements, or visual glitches before declaring job completion.
- **SSIM/PSNR Automated Testing**: Implement structural similarity regression testing against reference gold-standard animations.

### 5. ⚡ Infrastructure & High-Throughput Scaling
- **Asynchronous Task Workers**: Migrate in-memory job queue to **Redis + Celery / RQ** with worker auto-scaling for parallel rendering.
- **Cloud Blob Storage**: Abstract artifact storage (`artifact_store.py`) to support AWS S3, Google Cloud Storage (GCS), or Azure Blob Storage.
- **Containerized GPU Rendering**: Package Manim rendering environments into lightweight Docker containers optimized for GPU acceleration (NVIDIA CUDA / Cairo hardware rendering).

---

## 🛠️ Contribution Guidelines

We welcome contributions from the community! Follow these steps to contribute code or templates:

### How to Contribute:

1. **Adding Parametric Fallback Templates**:
   - Location: `app/generator/templates/`
   - Create new deterministic Manim scene templates for missing chemistry topics to ensure 100% fallback coverage.

2. **Improving Quality Gates**:
   - Location: `app/generator/quality_gates.py`
   - Add new AST safety rules or stricter FFprobe audio-video sync checks.

3. **Adding New LLM Providers**:
   - Location: `app/generator/agent.py`
   - Implement custom LLM provider functions (e.g., Ollama, vLLM, Anthropic Claude) adhering to the `_call_llm(system_prompt, user_message)` signature.

---

## 🧪 Development & Testing Workflow

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run unit tests & quality gate suite
pytest tests/test_quality_gates.py tests/test_api.py -v

# Run 5x consistency benchmark
pytest tests/test_consistency.py -v -s --timeout=600
```
