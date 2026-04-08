# VidSafe: AI-Based Video Content Moderation
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-purple)
![Model](https://img.shields.io/badge/Model-RT--DETR-blueviolet)
![Model](https://img.shields.io/badge/Model-CLIP-lightgrey)
![Model](https://img.shields.io/badge/Model-FasterWhisper-success)
![Model](https://img.shields.io/badge/Model-RoBERTa-orange)



> VidSafe is an AI-based video moderation system that analyzes both visual and audio content to detect harmful elements such as violence and offensive speech. It processes video frames to identify violent regions and analyzes audio to detect toxic language. Based on these detections, the system selectively moderates only the unsafe parts of the video by blurring harmful visuals and censoring inappropriate audio, while keeping the rest of the content unchanged.
In addition, the system generates a structured moderation report using predefined policies, providing details such as detected violations, timestamps, and recommended actions for better transparency and decision-making.


---

## 🎯 Key Features
- Performs multimodal analysis by processing both video and audio content.
- Detects violent content at the region level within video frames.
- Identifies and censors toxic or offensive speech from audio.
- Applies selective moderation by modifying only unsafe parts instead of removing the entire video.
- Generates policy-based moderation reports with detected violations and timestamps.
---

## 🔄 System Workflow

The VidSafe system follows a structured pipeline to analyze and moderate video content:

1. **Input Processing**  
   The input video is received and prepared for analysis by separating it into visual frames and audio.

2. **Frame Extraction & Audio Separation**  
   Video frames are extracted at regular intervals, and the audio stream is isolated for independent processing.

3. **Semantic Filtering**  
   Relevant frames are selected using CLIP based on similarity to predefined prompts, reducing unnecessary computation.

4. **Violence Detection (Visual Analysis)**  
   Selected frames are processed using RT-DETR to detect regions containing violent content.

5. **Temporal Consistency Filtering**  
   Detections are refined by ensuring they persist across consecutive frames, improving stability.

6. **Audio Transcription & Analysis**  
   The audio is converted to text using speech recognition, and the text is analyzed to detect toxic or offensive language.

7. **Multimodal Alignment**  
   Visual and audio detections are aligned using timestamps to ensure accurate mapping of events.

8. **Selective Moderation**  
   Detected harmful regions are blurred, and toxic audio segments are censored while preserving the rest of the video.

9. **Policy-Based Reasoning & Report Generation**  
   Detected violations are evaluated using predefined policies, and a structured moderation report is generated with timestamps and recommended actions.
---

## 🏗️ System Overview
<img width="1920" height="1080" alt="proposed" src="https://github.com/user-attachments/assets/5c989266-8821-4f67-a655-76901418ea7e" />
The system processes input video through visual and audio analysis modules. The extracted information is fused and evaluated using policy-based reasoning to generate a moderated video and a structured report.

---

## ⚙️ Tech Stack

### AI / Machine Learning
- RT-DETR – region-level violence detection  
- CLIP – semantic frame filtering  
- Faster-Whisper – speech-to-text transcription  
- Detoxify – sentence-level toxicity detection  
- RoBERTa – word-level toxicity analysis  
- LLaMA 3 (via Groq API) – moderation report generation  

### Frameworks & Libraries
- PyTorch  
- OpenCV  
- Hugging Face Transformers  

### UI
- Streamlit – interactive user interface  

### Tools & Utilities
- FFmpeg – audio extraction and processing  
- FAISS – vector database for policy retrieval  

### Environment
- Python 3.8+  
- Google Colab / VS Code

---

## 📂 Dataset & Training

A custom dataset was created for this project due to the lack of publicly available datasets for region-level violence detection in animated content.

- Total frames: 8,401  
- Violent: 3,819  
- Non-violent: 4,582  
- Annotated using CVAT  

🔗 Dataset: [Anime Violence Detection Dataset](https://www.kaggle.com/datasets/aparnamolks/anime-violence-detection-dataset)

The RT-DETR model was trained on this dataset to perform region-level violence detection. The model learns to identify and localize harmful visual content within video frames, enabling precise moderation.

The trained model is then integrated into the VidSafe pipeline for detecting and moderating unsafe video segments.

---

## 📊 Results

The performance of the trained RT-DETR model was evaluated on the custom annotated dataset.

### 🔹 Region-Level Detection
| Metric        | Value |
|--------------|------|
| Precision     | 0.690 |
| Recall        | 0.559 |
| mAP@50        | 0.622 |

### 🔹 Frame-Level Performance
| Metric        | Value |
|--------------|------|
| Accuracy      | 80.5% |
| Precision     | 0.732 |
| Recall        | 0.918 |
| F1 Score      | 0.815 |

--- 

## 🎬 Sample Output

### 🔴 Before Moderation
![Before](https://github.com/user-attachments/assets/35d916c7-9440-42fe-beb0-e2fa433aa1b7)

*Original video frame containing violent visual content.*

---

### 🟢 After Moderation
![After](https://github.com/user-attachments/assets/6bba9a26-210b-45e3-b19a-1432bed2044f)


*Detected violent regions are blurred, and harmful content is selectively moderated while preserving the rest of the video.*

---

### 📌 Moderation Report (Optional)
![Report](https://github.com/user-attachments/assets/a5fd2175-ce6a-49f8-9e1c-83c37af92292)


*A structured report generated using policy-based reasoning, showing detected violations, timestamps, and recommended actions.*

---

## ⚙️ Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Aparnamol-KS/VidSafe.git
cd vidsafe
```
### 2. Install Dependencies
```
pip install -r requirements.txt
```
### 3. Run the Application
```
streamlit run app.py
```
---

## 🚀 Future Scope

- Extend the system for real-time video moderation  
- Introduce age-based or user-specific content filtering  
- Expand detection to additional categories (e.g., explicit or sensitive content)  
- Improve multimodal fusion for handling complex scenarios  
- Optimize the system for deployment on large-scale platforms

---

## 👥 Team

- **Alicia Therese Dominic** – [GitHub](https://github.com/AliciaTherese)  
- **Aparnamol K S** – [GitHub](https://github.com/Aparnamol-KS)  
- **Hazel Nilson** – [GitHub](https://github.com/Hazel2004)

> Developed as part of a B.Tech AI & DS project.
