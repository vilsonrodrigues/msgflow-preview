# Quickstart – Build a Multimodal Voice Assistant with msgFlow

Welcome to your first project with **msgFlow**! In just a few lines, you'll create a **multimodal voice assistant** that:

- Transcribes a voice command  
- Uses a language model to generate a smart reply  
- Speaks the response aloud  
- And even generates an image related to the response  

Let’s dive in. This will take less than 5 minutes!

---

## Installation

First, install `msgflow`:

```bash
pip install msgflow
```

Make sure you have a working Python environment with `ffmpeg` installed (for audio handling).

---

## Components We’ll Use

This quickstart will demonstrate several of the **core modules** of `msgFlow`:

- `Transcriber`: Converts speech to text using models like Whisper.
- `Agent`: Generates replies using LLMs like GPT-3.5.
- `Speaker`: Synthesizes text into voice.
- `Designer`: Creates images from text prompts using models like DALL·E.

All of them follow a `torch.nn.Module`-like API.

---

## Building the Workflow

```python
import msgflow.nn as nn

# Load the modules
transcriber = nn.Transcriber.from_pretrained("whisper-tiny")
agent = nn.Agent.from_pretrained("gpt-3.5")
speaker = nn.Speaker.from_pretrained("tts-xyz")
designer = nn.Designer.from_pretrained("dalle-mini")

# Define the assistant as a flow of modules
class VoiceAssistant(nn.Module):
    def __init__(self):
        super().__init__()
        self.transcriber = transcriber
        self.agent = agent
        self.speaker = speaker
        self.designer = designer

    def forward(self, audio):
        text = self.transcriber(audio)
        reply = self.agent(text)
        audio_reply = self.speaker(reply)
        image = self.designer(reply)
        return {
            "input_text": text,
            "reply_text": reply,
            "reply_audio": audio_reply,
            "reply_image": image,
        }
```

---

## 🧪 Running the Assistant

Let’s run the assistant on a sample `.wav` file:

```python
assistant = VoiceAssistant()
result = assistant("audio/what_does_a_cat_on_mars_look_like.wav")

print("You said:", result["input_text"])
print("Assistant said:", result["reply_text"])

# Optionally play audio and display image
from utils import play_audio, show_image
play_audio(result["reply_audio"])
show_image(result["reply_image"])
```

---

## 🎉 What Just Happened?

- You gave an **audio command**
- It was transcribed, processed by a **language model**, converted to **speech**, and visually **illustrated**
- All with composable, versionable blocks you can expand and trace

---

## 🧩 Next Steps

- Swap the `Agent` for a custom fine-tuned LLM
- Use `Retriever` to inject external knowledge
- Chain in a `Database` to make your assistant persistent
- Deploy it as an API or in a UI

---

msgFlow makes it simple to **build complex, multimodal AI systems** in a structured and reproducible way.

Ready to go deeper? → [Check out the full examples →](link_to_full_examples)

---

Se quiser, posso adaptar esse conteúdo direto em um `.py` comentado também, ou criar variações com foco em outros domínios (ex: visão, tabular, etc.). Quando tiver a primeira versão do seu script, manda aqui que a gente lapida juntos!