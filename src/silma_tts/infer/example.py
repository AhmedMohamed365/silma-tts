
# SILMA TTS: A Lightweight Open Bilingual Text to Speech Model 


[![HF](https://img.shields.io/badge/-%F0%9F%A4%97%20Hugging%20Face-black)](https://huggingface.co/silma-ai/silma-tts)
[![hfspace](https://img.shields.io/badge/🤗-HF%20Space-yellow)](https://huggingface.co/spaces/silma-ai/silma-tts-v1-demo)


**SILMA TTS v1** is a high-performance, **150M-parameter** bilingual (Arabic/English) TTS model developed by [SILMA AI](https://silma.ai). Built on the cutting-edge **F5-TTS diffusion architecture**, the model was **pretrained from scratch** using tens of thousands of hours of high-quality public and proprietary data. To give back to the community, SILMA TTS is released under a highly permissive license, making state-of-the-art speech synthesis accessible for both **research and commercial use**.

## Installation

### Using pip 

```bash

# make sure ffmpeg is installed
apt-get update && apt install ffmpeg -y

# create and activate the environment
python -m venv silma-tts-env
source silma-tts-env/bin/activate

# install silma-tts library
pip install silma-tts

```

### From source

```bash

# make sure ffmpeg is installed
apt-get update && apt install ffmpeg -y

# create and activate the environment
python -m venv silma-tts-env
source silma-tts-env/bin/activate

# clone repo and install
git clone https://github.com/SILMA-AI/silma-tts.git
cd silma-tts
pip install -e .

```


## Usage & Inference

### Using gradio app

```bash

# Run the following command
silma-tts-app

```

Then open the following browser link
http://127.0.0.1:7860/


### Inference using python

```python

from silma_tts.api import SilmaTTS

silma_tts = SilmaTTS()

## the voice/style you want to clone
reference_audio_file = "src/infer/ref_audio_samples/ar.ref.24k.wav"
## the transcription of the reference_audio_file
reference_audio_text = "ويدقق النظر في القرآن الكريم وسائر الكتب السماوية ويتبع مسالك الرسل العظام عليهم الصلاة والسلام."


wav, sr, spec = silma_tts.infer(
    ref_file=reference_audio_file,
    ref_text=reference_audio_text, # can also be left None - will be transcribed on the fly
    gen_text="""
    أنا نموذج جديد من سلمى لتحويل النص إلى كلام؛ يمكنني التحدث باللغة العربية مع أو بدون علامات التشكيل.
    I am the new SILMA model for converting text to speech, I can speak Arabic with or without diacritics.
    """.strip(),
    file_wave=str("generated_audio.wav"),
    seed=None,
)

## Note 1: generated audio file (generated_audio.wav) will be saved in the current directory
## Note 2: You can also use the "wav" variable (raw waveform) to play the audio to return it via API

```

You can run the example above directly using the following command:

```bash

python src/silma_tts/infer/example.py

```


## Training

Our model is 100% compatible with [F5-TTS v1.1.7](https://github.com/SWivid/F5-TTS/releases/tag/1.1.7). This means you can make use of all the great resources and community experince in F5-TTS project


## Acknowledgements

This repo builds directly upon the excellent foundation laid by the [F5-TTS](https://github.com/SWivid/F5-TTS) project. The core architecture and the majority of the code is derived from their work. Our work introduces new pretrained weights and significant optimizations to the inference code.


## Support

Unfortunately we don't have capacity to actively support this repo. We also believe it is best to consolidate resources and knowledge in a single location. This is why we encourage you to visit the official F5-TTS repository, which is highly active and supported by a fantastic community.

* Please check if your question is already answered in the [F5-TTS Issues](https://github.com/SWivid/F5-TTS/issues?q=is%3Aissue) or open a new issue
* We monitor the F5-TTS Issues and will engage there if needed
* If you have a question that you think only us can answer, then please open a [community discussion](https://huggingface.co/silma-ai/silma-tts/discussions) on our HuggingFace repo
 


## Citation

```
@article{silma-tts-v1,
      title={SILMA TTS: A Lightweight Open Bilingual Text to Speech Model }, 
      author={SILMA AI},
      year={2026},
}
```
## License
1. Code: MIT License
2. Model Weights: Apache-2.0 License


