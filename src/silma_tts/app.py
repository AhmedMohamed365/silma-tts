#code inspired from https://huggingface.co/spaces/mrfakename/E2-F5-TTS
import gradio as gr
import tempfile
from importlib.resources import files
from silma_tts.api import SilmaTTS

print("Loading SILMA TTS model...", flush=True)
silma_tts = SilmaTTS()


def do_inference(ref_audio, ref_text, gen_text):

    _, output_wav_path = tempfile.mkstemp(suffix=".wav")

    wav, sr, _ = silma_tts.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=gen_text,
        file_wave=output_wav_path,
    )

    return output_wav_path

app = gr.Interface(
    fn=do_inference,
    inputs=[
        gr.Audio(label="Reference Audio", type="filepath", value=str(files("silma_tts").joinpath("infer/ref_audio_samples/ar.ref.24k.wav"))),
        gr.Textbox(label="Reference Text", value="ويدقق النظر في القرآن الكريم وسائر الكتب السماوية ويتبع مسالك الرسل العظام عليهم الصلاة والسلام."),
        gr.Textbox(label="Generation Text", value="""
أنا نموذج جديد من سلمى لتحويل النص إلى كلام، يمكنني التحدث باللغة العربية مع أو بدون علامات التشكيل.
I am the new SILMA model for converting text to speech, I can speak Arabic with or without diacritics.
        """.strip()
        ),
    ],
    outputs=gr.Audio(label="Generated Speech"),
    title="SILMA TTS Demo",
    description="Add generation text and reference audio then click generate speech.",
)

def main():
    global app
    print("Starting app...")
    app.queue().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()