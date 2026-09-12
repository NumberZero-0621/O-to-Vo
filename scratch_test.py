import whisperx
import pykakasi
import torch

kks = pykakasi.kakasi()

def to_hiragana(text):
    result = kks.convert(text)
    return ''.join([item['hira'] for item in result])

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
audio_path = "vocal.wav"

model = whisperx.load_model("large-v2", device, compute_type=compute_type)
audio = whisperx.load_audio(audio_path)
result = model.transcribe(audio, batch_size=8)

print("Original transcript:")
for seg in result["segments"]:
    print(seg["text"])

# Convert text to hiragana before alignment
for seg in result["segments"]:
    seg["text"] = to_hiragana(seg["text"])

print("\nHiragana transcript:")
for seg in result["segments"]:
    print(seg["text"])

model_a, metadata = whisperx.load_align_model(language_code="ja", device=device)
aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=True)

print("\nAligned characters:")
for segment in aligned_result["segments"]:
    if "chars" in segment:
        for char_info in segment["chars"]:
             print(char_info)
    break
