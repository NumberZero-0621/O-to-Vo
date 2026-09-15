import torch
import whisperx
import torchaudio
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from core.text_utils import get_word_moras

def filter_short_w2v2_segments(segments, min_duration=0.03):
    """
    Wav2Vec2が検出したセグメントの開始タイミングを比較し、
    1つ前のセグメントとの開始タイミングの差が極端に短い（30ms未満など）場合、そのセグメントを除外する。
    """
    if not segments:
        return []
        
    filtered = [segments[0]]
    for seg in segments[1:]:
        last_seg = filtered[-1]
        # 前のノートの開始タイミングとの差分を計算
        if (seg["start"] - last_seg["start"]) >= min_duration:
            filtered.append(seg)
    return filtered

# ==========================================
# 1. WhisperXで高精度なタイムスタンプを取得
# ==========================================
def load_whisperx_models(whisper_model_name="large-v3", w2v2_model_name="vumichien/wav2vec2-large-xlsr-japanese-hiragana"):
    print(f"WhisperXモデル({whisper_model_name}) および アライメントモデル({w2v2_model_name}) をロード中...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"使用デバイス: {device} ({compute_type})")

    model = whisperx.load_model(whisper_model_name, device, compute_type=compute_type, language="ja")
    model_a, metadata = whisperx.load_align_model(language_code="ja", device=device, model_name=w2v2_model_name)
    
    return model, model_a, metadata, device

def process_whisperx_chunk(audio_path, model, model_a, metadata, device, offset_seconds=0.0):
    audio = whisperx.load_audio(audio_path)
    
    try:
        result = model.transcribe(audio, batch_size=4)
    except Exception as e:
        print(f"transcribeエラー: {e}")
        return []

    hallucination_keywords = ["ご視聴ありがとうございました", "ごしちょーありがとーございました", "チャンネル登録", "お借りした素材"]
    valid_segments = []
    
    for segment in result["segments"]:
        text = segment["text"]
        # ハルシネーションチェック（元のテキストでチェック）
        is_hallu = False
        for kw in hallucination_keywords:
            if kw in text:
                is_hallu = True
                break
        
        if is_hallu:
            print(f"  [ハルシネーション除外] {text}")
            continue
            
        moras = get_word_moras(text)
        hira_text = "".join(moras)
        
        # ひらがな化後のテキストでもチェック
        for kw in hallucination_keywords:
            if kw in hira_text:
                is_hallu = True
                break
        
        if is_hallu:
            print(f"  [ハルシネーション除外] {hira_text}")
            continue

        segment["text"] = hira_text
        valid_segments.append(segment)

    result["segments"] = valid_segments

    if not result["segments"]:
        return []

    try:
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=True)
    except Exception as e:
        print(f"alignエラー: {e}")
        return []
        
    char_segments = []
    
    for segment in result["segments"]:
        for char_info in segment.get("chars", []):
            char_text = char_info.get("char", "")
            if not char_text.strip():
                continue
            if "start" not in char_info or "end" not in char_info:
                continue
                
            char_segments.append({
                "text": char_text,
                "start": float(char_info["start"]) + offset_seconds,
                "end": float(char_info["end"]) + offset_seconds
            })

    return char_segments

# ==========================================
# 1.5. Wav2Vec2の純粋なCTCデコード（デバッグ用）
# ==========================================
def load_wav2vec2_ctc_model(w2v2_model_name="vumichien/wav2vec2-large-xlsr-japanese-hiragana"):
    print(f"Wav2Vec2(CTC)モデル({w2v2_model_name})をロード中...")
    processor = Wav2Vec2Processor.from_pretrained(w2v2_model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Wav2Vec2ForCTC.from_pretrained(w2v2_model_name).to(device)
    return model, processor, device

def process_wav2vec2_ctc_chunk(audio_path, model, processor, device, offset_seconds=0.0):
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        logits = model(inputs.input_values).logits
        
    predicted_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy()
    
    vocab = processor.tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    
    # vumichien/wav2vec2-large-xlsr-japanese-hiragana のストライドから計算したフレーム長は20ms
    frame_duration = 0.02
    
    char_segments = []
    current_char = None
    start_frame = None
    
    for i, token_id in enumerate(predicted_ids):
        token = id_to_token.get(token_id, "")
        
        # [PAD] (CTCのブランク)の処理
        if token == "[PAD]":
            if current_char is not None:
                char_segments.append({
                    "text": current_char,
                    "start": start_frame * frame_duration + offset_seconds,
                    "end": i * frame_duration + offset_seconds
                })
                current_char = None
            continue
            
        # 文字が変わったか、ブランクの後に新しい文字が来た場合
        if token != current_char:
            if current_char is not None:
                char_segments.append({
                    "text": current_char,
                    "start": start_frame * frame_duration + offset_seconds,
                    "end": i * frame_duration + offset_seconds
                })
            current_char = token.replace('|', '')
            start_frame = i

    # 最後の文字
    if current_char is not None:
        char_segments.append({
            "text": current_char,
            "start": start_frame * frame_duration + offset_seconds,
            "end": len(predicted_ids) * frame_duration + offset_seconds
        })
        
    # 空白文字および除外対象文字を除去し、リストを返す
    exclude_chars = "’、っ。！?？（）() 　.,'\"-"
    char_segments = [s for s in char_segments if s["text"].strip() and s["text"] not in exclude_chars]
    return char_segments

def compute_forced_alignment(audio_path, whisper_segments, model, processor, device, offset_seconds=0.0):
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        logits = model(inputs.input_values).logits
        
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    vocab = processor.tokenizer.get_vocab()
    blank_id = processor.tokenizer.pad_token_id
    id_to_token = {v: k for k, v in vocab.items()}
    
    tokens = []
    token_to_whisper_seg = []
    for seg in whisper_segments:
        for char in seg["text"]:
            if char in vocab:
                tokens.append(vocab[char])
                token_to_whisper_seg.append(seg)
            elif char.replace('|', '') in vocab:
                tokens.append(vocab[char.replace('|', '')])
                token_to_whisper_seg.append(seg)
                
    if not tokens:
        return []
        
    targets = torch.tensor(tokens, dtype=torch.int32).unsqueeze(0).to(device)
    
    try:
        aligned_path, _ = torchaudio.functional.forced_align(log_probs, targets, blank=blank_id)
        aligned_path = aligned_path[0].cpu().numpy()
    except Exception as e:
        print(f"Forced alignment failed: {e}")
        return []
        
    frame_duration = 0.02
    char_segments = []
    
    target_idx = 0
    current_char_start_frame = -1
    
    for i, token_id in enumerate(aligned_path):
        if target_idx < len(tokens):
            expected_token = tokens[target_idx]
        else:
            break
            
        if token_id == expected_token:
            if current_char_start_frame == -1:
                current_char_start_frame = i
        else:
            if current_char_start_frame != -1:
                end_frame = i
                char_text = id_to_token[expected_token].replace('|', '')
                char_segments.append({
                    "text": char_text,
                    "start": current_char_start_frame * frame_duration + offset_seconds,
                    "end": end_frame * frame_duration + offset_seconds
                })
                current_char_start_frame = -1
                target_idx += 1
                
                if token_id != blank_id and target_idx < len(tokens) and token_id == tokens[target_idx]:
                    current_char_start_frame = i

    if current_char_start_frame != -1 and target_idx < len(tokens):
        char_text = id_to_token[tokens[target_idx]].replace('|', '')
        char_segments.append({
            "text": char_text,
            "start": current_char_start_frame * frame_duration + offset_seconds,
            "end": len(aligned_path) * frame_duration + offset_seconds
        })
        
    return char_segments

