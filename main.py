import os
import warnings
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk

# ==========================================
# ログ・警告の抑制設定
# ==========================================
# TensorFlowおよびoneDNNの情報ログ・警告を抑制
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# PyannoteやTransformersからのUserWarning等を無視
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="pyannote")

import torchaudio
import librosa
import numpy as np
from scipy.io import wavfile
import torch
import gc
import re
import pyworld as pw
import pyopenjtalk
import jaconv
import whisperx
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# PyannoteのReproducibilityWarning (TF32関連) を抑制
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 漢字・カタカナ等をひらがなのモーラに分割する関数
def get_word_moras(text):
    """pyopenjtalkとjaconvを用いてテキストをひらがなのモーラ（文字）リストに変換する"""
    if not text or text.strip() == "":
        return []
    
    try:
        features = pyopenjtalk.run_frontend(text)
        prons = []
        for f in features:
            pron = f.get('pron', '')
            if pron and pron != '*':
                prons.append(pron)
        
        if not prons:
            prons = [text]
            
        kata_str = "".join(prons)
        hira_str = jaconv.kata2hira(kata_str)
        
        # 除外対象文字のフィルタリング (ユーザー要望: ’、っ 等)
        exclude_chars = "’、っ。！?？（）() 　.,'\"-"
        moras = [c for c in hira_str if c not in exclude_chars]
        
        return moras
    except Exception as e:
        print(f"pyopenjtalk変換エラー: {e}")
        # フォールバックとして元の文字列を文字ごとに分割して返す
        exclude_chars = "’、っ。！?？（）() 　.,'\"-"
        return [c for c in text if c not in exclude_chars]

# 日本語文字から母音を判定するためのマッピング
def get_vowel(text):
    if not text:
        return "あ"
    # 最後の1文字で判定（きゃ、等の場合は2文字目が小文字だが、その母音に従う）
    char = text[-1]
    vowel_table = {
        "あ": "あ", "い": "い", "う": "う", "え": "え", "お": "お",
        "か": "あ", "き": "い", "く": "う", "け": "え", "こ": "お",
        "さ": "あ", "し": "い", "す": "う", "せ": "え", "そ": "お",
        "た": "あ", "ち": "い", "つ": "う", "て": "え", "と": "お",
        "な": "あ", "に": "い", "ぬ": "う", "ね": "え", "の": "お",
        "は": "あ", "ひ": "い", "ふ": "う", "へ": "え", "ほ": "お",
        "ま": "あ", "み": "い", "む": "う", "め": "え", "も": "お",
        "や": "あ", "ゆ": "う", "よ": "お",
        "ら": "あ", "り": "い", "る": "う", "れ": "え", "ろ": "お",
        "わ": "あ", "を": "お", "ん": "ん",
        "が": "あ", "ぎ": "い", "ぐ": "う", "げ": "え", "ご": "お",
        "ざ": "あ", "じ": "い", "ず": "う", "ぜ": "え", "ぞ": "お",
        "だ": "あ", "ぢ": "い", "づ": "う", "で": "え", "ど": "お",
        "ば": "あ", "び": "い", "ぶ": "う", "べ": "え", "ぼ": "お",
        "ぱ": "あ", "ぴ": "い", "ぷ": "う", "ぺ": "え", "ぽ": "お",
        "ゃ": "あ", "ゅ": "う", "ょ": "お", "ゎ": "あ",
        "ァ": "あ", "ィ": "い", "ゥ": "う", "ェ": "え", "ォ": "お",
        "ア": "あ", "イ": "い", "ウ": "う", "エ": "え", "オ": "お",
        "カ": "あ", "キ": "い", "ク": "う", "ケ": "え", "コ": "お",
        "サ": "あ", "シ": "い", "ス": "う", "セ": "え", "ソ": "お",
        "タ": "あ", "チ": "い", "ツ": "う", "テ": "え", "ト": "お",
        "ナ": "あ", "ニ": "い", "ヌ": "う", "ネ": "え", "ノ": "お",
        "ハ": "あ", "ヒ": "い", "フ": "う", "ヘ": "え", "ホ": "お",
        "マ": "あ", "ミ": "い", "ム": "う", "メ": "え", "モ": "お",
        "ヤ": "あ", "ユ": "う", "ヨ": "お",
        "ラ": "あ", "リ": "い", "ル": "う", "レ": "え", "ロ": "お",
        "ワ": "あ", "ヲ": "お", "ン": "ん",
        "ガ": "あ", "ギ": "い", "グ": "う", "ゲ": "え", "ゴ": "お",
        "ザ": "あ", "ジ": "い", "ズ": "う", "ゼ": "え", "ゾ": "お",
        "ダ": "あ", "ヂ": "い", "ヅ": "う", "デ": "え", "ド": "お",
        "バ": "あ", "ビ": "い", "ブ": "う", "ベ": "え", "ボ": "お",
        "パ": "あ", "ピ": "い", "プ": "う", "ペ": "え", "ポ": "お",
        "ャ": "あ", "ュ": "う", "ョ": "お", "ヮ": "あ"
    }
    return vowel_table.get(char, "あ")

def group_into_moras(chars):
    """文字リストをUTAUのノート単位（モーラ）にまとめる。きゃ、じゃ等の拗音に対応。"""
    small_chars = "ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ"
    moras = []
    for char in chars:
        if char in small_chars and moras:
            moras[-1] += char
        else:
            moras.append(char)
    return moras

def merge_small_chars_in_segments(segments):
    """
    隣接するセグメントのうち、後続が小書き文字（ぁぃぅぇぉゃゅょゎ等）である場合、
    前のセグメントにテキストを結合し、end時間を延長する。
    促音（っ）は発音しないが長さを持ち、特殊な結合はしないため対象外とする。
    """
    small_chars = "ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ"
    merged = []
    for seg in segments:
        text = seg["text"]
        if merged and all(c in small_chars for c in text):
            merged[-1]["text"] += text
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(dict(seg))
    return merged

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

def refine_note_boundaries_with_dtw(aligned_chars, time_array, confidence_array):
    if not aligned_chars or len(time_array) == 0:
        return aligned_chars
        
    synthetic_env = np.zeros(len(time_array))
    for seg in aligned_chars:
        s_idx = np.searchsorted(time_array, seg["start"])
        e_idx = np.searchsorted(time_array, seg["end"])
        if s_idx < len(synthetic_env):
            e_idx = min(e_idx, len(synthetic_env))
            synthetic_env[s_idx:e_idx] = 1.0
            
    try:
        import librosa
        D, wp = librosa.sequence.dtw(X=synthetic_env.reshape(1, -1), Y=confidence_array.reshape(1, -1))
        
        mapping = {}
        for x, y in wp:
            if x not in mapping:
                mapping[x] = []
            mapping[x].append(y)
            
        for x in mapping:
            mapping[x] = int(np.mean(mapping[x]))
            
        refined_chars = []
        for seg in aligned_chars:
            s_idx = np.searchsorted(time_array, seg["start"])
            e_idx = np.searchsorted(time_array, seg["end"])
            
            s_idx_new = mapping.get(s_idx, s_idx)
            e_idx_new = mapping.get(min(e_idx, len(time_array)-1), min(e_idx, len(time_array)-1))
            
            refined_chars.append({
                "text": seg["text"],
                "start": time_array[s_idx_new] if s_idx_new < len(time_array) else seg["start"],
                "end": time_array[e_idx_new] if e_idx_new < len(time_array) else seg["end"]
            })
        return refined_chars
    except Exception as e:
        print(f"DTW refinement failed: {e}")
        return aligned_chars

# ==========================================
# 2. ピッチ連続データを取得 (PyWorld / CREPE)
# ==========================================
def get_pitch_contour(audio_path, frame_period=10.0, f0_model="PyWorld", pyworld_silence_threshold=-40.0):
    print(f"{f0_model} でピッチ解析を実行中...")
    
    sr, audio = wavfile.read(audio_path)
    
    # モノラル化
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
        
    if f0_model == "CREPE":
        try:
            import torchcrepe
        except ImportError:
            raise ImportError("CREPEを使用するには torchcrepe が必要です。ターミナルで 'pip install torchcrepe' を実行してください。")
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # torchcrepe expects float32 audio tensor with shape (1, N)
        if audio.dtype == np.int16:
            audio_f32 = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio_f32 = audio.astype(np.float32) / 2147483648.0
        else:
            audio_f32 = audio.astype(np.float32)
            
        audio_tensor = torch.tensor(audio_f32).unsqueeze(0).to(device)
        
        # hop_length is calculated from frame_period (ms)
        hop_length = int(sr * (frame_period / 1000.0))
        
        # Estimate pitch
        f0, pd = torchcrepe.predict(
            audio_tensor,
            sr,
            hop_length,
            fmin=50,
            fmax=2000,
            model='full',
            batch_size=2048,
            device=device,
            return_periodicity=True
        )
        
        f0 = f0.squeeze().cpu().numpy()
        pd = pd.squeeze().cpu().numpy()
        
        # Create time array
        time = np.arange(len(f0)) * (frame_period / 1000.0)
        
        # Apply threshold to pd (periodicity/confidence) to simulate voiced/unvoiced
        confidence = (pd > 0.5).astype(float)
        f0[confidence == 0] = np.nan
        
    else: # PyWorld
        # pyworldはfloat64形式の1D numpy配列を要求する
        if audio.dtype == np.int16:
            audio = audio.astype(np.float64) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float64) / 2147483648.0
        else:
            audio = audio.astype(np.float64)
        
        # harvestでF0推定
        f0, time = pw.harvest(audio, sr, frame_period=frame_period)
        
        # 音量（RMS）を計算し、F0が検出できなくても音量が閾値以上なら有声とする
        hop_length = int(sr * (frame_period / 1000.0))
        rms = librosa.feature.rms(y=audio, frame_length=hop_length*2, hop_length=hop_length)[0]
        if len(rms) < len(f0):
            rms = np.pad(rms, (0, len(f0) - len(rms)), mode='edge')
        else:
            rms = rms[:len(f0)]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        confidence = ((f0 > 0) | (rms_db > pyworld_silence_threshold)).astype(float)
    
    # 無音や判定不能時の 0Hz または NaN に対して処理
    f0_safe = np.copy(f0)
    f0_safe[np.isnan(f0_safe)] = 0.0 # hz_to_midi is safer with non-nan, but librosa handles nan in recent versions.
    f0_safe[f0_safe == 0] = np.nan
    
    # 周波数(Hz)をMIDIノート番号（小数含む）に変換
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        midi_contour = librosa.hz_to_midi(f0_safe)
    
    # 補間処理：無音部分（NaN）のピッチが60に飛ぶと、不要なノート分割が発生するため、
    # 前後の有声音のピッチで補間（forward fill & backward fill）する。
    valid_mask = ~np.isnan(midi_contour)
    if valid_mask.any():
        # forward fill
        idx = np.where(valid_mask, np.arange(len(midi_contour)), 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        midi_contour = midi_contour[idx]
        
        # backward fill
        midi_rev = midi_contour[::-1]
        valid_mask_rev = ~np.isnan(midi_rev)
        idx_rev = np.where(valid_mask_rev, np.arange(len(midi_rev)), 0)
        np.maximum.accumulate(idx_rev, axis=0, out=idx_rev)
        midi_contour = midi_rev[idx_rev][::-1]
    else:
        midi_contour = np.full_like(midi_contour, 60.0)
    
    return time, midi_contour, confidence

# ==========================================
# DPを用いたWhisperXとWav2Vec2の歌詞マッピング
# ==========================================
def align_lyrics_to_timings(whisper_chars, w2v2_chars, skip_b_cost=0.5, last_mora_ratio=0.7):
    if not w2v2_chars:
        return []
    if not whisper_chars:
        return list(w2v2_chars)
        
    N = len(whisper_chars)
    M = len(w2v2_chars)
    
    dp = np.full((N + 1, M + 1), float('inf'))
    path = np.zeros((N + 1, M + 1), dtype=int)
    
    # 時間を完全に無視するため、コストのバランスを調整
    MATCH_REWARD = -5.0
    MISMATCH_PENALTY = 2.0
    SKIP_B_COST = skip_b_cost     # Wav2Vec2の不要な文字をスキップ（長音化）するコスト
    SQUEEZE_A_COST = 5.0  # Whisper側の文字を無理やり詰め込むコスト（基本避ける）
    
    dp[0][0] = 0.0
    for j in range(1, M + 1):
        dp[0][j] = dp[0][j-1] + SKIP_B_COST
        path[0][j] = 1 # 1: Skip B
        
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            a_mid = (whisper_chars[i-1]["start"] + whisper_chars[i-1]["end"]) / 2.0
            b_mid = (w2v2_chars[j-1]["start"] + w2v2_chars[j-1]["end"]) / 2.0
            tdiff = abs(a_mid - b_mid)
            
            # 時間はあくまで「文字が重複した時のタイブレーカー（微小なペナルティ）」としてのみ使う
            # tdiffが10秒あっても0.1のペナルティにしかならないため、マッチング報酬(-5.0)を覆すことはない
            time_penalty = tdiff * 0.01
            
            # Whisper側の文字(例: "ふぉ")の中にWav2Vec2の文字(例: "ふ"や"ぉ")が含まれていれば一致とみなす
            if w2v2_chars[j-1]["text"] in whisper_chars[i-1]["text"]:
                char_cost = MATCH_REWARD
            else:
                char_cost = MISMATCH_PENALTY
            
            cost_match = dp[i-1][j-1] + char_cost + time_penalty
            cost_skip_b = dp[i][j-1] + SKIP_B_COST
            cost_squeeze_a = dp[i-1][j] + SQUEEZE_A_COST
            
            costs = [cost_match, cost_skip_b, cost_squeeze_a]
            min_cost = min(costs)
            dp[i][j] = min_cost
            path[i][j] = costs.index(min_cost)
            
    i, j = N, M
    b_assigned = {idx: [] for idx in range(M)}
    b_skipped = [False] * M
    
    while i > 0 or j > 0:
        p = path[i][j]
        if p == 0: # Match
            i -= 1
            j -= 1
            b_assigned[j].insert(0, whisper_chars[i]["text"])
        elif p == 1: # Skip B
            j -= 1
            b_skipped[j] = True
        elif p == 2: # Squeeze A
            i -= 1
            b_assigned[j-1].insert(0, whisper_chars[i]["text"])
            
    aligned_notes = []
    for j in range(M):
        seg_template = dict(w2v2_chars[j])
        assigned_chars = b_assigned[j]
        
        if not assigned_chars:
            if b_skipped[j]:
                # 何も割り当てられなかった場合は元のWav2Vec2の結果を維持するか、スキップするか
                # 現状は元のテキスト（またはR等）を維持
                seg_template["text"] = "ー"  # 不要な文字を長音に変換
                seg_template["is_skipped"] = True
                aligned_notes.append(seg_template)
            else:
                # 原理上ここには来ないはずだが念のため
                aligned_notes.append(seg_template)
            continue
            
        # 割り当てられた文字をモーラ（ノート単位）にまとめる
        moras = group_into_moras(assigned_chars)
        
        # 複数のモーラがある場合、最後のモーラを長めにする
        total_duration = seg_template["end"] - seg_template["start"]
        if len(moras) > 1:
            other_mora_duration = (total_duration * (1 - last_mora_ratio)) / (len(moras) - 1)
            last_mora_duration = total_duration * last_mora_ratio
            
            current_offset = 0
            for k, mora in enumerate(moras):
                new_seg = dict(seg_template)
                new_seg["text"] = mora
                new_seg["start"] = seg_template["start"] + current_offset
                dur = last_mora_duration if k == len(moras) - 1 else other_mora_duration
                new_seg["end"] = new_seg["start"] + dur
                aligned_notes.append(new_seg)
                current_offset += dur
        else:
            seg_template["text"] = moras[0] if moras else ""
            aligned_notes.append(seg_template)
        
    return aligned_notes

# ==========================================
# 3. ノートの分割と文字の割り当て（ハイブリッド処理）
# ==========================================
def segment_and_align_notes(char_segments, time_array, midi_contour, confidence_array, min_duration=0.03, unvoiced_threshold_frames=10, low_pitch_threshold=47, low_pitch_drop_amount=18, pitch_split_threshold_frames=10, pitch_split_fluctuation=0.2, absorb_max_frames=10, enable_pitch_split=False):
    # print("統合タイムライン方式によるノート生成を実行中...")
    
    if len(time_array) == 0:
        return []

    # 1. タイムライン（歌詞情報）の構築
    # 10ms単位の各スロットにどの文字が割り当てられているかを埋める
    timeline_texts = [None] * len(time_array)
    
    # 開始時間でソート
    char_segments_sorted = sorted(char_segments, key=lambda x: x["start"])
    
    for i, char_info in enumerate(char_segments_sorted):
        # この文字の区間に対応するタイムラインのインデックス範囲を特定
        start_idx = np.searchsorted(time_array, char_info["start"])
        end_idx = np.searchsorted(time_array, char_info["end"])
        
        # 抽出されたノートが短すぎてPyWorldの1フレーム未満になってしまった場合でも、
        # 最低1フレームを割り当てることでノートの欠落を防ぐ
        if start_idx == end_idx and start_idx < len(time_array):
            end_idx = start_idx + 1
            
        for idx in range(start_idx, end_idx):
            if idx < len(timeline_texts):
                timeline_texts[idx] = {"text": char_info["text"], "id": i}

    # ギャップ（Wav2Vec2で抽出されなかった母音の余韻など）の補間
    # 有声音（confidence > 0）が続く限り、直前の文字を後方に延長する。
    # 無音が挟まった場合は延長をストップするため、休符を飛び越えたゴーストノートは発生しない。
    for i in range(1, len(timeline_texts)):
        if timeline_texts[i] is None and timeline_texts[i-1] is not None:
            if confidence_array[i] > 0:
                timeline_texts[i] = timeline_texts[i-1]

    # 無音（休符）の判定
    # ピッチが取れなかった（無声音/無音）区間が一定時間以上続いた場合、Rest (None) にする
    # 子音の無声化（k, s, t など）は通常短いため、短時間の無声音はそのまま歌詞を継続させる
    current_unvoiced_len = 0
    
    for i in range(len(confidence_array)):
        if confidence_array[i] == 0:
            current_unvoiced_len += 1
        else:
            if current_unvoiced_len >= unvoiced_threshold_frames:
                # 閾値以上無声音が続いた場合、その区間を None (休符) に書き換える
                for j in range(i - current_unvoiced_len, i):
                    timeline_texts[j] = None
            current_unvoiced_len = 0
            
    # 最後の部分の休符処理
    if current_unvoiced_len >= unvoiced_threshold_frames:
        for j in range(len(confidence_array) - current_unvoiced_len, len(confidence_array)):
            timeline_texts[j] = None

    if enable_pitch_split:
        # 1.5. ピッチ推移に基づく自動分割
        diff_threshold = pitch_split_fluctuation
        original_max_id = max((t["id"] for t in timeline_texts if t is not None), default=0)
        next_new_id = original_max_id + 1
        
        i = 0
        while i < len(timeline_texts):
            if timeline_texts[i] is None:
                i += 1
                continue
                
            current_id = timeline_texts[i]["id"]
            start_idx = i
            while i < len(timeline_texts) and timeline_texts[i] is not None and timeline_texts[i]["id"] == current_id:
                i += 1
            end_idx = i
            
            # [start_idx, end_idx) の区間でピッチをチェック
            pitches = midi_contour[start_idx:end_idx]
            valid_pitches = pitches[~np.isnan(pitches)]
            if len(valid_pitches) > 0:
                rounded_pitches = np.round(valid_pitches).astype(int)
                rounded_pitches = rounded_pitches[rounded_pitches >= 0]
                if len(rounded_pitches) > 0:
                    counts = np.bincount(rounded_pitches)
                    base_pitch = int(np.argmax(counts))
                    
                    # 安定セグメントを探索
                    segments = []
                    current_segment = []
                    for j, p in enumerate(pitches):
                        if np.isnan(p):
                            if current_segment:
                                segments.append(current_segment)
                                current_segment = []
                            continue
                        
                        if not current_segment:
                            current_segment.append(j)
                        else:
                            prev_p = pitches[current_segment[-1]]
                            if abs(p - prev_p) < diff_threshold:
                                current_segment.append(j)
                            else:
                                segments.append(current_segment)
                                current_segment = [j]
                    if current_segment:
                        segments.append(current_segment)
                        
                    for seg in segments:
                        if len(seg) >= pitch_split_threshold_frames:
                            seg_pitches = pitches[seg]
                            seg_median = np.median(seg_pitches)
                            if abs(seg_median - base_pitch) >= diff_threshold:
                                # 分割実行
                                for j in seg:
                                    original_text = timeline_texts[start_idx]["text"]
                                    vowel_text = get_vowel(original_text)
                                    timeline_texts[start_idx + j] = {"text": vowel_text, "id": next_new_id}
                                next_new_id += 1

        # 1.6. 短いノートの吸収（Wav2Vec2のアライメント補正）
        j = 0
        while j < len(timeline_texts):
            if timeline_texts[j] is None:
                j += 1
                continue
                
            current_id = timeline_texts[j]["id"]
            start_idx = j
            while j < len(timeline_texts) and timeline_texts[j] is not None and timeline_texts[j]["id"] == current_id:
                j += 1
            end_idx = j
            
            note_len = end_idx - start_idx
            if note_len <= absorb_max_frames:
                note_text = timeline_texts[start_idx]["text"]
                if end_idx < len(timeline_texts) and timeline_texts[end_idx] is not None and timeline_texts[end_idx]["id"] > original_max_id:
                    target_id = timeline_texts[end_idx]["id"]
                    k = end_idx
                    while k < len(timeline_texts) and timeline_texts[k] is not None and timeline_texts[k]["id"] == target_id:
                        timeline_texts[k]["text"] = note_text
                        k += 1
                    timeline_texts[start_idx] = {"text": note_text, "id": target_id}
                elif start_idx > 0 and timeline_texts[start_idx-1] is not None and timeline_texts[start_idx-1]["id"] > original_max_id:
                    target_id = timeline_texts[start_idx-1]["id"]
                    k = start_idx - 1
                    while k >= 0 and timeline_texts[k] is not None and timeline_texts[k]["id"] == target_id:
                        timeline_texts[k]["text"] = note_text
                        k -= 1
                    timeline_texts[start_idx] = {"text": note_text, "id": target_id}

    # 2. チャンク化（歌詞と丸められたピッチの両方が同じ区間を統合）
    raw_notes = []
    current_info = timeline_texts[0]
    current_midi = int(round(midi_contour[0]))
    current_start = time_array[0]
    for i in range(1, len(time_array)):
        this_info = timeline_texts[i]
        
        # ノートIDが変わるタイミングでのみ区切る（同じ文字でもIDが違えば区切る）
        if this_info != current_info:
            # 区間内のピッチの中央値を採用する
            start_frame = np.searchsorted(time_array, current_start)
            segment_pitches = midi_contour[start_frame:i]
            valid_pitches = segment_pitches[~np.isnan(segment_pitches)]
            if len(valid_pitches) > 0:
                rounded_pitches = np.round(valid_pitches).astype(int)
                rounded_pitches = rounded_pitches[rounded_pitches >= 0]
                if len(rounded_pitches) > 0:
                    counts = np.bincount(rounded_pitches)
                    note_pitch = int(np.argmax(counts))
                else:
                    note_pitch = current_midi
            else:
                note_pitch = current_midi
                
            final_pitch_curve = []
            if current_info and len(segment_pitches) > 0:
                curve = segment_pitches.copy()
                valid_mask = ~np.isnan(curve)
                if np.any(valid_mask):
                    deviations = curve[valid_mask] - note_pitch
                    max_abs_dev = np.max(np.abs(deviations))
                    if max_abs_dev > 7.0:
                        scale_factor = 7.0 / max_abs_dev
                        curve[valid_mask] = note_pitch + (deviations * scale_factor)
                final_pitch_curve = curve.tolist()
                
            raw_notes.append({
                "lyric": current_info["text"] if current_info else "R",
                "start": current_start,
                "end": time_array[i],
                "pitch": note_pitch if current_info else 60,
                "pitch_curve": final_pitch_curve
            })
            current_info = this_info
            current_midi = note_pitch # 次の区間のデフォルトピッチとして更新
            current_start = time_array[i]
            
    # 最後の区間を追加
    # 最後の区間のピッチ計算
    start_frame = np.searchsorted(time_array, current_start)
    segment_pitches = midi_contour[start_frame:]
    valid_pitches = segment_pitches[~np.isnan(segment_pitches)]
    if len(valid_pitches) > 0:
        rounded_pitches = np.round(valid_pitches).astype(int)
        rounded_pitches = rounded_pitches[rounded_pitches >= 0]
        if len(rounded_pitches) > 0:
            counts = np.bincount(rounded_pitches)
            note_pitch = int(np.argmax(counts))
        else:
            note_pitch = current_midi
    else:
        note_pitch = current_midi

    final_pitch_curve = []
    if current_info and len(segment_pitches) > 0:
        curve = segment_pitches.copy()
        valid_mask = ~np.isnan(curve)
        if np.any(valid_mask):
            deviations = curve[valid_mask] - note_pitch
            max_abs_dev = np.max(np.abs(deviations))
            if max_abs_dev > 7.0:
                scale_factor = 7.0 / max_abs_dev
                curve[valid_mask] = note_pitch + (deviations * scale_factor)
        final_pitch_curve = curve.tolist()

    raw_notes.append({
        "lyric": current_info["text"] if current_info else "R",
        "start": current_start,
        "end": time_array[-1],
        "pitch": note_pitch if current_info else 60,
        "pitch_curve": final_pitch_curve
    })

    # 短いノートの統合（ノイズ除去）は撤廃し、すべてのノートをそのまま出力する
    merged_raw_notes = raw_notes

    # 3. データの整形と「ー」の割当
    final_notes = []
    last_lyric_base = None
    
    for note in merged_raw_notes:
        final_notes.append({
            "text": note["lyric"],
            "original_text": note["lyric"],
            "start": note["start"],
            "end": note["end"],
            "pitch": note["pitch"],
            "pitch_curve": note.get("pitch_curve", [])
        })
        
    # 4. 極端な低音ノイズの削除（休符化）
    # 直前の有効なノートより一定以上低く、かつ一定以下のノートを休符（R）にする
    last_valid_pitch = None
    for note in final_notes:
        if note["text"] == "R":
            continue
            
        if last_valid_pitch is not None:
            if note["pitch"] <= low_pitch_threshold and note["pitch"] <= last_valid_pitch - low_pitch_drop_amount:
                note["text"] = "R"
                note["original_text"] = "R"
                continue
                
        last_valid_pitch = note["pitch"]
        
    return final_notes

# ==========================================
# 4. USTファイルの書き出し
# ==========================================
def export_to_ust(ust_notes, output_path, tempo=170):
    print(f"USTファイルを書き出し中: {output_path}")
    if not ust_notes:
        print("警告: 書き出せるノートがありません。")
        return

    try:
        with open(output_path, "w", encoding="shift_jis") as f:
            f.write("[#VERSION]\n")
            f.write("UST Version1.2\n")
            f.write("[#SETTING]\n")
            f.write(f"Tempo={tempo}\n")
            f.write("Tracks=1\n")
            f.write("ProjectName=O-to-Vo_Output\n")
            f.write("Mode2=True\n")
            
            ticks_per_second = (tempo * 480) / 60
            
            # 各ノートの絶対的な開始・終了Tickを計算
            for note in ust_notes:
                note["start_tick"] = int(round(note["start"] * ticks_per_second))
                note["end_tick"] = int(round(note["end"] * ticks_per_second))
                
            # 開始タイミング順にソートする（チャンク結合時の順序ブレやオーバーラップを防ぐため）
            ust_notes.sort(key=lambda x: x["start_tick"])
                
            # オーバーラップの解消（前のノートの後ろを削ることで、開始タイミングを死守する）
            for i in range(len(ust_notes) - 1):
                if ust_notes[i]["end_tick"] > ust_notes[i+1]["start_tick"]:
                    ust_notes[i]["end_tick"] = ust_notes[i+1]["start_tick"]
                    
            # 15ティック未満の非常に短いノートを除外（UTAUでの発音エラー回避のため）
            ust_notes = [note for note in ust_notes if (note["end_tick"] - note["start_tick"]) >= 15]
                    
            current_tick = 0
            prev_vowel = "あ"
            note_idx = 0

            # ノートの書き出し
            for i, note in enumerate(ust_notes):
                start_tick = note["start_tick"]
                end_tick = note["end_tick"]
                
                # 万が一の逆転を防ぐための厳密なチェック
                if start_tick < current_tick:
                    start_tick = current_tick
                
                # 休符の挿入（空き時間がある場合）
                if start_tick > current_tick:
                    rest_length = start_tick - current_tick
                    f.write(f"[#{str(note_idx).zfill(4)}]\n")
                    f.write(f"Length={rest_length}\n")
                    f.write("Lyric=R\n")
                    f.write("NoteNum=60\n")
                    f.write("PreUtterance=0\n")
                    f.write("VoiceOverlap=0\n")
                    note_idx += 1
                    current_tick = start_tick
                
                length_ticks = end_tick - start_tick
                if length_ticks <= 0:
                    continue

                f.write(f"[#{str(note_idx).zfill(4)}]\n")
                
                lyric = note["text"]
                if lyric == "ー":
                    lyric = prev_vowel
                elif lyric == "R":
                    pass
                else:
                    prev_vowel = get_vowel(lyric)

                f.write(f"Length={length_ticks}\n")
                f.write(f"Lyric={lyric}\n")
                f.write(f"NoteNum={note['pitch']}\n")
                f.write("PreUtterance=0\n")
                f.write("VoiceOverlap=0\n")
                
                # Mode2 ピッチカーブの出力
                pitch_curve = note.get("pitch_curve", [])
                if lyric != "R" and len(pitch_curve) > 0:
                    note_pitch = note['pitch']
                    pby_list = []
                    for p in pitch_curve:
                        if np.isnan(p):
                            pby_list.append("") # 無効値は空にする
                        else:
                            # PBY = 偏差(半音) * 10
                            diff = (p - note_pitch) * 10
                            pby_list.append(str(int(round(diff))))
                    
                    # 連続する空のPBYをクリーンアップするか、最初の有効値を見つける
                    while len(pby_list) > 0 and pby_list[0] == "":
                        pby_list[0] = "0"
                        
                    if len(pby_list) > 0:
                        # 計算上のノート長(ms)
                        note_length_ms = (length_ticks / ticks_per_second) * 1000
                        
                        # 現在のピッチカーブの長さ(ms)
                        current_curve_ms = 10 * max(0, len(pby_list) - 1)
                        
                        pbw_list = ["10"] * max(0, len(pby_list) - 1)
                        
                        # ピッチカーブがノート長に満たない場合、ノート終端ギリギリに最後のピッチを維持する点を追加
                        remaining_ms = note_length_ms - current_curve_ms
                        if remaining_ms > 0:
                            # 最後の有効なピッチ値を探す
                            last_y = "0"
                            for y in reversed(pby_list):
                                if y != "":
                                    last_y = y
                                    break
                            
                            pbw_list.append(str(int(round(remaining_ms))))
                            pby_list.append(last_y)

                        # PBS: 0msの位置から開始し、最初のY値を設定
                        y0 = pby_list[0] if pby_list[0] != "" else "0"
                        f.write(f"PBS=0;{y0}\n")
                        
                        # PBW
                        if len(pbw_list) > 0:
                            f.write(f"PBW={','.join(pbw_list)}\n")
                        
                        # PBY: 2番目以降の値
                        if len(pby_list) > 1:
                            f.write(f"PBY={','.join(pby_list[1:])}\n")
                
                note_idx += 1
                current_tick = end_tick
                
            f.write("[#TRACKEND]\n")
        print(f"成功: {output_path} が生成されました。")
    except Exception as e:
        print(f"UST書き出し中にエラーが発生しました: {e}")

# ==========================================
# 4.5 MusicXMLファイルの書き出し
# ==========================================
def quantize_pitch_curve_to_notes(pitch_curve, base_pitch, s_tick, e_tick, lyric_text):
    import numpy as np
    if not pitch_curve:
        return [{"pitch": base_pitch, "start_tick": s_tick, "end_tick": e_tick, "text": lyric_text}]
        
    valid_indices = [i for i, p in enumerate(pitch_curve) if not np.isnan(p)]
    if not valid_indices:
        return [{"pitch": base_pitch, "start_tick": s_tick, "end_tick": e_tick, "text": lyric_text}]
        
    # Fill NaNs
    filled_curve = []
    for i in range(len(pitch_curve)):
        if not np.isnan(pitch_curve[i]):
            filled_curve.append(pitch_curve[i])
        else:
            nearest_idx = min(valid_indices, key=lambda x: abs(x - i))
            filled_curve.append(pitch_curve[nearest_idx])
            
    rounded_pitches = [int(round(p)) for p in filled_curve]
    
    # Smooth with median filter to remove micro-vibrato (window size ~ 9 frames = 90ms)
    window_size = 9
    smoothed_pitches = []
    for i in range(len(rounded_pitches)):
        start = max(0, i - window_size // 2)
        end = min(len(rounded_pitches), i + window_size // 2 + 1)
        smoothed_pitches.append(int(round(np.median(rounded_pitches[start:end]))))
        
    # Group identical blocks
    blocks = []
    current_p = smoothed_pitches[0]
    current_start = 0
    for i in range(1, len(smoothed_pitches)):
        if smoothed_pitches[i] != current_p:
            blocks.append((current_p, current_start, i))
            current_p = smoothed_pitches[i]
            current_start = i
    blocks.append((current_p, current_start, len(smoothed_pitches)))
    
    # Merge short blocks (< 10 frames = 100ms) into previous to prevent glitchy short notes
    min_block_len = 10
    merged_blocks = []
    for b in blocks:
        if not merged_blocks:
            merged_blocks.append(b)
        else:
            if b[2] - b[1] < min_block_len:
                prev = merged_blocks[-1]
                merged_blocks[-1] = (prev[0], prev[1], b[2])
            else:
                merged_blocks.append(b)
                
    final_blocks = []
    for b in merged_blocks:
        if not final_blocks:
            final_blocks.append(b)
        else:
            if final_blocks[-1][0] == b[0]:
                prev = final_blocks[-1]
                final_blocks[-1] = (prev[0], prev[1], b[2])
            else:
                final_blocks.append(b)
                
    total_points = len(smoothed_pitches)
    tick_length = e_tick - s_tick
    
    sub_notes = []
    for i, b in enumerate(final_blocks):
        p, b_s_idx, b_e_idx = b
        chunk_s_tick = s_tick + int((b_s_idx / total_points) * tick_length)
        chunk_e_tick = s_tick + int((b_e_idx / total_points) * tick_length)
        if chunk_e_tick <= chunk_s_tick:
            continue
            
        txt = lyric_text if i == 0 else "ー"
        sub_notes.append({
            "pitch": p,
            "start_tick": chunk_s_tick,
            "end_tick": chunk_e_tick,
            "text": txt
        })
        
    for i in range(len(sub_notes) - 1):
        sub_notes[i]["end_tick"] = sub_notes[i+1]["start_tick"]
    if sub_notes:
        sub_notes[-1]["end_tick"] = e_tick
        
    return sub_notes

def export_to_musicxml(ust_notes, output_path, tempo=170):
    print(f"MusicXMLファイルを書き出し中: {output_path}")
    if not ust_notes:
        print("警告: 書き出せるノートがありません。")
        return

    divisions = 480
    ticks_per_measure = divisions * 4 # 4/4 time
    
    # 1. イベントリストの作成 (休符も含める)
    events = []
    current_tick = 0
    
    for note in ust_notes:
        start_tick = note.get("start_tick", int(round(note["start"] * ((tempo * 480)/60))))
        end_tick = note.get("end_tick", int(round(note["end"] * ((tempo * 480)/60))))
        
        if start_tick < current_tick:
            start_tick = current_tick
            
        if start_tick > current_tick:
            events.append({
                "type": "rest",
                "start": current_tick,
                "end": start_tick
            })
            
        if end_tick > start_tick:
            if note["text"] == "R":
                events.append({
                    "type": "rest",
                    "start": start_tick,
                    "end": end_tick
                })
            else:
                sub_notes = quantize_pitch_curve_to_notes(
                    note.get("pitch_curve", []), note["pitch"], start_tick, end_tick, note["text"]
                )
                for sub_note in sub_notes:
                    events.append({
                        "type": "note",
                        "start": sub_note["start_tick"],
                        "end": sub_note["end_tick"],
                        "pitch": sub_note["pitch"],
                        "text": sub_note["text"]
                    })
        current_tick = end_tick
        
    # 2. 小節ごとに分割
    measures = {}
    
    for event in events:
        start = event["start"]
        end = event["end"]
        
        while start < end:
            measure_idx = start // ticks_per_measure
            measure_start = measure_idx * ticks_per_measure
            measure_end = measure_start + ticks_per_measure
            
            chunk_end = min(end, measure_end)
            chunk_duration = chunk_end - start
            
            if measure_idx not in measures:
                measures[measure_idx] = []
                
            is_start_of_note = (start == event["start"])
            is_end_of_note = (chunk_end == event["end"])
            
            chunk = {
                "type": event["type"],
                "duration": chunk_duration,
                "is_start": is_start_of_note,
                "is_end": is_end_of_note
            }
            if event["type"] == "note":
                chunk["pitch"] = event["pitch"]
                chunk["text"] = event["text"]
                
            measures[measure_idx].append(chunk)
            start = chunk_end

    # 3. XML文字列の構築
    xml_str = []
    xml_str.append('<?xml version="1.0" encoding="UTF-8"?>')
    xml_str.append('<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">')
    xml_str.append('<score-partwise version="3.1">')
    xml_str.append('  <part-list>')
    xml_str.append('    <score-part id="P1">')
    xml_str.append('      <part-name>Vocal</part-name>')
    xml_str.append('    </score-part>')
    xml_str.append('  </part-list>')
    xml_str.append('  <part id="P1">')
    
    max_measure = max(measures.keys()) if measures else 0
    
    for m in range(max_measure + 1):
        xml_str.append(f'    <measure number="{m+1}">')
        if m == 0:
            xml_str.append('      <attributes>')
            xml_str.append(f'        <divisions>{divisions}</divisions>')
            xml_str.append('        <key><fifths>0</fifths></key>')
            xml_str.append('        <time><beats>4</beats><beat-type>4</beat-type></time>')
            xml_str.append('        <clef><sign>G</sign><line>2</line></clef>')
            xml_str.append('      </attributes>')
            
            xml_str.append('      <direction placement="above">')
            xml_str.append('        <direction-type>')
            xml_str.append('          <metronome>')
            xml_str.append('            <beat-unit>quarter</beat-unit>')
            xml_str.append(f'            <per-minute>{int(tempo)}</per-minute>')
            xml_str.append('          </metronome>')
            xml_str.append('        </direction-type>')
            xml_str.append(f'        <sound tempo="{int(tempo)}"/>')
            xml_str.append('      </direction>')
            
        measure_events = measures.get(m, [])
        current_measure_tick = 0
        for chunk in measure_events:
            xml_str.append('      <note>')
            if chunk["type"] == "rest":
                xml_str.append('        <rest/>')
                xml_str.append(f'        <duration>{chunk["duration"]}</duration>')
            else:
                midi_pitch = int(round(chunk["pitch"]))
                octave = (midi_pitch // 12) - 1
                note_idx = midi_pitch % 12
                steps = ['C', 'C', 'D', 'D', 'E', 'F', 'F', 'G', 'G', 'A', 'A', 'B']
                alters = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
                
                step = steps[note_idx]
                alter = alters[note_idx]
                
                xml_str.append('        <pitch>')
                xml_str.append(f'          <step>{step}</step>')
                if alter != 0:
                    xml_str.append(f'          <alter>{alter}</alter>')
                xml_str.append(f'          <octave>{octave}</octave>')
                xml_str.append('        </pitch>')
                xml_str.append(f'        <duration>{chunk["duration"]}</duration>')
                
                # Tie tags
                if not chunk["is_start"]:
                    xml_str.append('        <tie type="stop"/>')
                if not chunk["is_end"]:
                    xml_str.append('        <tie type="start"/>')
                    
                # Notations for ties
                if not chunk["is_start"] or not chunk["is_end"]:
                    xml_str.append('        <notations>')
                    if not chunk["is_start"]:
                        xml_str.append('          <tied type="stop"/>')
                    if not chunk["is_end"]:
                        xml_str.append('          <tied type="start"/>')
                    xml_str.append('        </notations>')

                lyric_text = chunk["text"]
                if lyric_text == "ー":
                    pass
                elif lyric_text == "R":
                    lyric_text = ""
                
                if lyric_text and chunk["is_start"]:
                    xml_str.append('        <lyric>')
                    xml_str.append('          <syllabic>single</syllabic>')
                    xml_str.append(f'          <text>{lyric_text}</text>')
                    xml_str.append('        </lyric>')
                    
            xml_str.append('      </note>')
            current_measure_tick += chunk["duration"]
            
        # 小節の残りを休符で埋める
        if current_measure_tick < ticks_per_measure:
            padding = ticks_per_measure - current_measure_tick
            xml_str.append('      <note>')
            xml_str.append('        <rest/>')
            xml_str.append(f'        <duration>{padding}</duration>')
            xml_str.append('      </note>')
            
        xml_str.append('    </measure>')
        
    xml_str.append('  </part>')
    xml_str.append('</score-partwise>')
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(xml_str))
        print(f"成功: {output_path} が生成されました（小節分割対応済み）。")
    except Exception as e:
        print(f"エラー: MusicXMLファイルの書き出しに失敗しました。\n{e}")

def write_var_len(val):
    buf = bytearray()
    buf.append(val & 0x7F)
    val >>= 7
    while val:
        buf.insert(0, (val & 0x7F) | 0x80)
        val >>= 7
    return bytes(buf)

def export_to_midi(ust_notes, output_path, tempo=170):
    print(f"MIDIファイルを書き出し中: {output_path}")
    if not ust_notes:
        print("警告: 書き出せるノートがありません。")
        return
        
    divisions = 480
    events = []
    
    for note in ust_notes:
        start_tick = note.get("start_tick", int(round(note["start"] * ((tempo * 480)/60))))
        end_tick = note.get("end_tick", int(round(note["end"] * ((tempo * 480)/60))))
        
        if end_tick > start_tick:
            if note.get("text", "") == "R":
                continue # rests are just empty space in MIDI
            
            sub_notes = quantize_pitch_curve_to_notes(
                note.get("pitch_curve", []), note["pitch"], start_tick, end_tick, ""
            )
            for sub_note in sub_notes:
                p = sub_note["pitch"]
                p = max(0, min(127, int(p)))
                events.append((sub_note["start_tick"], 'note_on', p))
                events.append((sub_note["end_tick"], 'note_off', p))
                
    mpqn = int(60000000 / tempo)
    events.append((0, 'tempo', mpqn))
    
    def sort_key(e):
        return (e[0], 0 if e[1] == 'tempo' else (1 if e[1] == 'note_off' else 2))
    events.sort(key=sort_key)
    
    track_data = bytearray()
    last_tick = 0
    for e in events:
        tick = e[0]
        delta = tick - last_tick
        track_data.extend(write_var_len(delta))
        
        if e[1] == 'tempo':
            m = e[2]
            track_data.extend(bytes([0xFF, 0x51, 0x03, (m >> 16) & 0xFF, (m >> 8) & 0xFF, m & 0xFF]))
        elif e[1] == 'note_on':
            track_data.extend(bytes([0x90, e[2], 100]))
        elif e[1] == 'note_off':
            track_data.extend(bytes([0x80, e[2], 0]))
            
        last_tick = tick
        
    track_data.extend(bytes([0x00, 0xFF, 0x2F, 0x00]))
    
    header = bytearray(b'MThd')
    header.extend(bytes([0, 0, 0, 6, 0, 0, 0, 1, (divisions >> 8) & 0xFF, divisions & 0xFF]))
    
    trk_header = bytearray(b'MTrk')
    length = len(track_data)
    trk_header.extend(bytes([(length >> 24) & 0xFF, (length >> 16) & 0xFF, (length >> 8) & 0xFF, length & 0xFF]))
    
    try:
        with open(output_path, "wb") as f:
            f.write(header)
            f.write(trk_header)
            f.write(track_data)
        print(f"成功: {output_path} が生成されました。")
    except Exception as e:
        print(f"エラー: MIDIファイルの書き出しに失敗しました。\n{e}")

def export_to_svp(ust_notes, output_path, tempo=170):
    import json
    import uuid
    import numpy as np
    
    print(f"SVPファイルを書き出し中: {output_path}")
    if not ust_notes:
        print("警告: 書き出せるノートがありません。")
        return
        
    blicks_per_quarter = 705600000
    blicks_per_second = blicks_per_quarter * tempo / 60
    
    svp_notes = []
    pitch_points = []
    
    for note in ust_notes:
        start_blick = int(round(note["start"] * blicks_per_second))
        end_blick = int(round(note["end"] * blicks_per_second))
        duration_blick = end_blick - start_blick
        
        if duration_blick <= 0:
            continue
            
        lyric = note.get("text", "a")
        if lyric == "R":
            continue
            
        base_pitch = int(note["pitch"])
        
        svp_notes.append({
            "attributes": {},
            "duration": duration_blick,
            "lyrics": lyric,
            "onset": start_blick,
            "phonemes": "",
            "pitch": base_pitch
        })
        
        pitch_curve = note.get("pitch_curve", [])
        if pitch_curve:
            # pitch_curveの補間（NaNを埋める）
            valid_indices = [i for i, p in enumerate(pitch_curve) if not np.isnan(p)]
            if valid_indices:
                filled_curve = []
                for i in range(len(pitch_curve)):
                    if not np.isnan(pitch_curve[i]):
                        filled_curve.append(pitch_curve[i])
                    else:
                        nearest_idx = min(valid_indices, key=lambda x: abs(x - i))
                        filled_curve.append(pitch_curve[nearest_idx])
                        
                tick_length = end_blick - start_blick
                total_points = len(filled_curve)
                for i, p in enumerate(filled_curve):
                    pt_blick = start_blick + int((i / total_points) * tick_length)
                    # セント単位に変換 (半音=100セント)
                    delta_cents = (p - base_pitch) * 100
                    pitch_points.extend([pt_blick, float(delta_cents)])
                    
    svp_data = {
        "version": 153,
        "time": {
            "meter": [{"denominator": 4, "index": 0, "numerator": 4}],
            "tempo": [{"bpm": float(tempo), "position": 0}]
        },
        "library": [],
        "tracks": [
            {
                "name": "O-to-Vo Export",
                "dispColor": "ff7db235",
                "dispOrder": 0,
                "renderEnabled": False,
                "mixer": {"gainDecibel": 0.0, "pan": 0.0, "mute": False, "solo": False, "display": True},
                "mainGroup": {
                    "name": "main",
                    "uuid": str(uuid.uuid4()),
                    "parameters": {
                        "pitchDelta": {
                            "mode": "cosine",
                            "points": pitch_points
                        }
                    },
                    "notes": svp_notes
                }
            }
        ]
    }
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(svp_data, f, ensure_ascii=False, separators=(',', ':'))
        print(f"成功: {output_path} が生成されました。")
    except Exception as e:
        print(f"エラー: SVPファイルの書き出しに失敗しました。\n{e}")

def export_to_vsqx(ust_notes, output_path, tempo=170):
    import xml.etree.ElementTree as ET
    import xml.dom.minidom as minidom
    import numpy as np

    print(f"VSQXファイルを書き出し中: {output_path}")
    if not ust_notes:
        print("警告: 書き出せるノートがありません。")
        return

    def get_vocaloid_phonemes(lyric):
        mapping = {
            'あ': 'a', 'い': 'i', 'う': 'M', 'え': 'e', 'お': 'o',
            'か': 'k a', 'き': "k' i", 'く': 'k M', 'け': 'k e', 'こ': 'k o',
            'さ': 's a', 'し': 'S i', 'す': 's M', 'せ': 's e', 'そ': 's o',
            'た': 't a', 'ち': 'tS i', 'つ': 'ts M', 'て': 't e', 'と': 't o',
            'な': 'n a', 'に': 'J i', 'ぬ': 'n M', 'ね': 'n e', 'の': 'n o',
            'は': 'h a', 'ひ': 'C i', 'ふ': 'p\\ M', 'へ': 'h e', 'ほ': 'h o',
            'ま': 'm a', 'み': "m' i", 'む': 'm M', 'め': 'm e', 'も': 'm o',
            'や': 'j a', 'ゆ': 'j M', 'よ': 'j o',
            'ら': '4 a', 'り': "4' i", 'る': '4 M', 'れ': '4 e', 'ろ': '4 o',
            'わ': 'w a', 'を': 'o', 'ん': 'n',
            'が': 'g a', 'ぎ': "g' i", 'ぐ': 'g M', 'げ': 'g e', 'ご': 'g o',
            'ざ': 'dz a', 'じ': 'dZ i', 'ず': 'dz M', 'ぜ': 'dz e', 'ぞ': 'dz o',
            'だ': 'd a', 'ぢ': 'dZ i', 'づ': 'dz M', 'で': 'd e', 'ど': 'd o',
            'ば': 'b a', 'び': "b' i", 'ぶ': 'b M', 'べ': 'b e', 'ぼ': 'b o',
            'ぱ': 'p a', 'ぴ': "p' i", 'ぷ': 'p M', 'ぺ': 'p e', 'ぽ': 'p o',
            'きゃ': "k' a", 'きゅ': "k' M", 'きょ': "k' o",
            'ぎゃ': "g' a", 'ぎゅ': "g' M", 'ぎょ': "g' o",
            'しゃ': 'S a', 'しゅ': 'S M', 'しょ': 'S o',
            'じゃ': 'dZ a', 'じゅ': 'dZ M', 'じょ': 'dZ o',
            'ちゃ': 'tS a', 'ちゅ': 'tS M', 'ちょ': 'tS o',
            'にゃ': 'J a', 'にゅ': 'J M', 'にょ': 'J o',
            'ひゃ': 'C a', 'ひゅ': 'C M', 'ひょ': 'C o',
            'びゃ': "b' a", 'びゅ': "b' M", 'びょ': "b' o",
            'ぴゃ': "p' a", 'ぴゅ': "p' M", 'ぴょ': "p' o",
            'みゃ': "m' a", 'みゅ': "m' M", 'みょ': "m' o",
            'りゃ': "4' a", 'りゅ': "4' M", 'りょ': "4' o",
            'ふぁ': 'p\\ a', 'ふぃ': 'p\\ i', 'ふぇ': 'p\\ e', 'ふぉ': 'p\\ o',
            'てぃ': "t' i", 'でぃ': "d' i", 'とぅ': 't M', 'どぅ': 'd M',
            'うぇ': 'w e', 'うぃ': 'w i', 'つぁ': 'ts a', 'つぃ': 'ts i', 'つぇ': 'ts e', 'つぉ': 'ts o',
            'ー': '-'
        }
        # カタカナをひらがなに変換（長音記号などはそのまま）
        hiragana = "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in lyric)
        return mapping.get(hiragana, 'a')

    def sub(parent, tag, text=None, cdata=False):
        elem = ET.SubElement(parent, tag)
        if text is not None:
            if cdata:
                # 後で置換してCDATAにするためのプレースホルダー
                elem.text = f"__CDATA_START__{text}__CDATA_END__"
            else:
                elem.text = str(text)
        return elem

    root = ET.Element("vsq4", {
        "xmlns": "http://www.yamaha.co.jp/vocaloid/schema/vsq4/",
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:schemaLocation": "http://www.yamaha.co.jp/vocaloid/schema/vsq4/ vsq4.xsd"
    })

    sub(root, "vender", "Yamaha corporation", True)
    sub(root, "version", "4.0.0.3", True)

    vVoiceTable = sub(root, "vVoiceTable")
    vVoice = sub(vVoiceTable, "vVoice")
    sub(vVoice, "bs", "0")
    sub(vVoice, "pc", "0")
    sub(vVoice, "id", "BMLTD846MLYP2MEK", True)
    sub(vVoice, "name", "VY2V3", True)
    vPrm = sub(vVoice, "vPrm")
    for prm in ["bre", "bri", "cle", "gen", "ope"]:
        sub(vPrm, prm, "0")

    mixer = sub(root, "mixer")
    masterUnit = sub(mixer, "masterUnit")
    sub(masterUnit, "oDev", "0")
    sub(masterUnit, "rLvl", "0")
    sub(masterUnit, "vol", "0")
    vsUnit = sub(mixer, "vsUnit")
    sub(vsUnit, "tNo", "0")
    sub(vsUnit, "iGin", "0")
    sub(vsUnit, "sLvl", "-898")
    sub(vsUnit, "sEnable", "0")
    sub(vsUnit, "m", "0")
    sub(vsUnit, "s", "0")
    sub(vsUnit, "pan", "64")
    sub(vsUnit, "vol", "0")
    
    monoUnit = sub(mixer, "monoUnit")
    sub(monoUnit, "iGin", "0")
    sub(monoUnit, "sLvl", "-898")
    sub(monoUnit, "sEnable", "0")
    sub(monoUnit, "m", "0")
    sub(monoUnit, "s", "0")
    sub(monoUnit, "pan", "64")
    sub(monoUnit, "vol", "0")
    
    stUnit = sub(mixer, "stUnit")
    sub(stUnit, "iGin", "0")
    sub(stUnit, "m", "0")
    sub(stUnit, "s", "0")
    sub(stUnit, "vol", "-129")

    masterTrack = sub(root, "masterTrack")
    sub(masterTrack, "seqName", "Untitled0", True)
    sub(masterTrack, "comment", "New VSQ File", True)
    sub(masterTrack, "resolution", "480")
    sub(masterTrack, "preMeasure", "1")
    
    timeSig = sub(masterTrack, "timeSig")
    sub(timeSig, "m", "0")
    sub(timeSig, "nu", "4")
    sub(timeSig, "de", "4")
    
    tempo_elem = sub(masterTrack, "tempo")
    sub(tempo_elem, "t", "0")
    sub(tempo_elem, "v", str(int(tempo * 100)))

    vsTrack = sub(root, "vsTrack")
    sub(vsTrack, "tNo", "0")
    sub(vsTrack, "name", "Track 1", True)
    sub(vsTrack, "comment", "Track", True)
    
    vsPart = sub(vsTrack, "vsPart")
    sub(vsPart, "t", "1920")
    playTime = sub(vsPart, "playTime", "0")
    sub(vsPart, "name", "O-to-Vo Part", True)
    sub(vsPart, "comment", "O-to-Vo Part", True)
    
    sPlug = sub(vsPart, "sPlug")
    sub(sPlug, "id", "ACA9C502-A04B-42b5-B2EB-5CEA36D16FCE", True)
    sub(sPlug, "name", "VOCALOID2 Compatible Style", True)
    sub(sPlug, "version", "3.0.0.1", True)
    
    pStyle = sub(vsPart, "pStyle")
    for k, v in [("accent", 50), ("bendDep", 8), ("bendLen", 0), ("decay", 50), ("fallPort", 0), ("opening", 127), ("risePort", 0)]:
        elem = sub(pStyle, "v", v)
        elem.set("id", k)
        
    singer = sub(vsPart, "singer")
    sub(singer, "t", "0")
    sub(singer, "bs", "0")
    sub(singer, "pc", "0")

    ticks_per_second = (tempo * 480) / 60.0
    max_tick = 0
    
    note_elements = []
    cc_s_events = {}
    cc_p_events = {}
    
    for note in ust_notes:
        start_tick = int(round(note["start"] * ticks_per_second))
        end_tick = int(round(note["end"] * ticks_per_second))
        dur_tick = end_tick - start_tick
        
        if dur_tick <= 0:
            continue
            
        lyric = note.get("text", "a")
        if lyric == "R":
            continue
            
        base_pitch = int(note["pitch"])
        
        note_elem = ET.Element("note")
        sub(note_elem, "t", start_tick)
        sub(note_elem, "dur", dur_tick)
        sub(note_elem, "n", base_pitch)
        sub(note_elem, "v", "64")
        sub(note_elem, "y", lyric, True)
        phoneme = get_vocaloid_phonemes(lyric)
        elem_p = sub(note_elem, "p", phoneme, True)
        elem_p.set("lock", "1")
        
        nStyle = sub(note_elem, "nStyle")
        for k, v in [("accent", 50), ("bendDep", 0), ("bendLen", 0), ("decay", 50), ("fallPort", 0), ("opening", 127), ("risePort", 0), ("vibLen", 0), ("vibType", 0)]:
            elem = sub(nStyle, "v", v)
            elem.set("id", k)
            
        note_elements.append(note_elem)
        
        max_tick = max(max_tick, end_tick)
        
        pitch_curve = note.get("pitch_curve", [])
        if pitch_curve:
            valid_indices = [i for i, p in enumerate(pitch_curve) if not np.isnan(p)]
            if valid_indices:
                filled_curve = []
                for i in range(len(pitch_curve)):
                    if not np.isnan(pitch_curve[i]):
                        filled_curve.append(pitch_curve[i])
                    else:
                        nearest_idx = min(valid_indices, key=lambda x: abs(x - i))
                        filled_curve.append(pitch_curve[nearest_idx])
                        
                max_diff = max(abs(p - base_pitch) for p in filled_curve)
                pbs_val = max(2, int(np.ceil(max_diff)))
                
                cc_s_events[start_tick] = pbs_val
                
                total_points = len(filled_curve)
                for i, p in enumerate(filled_curve):
                    pt_tick = start_tick + int((i / total_points) * dur_tick)
                    delta_semi = p - base_pitch
                    pit_val = int(round(delta_semi * (8192.0 / pbs_val)))
                    pit_val = max(-8192, min(8191, pit_val))
                    
                    cc_p_events[pt_tick] = pit_val
                
                # ノート終了時にピッチベンドをリセット（次のノートへの影響を防ぐ）
                cc_p_events[end_tick] = 0

    # 重複を排除し、時刻順かつグループごとに要素を追加
    for t in sorted(cc_s_events.keys()):
        cc_s = ET.Element("cc")
        sub(cc_s, "t", t)
        elem = sub(cc_s, "v", str(cc_s_events[t]))
        elem.set("id", "S")
        vsPart.append(cc_s)
        
    for t in sorted(cc_p_events.keys()):
        cc_p = ET.Element("cc")
        sub(cc_p, "t", t)
        elem = sub(cc_p, "v", str(cc_p_events[t]))
        elem.set("id", "P")
        vsPart.append(cc_p)

    note_elements.sort(key=lambda x: int(x.find("t").text))
    for note_elem in note_elements:
        vsPart.append(note_elem)
        
    sub(vsPart, "plane", "0")

    playTime.text = str(max_tick + 480)
    
    monoTrack = sub(root, "monoTrack")
    stTrack = sub(root, "stTrack")
    aux = sub(root, "aux")
    sub(aux, "id", "AUX_VST_HOST_CHUNK_INFO", True)
    sub(aux, "content", "VlNDSwAAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=", True)
    
    xml_str = ET.tostring(root, encoding="utf-8").decode("utf-8")
    xml_str = xml_str.replace("__CDATA_START__", "<![CDATA[")
    xml_str = xml_str.replace("__CDATA_END__", "]]>")
    
    parsed = minidom.parseString(xml_str)
    pretty_xml = parsed.toprettyxml(indent="  ")
    pretty_xml = "\n".join([line for line in pretty_xml.split("\n") if line.strip()])
    pretty_xml = pretty_xml.replace('<?xml version="1.0" ?>', '<?xml version="1.0" encoding="UTF-8" standalone="no"?>')

    try:
        with open(output_path, "w", encoding="utf-8", newline='\n') as f:
            f.write(pretty_xml)
        print(f"成功: {output_path} が生成されました。")
    except Exception as e:
        print(f"エラー: VSQXファイルの書き出しに失敗しました。\n{e}")

def export_to_ccs(ust_notes, output_path, tempo=170):
    import xml.etree.ElementTree as ET
    import xml.dom.minidom as minidom
    import numpy as np
    import math

    print(f"CCSファイルを書き出し中: {output_path}")
    if not ust_notes:
        print("警告: 書き出せるノートがありません。")
        return

    def sub(parent, tag, text=None, **kwargs):
        elem = ET.SubElement(parent, tag, **kwargs)
        if text is not None:
            elem.text = str(text)
        return elem

    root = ET.Element("Scenario", Code="7251BC4B6168E7B2992FA620BD3E1E77")
    generation = sub(root, "Generation")
    sub(generation, "Author", Version="3.2.21.2")
    tts = sub(generation, "TTS", Version="3.1.0")
    sub(tts, "Dictionary", Version="1.4.0")
    sub(tts, "SoundSources")
    svss = sub(generation, "SVSS", Version="3.0.5")
    sub(svss, "Dictionary", Version="1.0.0")
    sound_sources = sub(svss, "SoundSources")
    sub(sound_sources, "SoundSource", Version="1.0.0", Id="XSV-JPM-P", Name="O-to-Vo")

    sequence = sub(root, "Sequence", Id="")
    scene = sub(sequence, "Scene", Id="")
    units = sub(scene, "Units")
    unit = sub(units, "Unit", Version="1.0", Id="", Category="SingerSong", Group="5041941f-7111-4049-a470-31971092d202", StartTime="00:00:00", Duration="00:10:00", CastId="XSV-JPM-P", Language="Japanese")
    song = sub(unit, "Song", Version="1.02")
    tempo_elem = sub(song, "Tempo")
    sub(tempo_elem, "Sound", Clock="0", Tempo=f"{tempo:.2f}")
    beat = sub(song, "Beat")
    sub(beat, "Time", Clock="0", Beats="4", BeatType="4")
    score = sub(song, "Score")
    sub(score, "Key", Clock="0", Fifths="0", Mode="0")

    ticks_per_second = (tempo * 960) / 60.0
    frame_period = 0.005 

    param = sub(song, "Parameter")
    logf0_elem = sub(param, "LogF0")
    logf0_data = {}

    for note in ust_notes:
        start_tick = int(round(note["start"] * ticks_per_second))
        end_tick = int(round(note["end"] * ticks_per_second))
        dur_tick = end_tick - start_tick
        
        if dur_tick <= 0:
            continue
            
        lyric = note.get("text", "a")
        if lyric == "R":
            continue
            
        base_pitch = int(note["pitch"])
        pitch_octave = (base_pitch // 12) - 1
        pitch_step = base_pitch % 12
        
        hiragana = "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in lyric)
        if hiragana == "-": hiragana = "ー"
        
        sub(score, "Note", Clock=str(start_tick), PitchStep=str(pitch_step), PitchOctave=str(pitch_octave), Duration=str(dur_tick), Lyric=hiragana)
        
        pitch_curve = note.get("pitch_curve", [])
        if pitch_curve:
            valid_indices = [i for i, p in enumerate(pitch_curve) if not np.isnan(p)]
            if valid_indices:
                filled_curve = []
                for i in range(len(pitch_curve)):
                    if not np.isnan(pitch_curve[i]):
                        filled_curve.append(pitch_curve[i])
                    else:
                        nearest_idx = min(valid_indices, key=lambda x: abs(x - i))
                        filled_curve.append(pitch_curve[nearest_idx])
                        
                total_points = len(filled_curve)
                for i, p in enumerate(filled_curve):
                    time_sec = note["start"] + (i / total_points) * (note["end"] - note["start"])
                    frame_idx = int(round(time_sec / frame_period))
                    
                    f0_hz = 440.0 * (2.0 ** ((p - 69.0) / 12.0))
                    if f0_hz > 0:
                        logf0_data[frame_idx] = math.log(f0_hz)

    if logf0_data:
        max_idx = max(logf0_data.keys())
        logf0_elem.set("Length", str(max_idx + 1))
        
        sorted_indices = sorted(logf0_data.keys())
        last_idx = -2
        for idx in sorted_indices:
            val = logf0_data[idx]
            if idx == last_idx + 1:
                sub(logf0_elem, "Data", text=str(val))
            else:
                sub(logf0_elem, "Data", text=str(val), Index=str(idx))
            last_idx = idx

    groups = sub(scene, "Groups")
    sub(groups, "Group", Version="1.0", Id="5041941f-7111-4049-a470-31971092d202", Category="SingerSong", Name="vocal_hybrid", Color="#FFAF1F14", Volume="0", Pan="0", IsSolo="false", IsMuted="false", CastId="XSV-JPM-P", Language="Japanese")
    sub(scene, "SoundSetting", Rhythm="4/4", Tempo=str(int(tempo)))

    xml_str = ET.tostring(root, encoding="utf-8").decode("utf-8")
    parsed = minidom.parseString(xml_str)
    pretty_xml = parsed.toprettyxml(indent="  ")
    pretty_xml = "\n".join([line for line in pretty_xml.split("\n") if line.strip()])
    pretty_xml = pretty_xml.replace('<?xml version="1.0" ?>', '<?xml version="1.0" encoding="utf-8"?>')

    try:
        with open(output_path, "w", encoding="utf-8", newline='\n') as f:
            f.write(pretty_xml)
        print(f"成功: {output_path} が生成されました。")
    except Exception as e:
        print(f"エラー: CCSファイルの書き出しに失敗しました。\n{e}")

def estimate_tempo(audio_path, default_tempo=120):
    print("BPM(テンポ)を自動推定中...")
    try:
        y, sr = librosa.load(audio_path)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        estimated_tempo = float(np.round(tempo[0]) if isinstance(tempo, np.ndarray) else np.round(tempo))
        print(f"推定されたBPM: {estimated_tempo}")
        return estimated_tempo
    except Exception as e:
        print(f"BPMの推定に失敗しました（デフォルトの {default_tempo} を使用します）: {e}")
        return default_tempo

# ==========================================
# メイン処理（実行フロー）
# ==========================================
def run_conversion(audio_file, output_base_path, user_specified_tempo, min_duration=0.03, export_hybrid=True, export_w2v2=False, export_whisper=False,
                   unvoiced_threshold_frames=10, frame_period=10.0, low_pitch_threshold=47, low_pitch_drop_amount=18, top_db=40, skip_b_cost=0.5, last_mora_ratio=0.7,
                   whisper_model_name="large-v3", w2v2_model_name="vumichien/wav2vec2-large-xlsr-japanese-hiragana", f0_model="PyWorld",
                   output_formats=None, pyworld_silence_threshold=-40.0, pitch_split_threshold_ms=100.0, pitch_split_fluctuation=0.2, absorb_max_ms=100.0, enable_pitch_split=False):
    if output_formats is None:
        output_formats = ["ust"]
    pitch_split_threshold_frames = max(1, int(pitch_split_threshold_ms / frame_period))
    absorb_max_frames = max(1, int(absorb_max_ms / frame_period))
    # Normalize paths to avoid mixed slashes on Windows
    audio_file = os.path.normpath(audio_file)
    input_dir = os.path.dirname(audio_file)
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    
    # User specified output base path
    output_base_path = os.path.normpath(output_base_path)
    
    # 拡張子が .wav でない場合、一時的にWAVファイルに変換する
    is_temp_wav = False
    process_audio_file = audio_file
    if not audio_file.lower().endswith(".wav"):
        process_audio_file = os.path.join(input_dir, f"{base_name}_temp.wav")
        print(f"入力ファイルを読み込み、WAV形式に変換しています: {process_audio_file}")
        
        import tempfile
        import shutil
        import subprocess
        
        # FFmpeg等の一部ツールはWindowsで日本語パス（千本桜など）を正しく開けないため、
        # 一旦安全なTempフォルダにコピーしてから処理する
        temp_dir = tempfile.gettempdir()
        _, ext = os.path.splitext(audio_file)
        safe_input_path = os.path.join(temp_dir, f"oto_vo_temp_in{ext}")
        safe_output_path = os.path.join(temp_dir, "oto_vo_temp_out.wav")
        
        try:
            # 入力ファイルをTempにコピー
            shutil.copy2(audio_file, safe_input_path)
            
            # まずはFFmpegで直接変換を試みる（librosaより高速で確実）
            subprocess.run(
                ["ffmpeg", "-y", "-i", safe_input_path, "-ar", "16000", "-ac", "1", safe_output_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            
            # 成功したら元の出力先へコピー
            shutil.copy2(safe_output_path, process_audio_file)
            is_temp_wav = True
            
        except Exception:
            # FFmpegが失敗した場合、librosaでの読み込みを試みる
            try:
                y, sr_load = librosa.load(safe_input_path, sr=16000, mono=True)
                wavfile.write(process_audio_file, sr_load, y)
                is_temp_wav = True
            except Exception as e2:
                print(f"エラー: 音声ファイルの読み込みまたはWAV変換に失敗しました。\nファイルが破損しているか、対応していない形式の可能性があります。\n詳細: {e2}")
                if os.path.exists(safe_input_path): os.remove(safe_input_path)
                if os.path.exists(safe_output_path): os.remove(safe_output_path)
                return
                
        # Tempファイルのお掃除
        if os.path.exists(safe_input_path): os.remove(safe_input_path)
        if os.path.exists(safe_output_path): os.remove(safe_output_path)

    if not os.path.exists(process_audio_file):
        print(f"エラー: {process_audio_file} が見つかりません。")
        return
        
    if user_specified_tempo:
        target_tempo = user_specified_tempo
        print(f"ユーザー指定のBPMを使用します: {target_tempo}")
    else:
        target_tempo = estimate_tempo(process_audio_file)
        
    # モデルの事前ロード
    model, model_a, metadata, device = load_whisperx_models(whisper_model_name, w2v2_model_name)
    model_w2v2, processor_w2v2, device_w2v2 = load_wav2vec2_ctc_model(w2v2_model_name)
    
    sr, full_audio = wavfile.read(process_audio_file)
    # モノラル化
    if len(full_audio.shape) > 1:
        full_audio = np.mean(full_audio, axis=1).astype(full_audio.dtype)
        
    total_samples = len(full_audio)
    
    # VADによるチャンク分割 (librosa.effects.splitを使用)
    print("VADを用いて無音区間で音声を分割中...")
    if full_audio.dtype == np.int16:
        audio_float = full_audio.astype(np.float32) / 32768.0
    elif full_audio.dtype == np.int32:
        audio_float = full_audio.astype(np.float32) / 2147483648.0
    else:
        audio_float = full_audio.astype(np.float32)
        
    intervals = librosa.effects.split(audio_float, top_db=top_db, frame_length=2048, hop_length=512)
    
    padding_samples = int(0.1 * sr)
    padded_intervals = []
    for start, end in intervals:
        p_start = max(0, start - padding_samples)
        p_end = min(total_samples, end + padding_samples)
        padded_intervals.append([p_start, p_end])

    merged_intervals = []
    for interval in padded_intervals:
        if not merged_intervals:
            merged_intervals.append(interval)
        else:
            last = merged_intervals[-1]
            if interval[0] <= last[1]:
                last[1] = max(last[1], interval[1])
            else:
                merged_intervals.append(interval)

    final_chunks = merged_intervals

    all_final_notes = []
    all_final_notes_w2v2 = []  # Wav2Vec2単独出力用
    all_final_notes_whisper = [] # Whisperデバッグ出力用
    print(f"全 {len(final_chunks)} チャンクに分割しました。処理を開始します...")
    
    for i, (start_sample, end_sample) in enumerate(final_chunks):
        chunk_audio = full_audio[start_sample:end_sample]
        offset_seconds = start_sample / sr
        
        print(f"--- チャンク {i+1}/{len(final_chunks)} 処理中: {offset_seconds:.2f}s - {end_sample/sr:.2f}s ---")
        
        # 音声が短すぎる場合はスキップ
        if (end_sample - start_sample) / sr < 0.5:
            print("チャンクが短すぎるためスキップします。")
            continue
            
        temp_audio_file = os.path.join(input_dir, "temp_chunk.wav")
        wavfile.write(temp_audio_file, sr, chunk_audio)
        
        # 1. 音声認識とアライメント (WhisperXハイブリッド)
        char_segments = process_whisperx_chunk(temp_audio_file, model, model_a, metadata, device, offset_seconds)
        
        # 認識されたひらがなをコンソールに表示
        chunk_lyric = "".join([seg["text"] for seg in char_segments])
        print(f"  -> WhisperX認識結果: {chunk_lyric} (文字数: {len(char_segments)})")
        
        # 1.5. Wav2Vec2による強制アライメント (CTC Forced Alignment)
        char_segments_merged_whisper = merge_small_chars_in_segments(char_segments)
        char_segments_w2v2_aligned = compute_forced_alignment(temp_audio_file, char_segments_merged_whisper, model_w2v2, processor_w2v2, device_w2v2, offset_seconds)
        
        # 強制アライメント後、分割されてしまった小文字（ょ等）を再結合する
        char_segments_w2v2_aligned = merge_small_chars_in_segments(char_segments_w2v2_aligned)
        
        print(f"  -> Wav2Vec2 強制アライメント完了 (抽出数: {len(char_segments_w2v2_aligned)})")
        
        # 2. ピッチ推移の取得
        time_array, midi_contour, confidence_array = get_pitch_contour(temp_audio_file, frame_period=frame_period, f0_model=f0_model, pyworld_silence_threshold=pyworld_silence_threshold)
        
        # タイムアレイにオフセットを加算
        if len(time_array) > 0:
            time_array = time_array + offset_seconds
            
            # 2.5 ぼかりす的アプローチ：DTWによるF0/Power境界微調整（デバッグのため一時無効化）
            # aligned_w2v2_chars = refine_note_boundaries_with_dtw(char_segments_w2v2_aligned, time_array, confidence_array)
            aligned_w2v2_chars = char_segments_w2v2_aligned
            
            # 3. データの結合とノート化 (ハイブリッド出力用)
            # Wav2Vec2の正確なタイミングをベースに、隙間をピッチ追従の「ー」で埋める
            final_notes = segment_and_align_notes(aligned_w2v2_chars, time_array, midi_contour, confidence_array, min_duration=min_duration, unvoiced_threshold_frames=unvoiced_threshold_frames, low_pitch_threshold=low_pitch_threshold, low_pitch_drop_amount=low_pitch_drop_amount, pitch_split_threshold_frames=pitch_split_threshold_frames, pitch_split_fluctuation=pitch_split_fluctuation, absorb_max_frames=absorb_max_frames, enable_pitch_split=enable_pitch_split)
            all_final_notes.extend(final_notes)
            
            # Whisperデバッグ用: is_skipped な文字を "R" (休符) に置き換える
            aligned_whisper_chars = []
            for seg in aligned_w2v2_chars:
                new_seg = dict(seg)
                if new_seg.get("is_skipped", False):
                    new_seg["text"] = "R"
                aligned_whisper_chars.append(new_seg)
            
            final_notes_whisper = segment_and_align_notes(aligned_whisper_chars, time_array, midi_contour, confidence_array, min_duration=min_duration, unvoiced_threshold_frames=unvoiced_threshold_frames, low_pitch_threshold=low_pitch_threshold, low_pitch_drop_amount=low_pitch_drop_amount, pitch_split_threshold_frames=pitch_split_threshold_frames, pitch_split_fluctuation=pitch_split_fluctuation, absorb_max_frames=absorb_max_frames)
            all_final_notes_whisper.extend(final_notes_whisper)
            
            # Wav2Vec2単独出力用のノートリスト構築
            final_notes_w2v2 = []
            for seg in aligned_w2v2_chars:
                mid_time = (seg["start"] + seg["end"]) / 2.0
                idx = np.searchsorted(time_array, mid_time)
                if idx >= len(midi_contour):
                    idx = len(midi_contour) - 1
                pitch = int(round(midi_contour[idx])) if len(midi_contour) > 0 else 60
                final_notes_w2v2.append({
                    "text": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "pitch": pitch
                })
            all_final_notes_w2v2.extend(final_notes_w2v2)
        
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)
            
    # メモリ解放
    del model, model_a, model_w2v2, processor_w2v2
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    # 5. 各フォーマットでファイル出力
    def save_formats(notes_data, source_name):
        # Tick計算（UTAUやMusicXML共通）
        ticks_per_second = (target_tempo * 480) / 60
        for note in notes_data:
            note["start_tick"] = int(round(note["start"] * ticks_per_second))
            note["end_tick"] = int(round(note["end"] * ticks_per_second))
            
        notes_data.sort(key=lambda x: x["start_tick"])
        for i in range(len(notes_data) - 1):
            if notes_data[i]["end_tick"] > notes_data[i+1]["start_tick"]:
                notes_data[i]["end_tick"] = notes_data[i+1]["start_tick"]
        notes_data = [note for note in notes_data if (note["end_tick"] - note["start_tick"]) >= 15]

        for fmt in output_formats:
            ext = ".musicxml" if fmt == "musicxml" else ".mid" if fmt == "midi" else f".{fmt}"
            out_path = f"{output_base_path}_{source_name}{ext}"
            
            if fmt == "ust":
                export_to_ust(notes_data, out_path, tempo=target_tempo)
            elif fmt == "musicxml":
                export_to_musicxml(notes_data, out_path, tempo=target_tempo)
            elif fmt == "midi":
                export_to_midi(notes_data, out_path, tempo=target_tempo)
            elif fmt == "svp":
                export_to_svp(notes_data, out_path, tempo=target_tempo)
            elif fmt == "vsqx":
                export_to_vsqx(notes_data, out_path, tempo=target_tempo)
            elif fmt == "ccs":
                export_to_ccs(notes_data, out_path, tempo=target_tempo)
            elif fmt == "tssln":
                print(f"Warning: Tssln export is not yet implemented ({out_path})")

    if export_hybrid:
        save_formats(all_final_notes, "hybrid")
    if export_w2v2:
        save_formats(all_final_notes_w2v2, "wav2vec2")
    if export_whisper:
        save_formats(all_final_notes_whisper, "whisper")
        
    if is_temp_wav and os.path.exists(process_audio_file):
        os.remove(process_audio_file)
        
    print("すべての処理が完了しました。")

class ThreadSafeTextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.after(0, self._write, string)

    def _write(self, string):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')

    def flush(self):
        pass

class OToVoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("O-to-Vo")
        self.root.geometry("650x750")
        
        self.audio_file_path = tk.StringVar()
        self.output_base_path_var = tk.StringVar()
        self.tempo_var = tk.StringVar()
        self.min_duration_var = tk.StringVar(value="0.03")
        self.unvoiced_threshold_var = tk.StringVar(value="10")
        self.frame_period_var = tk.StringVar(value="10.0")
        self.low_pitch_threshold_var = tk.StringVar(value="47")
        self.low_pitch_drop_amount_var = tk.StringVar(value="18")
        self.top_db_var = tk.StringVar(value="40")
        self.pyworld_silence_threshold_var = tk.StringVar(value="-40.0")
        self.pitch_split_threshold_ms_var = tk.StringVar(value="100")
        self.pitch_split_fluctuation_var = tk.StringVar(value="0.2")
        self.absorb_max_ms_var = tk.StringVar(value="100")
        self.enable_pitch_split_var = tk.BooleanVar(value=False)
        self.skip_b_cost_var = tk.StringVar(value="0.5")
        self.last_mora_ratio_var = tk.StringVar(value="0.7")
        self.whisper_model_var = tk.StringVar(value="large-v3")
        self.w2v2_model_var = tk.StringVar(value="vumichien/wav2vec2-large-xlsr-japanese-hiragana")
        self.f0_model_var = tk.StringVar(value="PyWorld")
        self.export_hybrid_var = tk.BooleanVar(value=True)
        self.export_w2v2_var = tk.BooleanVar(value=False)
        self.export_whisper_var = tk.BooleanVar(value=False)
        
        self.fmt_ust_var = tk.BooleanVar(value=False)
        self.fmt_musicxml_var = tk.BooleanVar(value=False)
        self.fmt_svp_var = tk.BooleanVar(value=False)
        self.fmt_vsqx_var = tk.BooleanVar(value=False)
        self.fmt_ccs_var = tk.BooleanVar(value=False)
        self.fmt_tssln_var = tk.BooleanVar(value=False)
        self.fmt_midi_var = tk.BooleanVar(value=True)
        
        self.create_widgets()
        
        # 標準出力と標準エラー出力をテキストボックスにリダイレクト
        sys.stdout = ThreadSafeTextRedirector(self.log_text)
        sys.stderr = ThreadSafeTextRedirector(self.log_text)

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection
        file_frame = ttk.Frame(frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="音声ファイル:").pack(side=tk.LEFT)
        ttk.Entry(file_frame, textvariable=self.audio_file_path, state='readonly', width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="参照...", command=self.browse_file).pack(side=tk.LEFT)
        
        # Output selection
        output_frame = ttk.Frame(frame)
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_frame, text="出力ベースパス:").pack(side=tk.LEFT)
        ttk.Entry(output_frame, textvariable=self.output_base_path_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(output_frame, text="保存先...", command=self.browse_output).pack(side=tk.LEFT)
        
        # Model selection
        model_frame = ttk.LabelFrame(frame, text="モデル設定", padding="5")
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_frame, text="WhisperX (歌詞認識):").grid(row=0, column=0, sticky=tk.W, pady=2)
        whisper_models = ["large-v3", "large-v2", "medium", "small", "base", "tiny", "URLを指定してモデル追加"]
        whisper_cb = ttk.Combobox(model_frame, textvariable=self.whisper_model_var, values=whisper_models, state="readonly", width=55)
        whisper_cb.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        whisper_cb.bind("<<ComboboxSelected>>", self.on_model_select)
        
        ttk.Label(model_frame, text="Wav2Vec2 (タイミング):").grid(row=1, column=0, sticky=tk.W, pady=2)
        w2v2_models = ["vumichien/wav2vec2-large-xlsr-japanese-hiragana", "jonatasgrosman/wav2vec2-large-xlsr-53-japanese", "URLを指定してモデル追加"]
        w2v2_cb = ttk.Combobox(model_frame, textvariable=self.w2v2_model_var, values=w2v2_models, state="readonly", width=55)
        w2v2_cb.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        w2v2_cb.bind("<<ComboboxSelected>>", self.on_model_select)
        
        ttk.Label(model_frame, text="F0推定 (ピッチ):").grid(row=2, column=0, sticky=tk.W, pady=2)
        f0_models = ["PyWorld", "CREPE"]
        f0_cb = ttk.Combobox(model_frame, textvariable=self.f0_model_var, values=f0_models, state="readonly", width=55)
        f0_cb.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Options frame
        options_frame = ttk.Frame(frame)
        options_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(options_frame, text="BPM (空欄で自動推定):").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(options_frame, textvariable=self.tempo_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # --- 追加パラメータ（詳細設定予定） ---
        advanced_frame = ttk.LabelFrame(frame, text="詳細設定", padding="5")
        advanced_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(advanced_frame, text="休符判定フレーム数:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(advanced_frame, textvariable=self.unvoiced_threshold_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(advanced_frame, text="ピッチ解析間隔(ms):").grid(row=0, column=2, sticky=tk.W, pady=2, padx=(10,0))
        ttk.Entry(advanced_frame, textvariable=self.frame_period_var, width=10).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(advanced_frame, text="低音ノイズ削除閾値(MIDI):").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(advanced_frame, textvariable=self.low_pitch_threshold_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(advanced_frame, text="低音ノイズ落差(半音):").grid(row=1, column=2, sticky=tk.W, pady=2, padx=(10,0))
        ttk.Entry(advanced_frame, textvariable=self.low_pitch_drop_amount_var, width=10).grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(advanced_frame, text="無音分割閾値(top_db):").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(advanced_frame, textvariable=self.top_db_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(advanced_frame, text="文字スキップコスト:").grid(row=2, column=2, sticky=tk.W, pady=2, padx=(10,0))
        ttk.Entry(advanced_frame, textvariable=self.skip_b_cost_var, width=10).grid(row=2, column=3, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(advanced_frame, text="最終モーラ時間比率:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Entry(advanced_frame, textvariable=self.last_mora_ratio_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(advanced_frame, text="最小ノート長(秒):").grid(row=3, column=2, sticky=tk.W, pady=2, padx=(10,0))
        ttk.Entry(advanced_frame, textvariable=self.min_duration_var, width=10).grid(row=3, column=3, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(advanced_frame, text="PyWorld有声閾値(dB):").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Entry(advanced_frame, textvariable=self.pyworld_silence_threshold_var, width=10).grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(advanced_frame, text="ピッチ分割閾値(ms):").grid(row=4, column=2, sticky=tk.W, pady=2, padx=(10,0))
        ttk.Entry(advanced_frame, textvariable=self.pitch_split_threshold_ms_var, width=10).grid(row=4, column=3, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(advanced_frame, text="ピッチ分割変動幅(半音):").grid(row=5, column=0, sticky=tk.W, pady=2)
        ttk.Entry(advanced_frame, textvariable=self.pitch_split_fluctuation_var, width=10).grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(advanced_frame, text="吸収最大ノート長(ms):").grid(row=5, column=2, sticky=tk.W, pady=2, padx=(10,0))
        ttk.Entry(advanced_frame, textvariable=self.absorb_max_ms_var, width=10).grid(row=5, column=3, sticky=tk.W, padx=5, pady=2)
        
        ttk.Checkbutton(advanced_frame, text="ピッチ推移による自動ノート分割・結合を有効にする", variable=self.enable_pitch_split_var).grid(row=6, column=0, columnspan=4, sticky=tk.W, pady=2)
        
        # Export Source checkboxes frame
        export_frame = ttk.LabelFrame(frame, text="出力する認識データ (入力ソース)", padding="5")
        export_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(export_frame, text="Hybrid (推奨)", variable=self.export_hybrid_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(export_frame, text="Wav2Vec2", variable=self.export_w2v2_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(export_frame, text="WhisperX", variable=self.export_whisper_var).pack(side=tk.LEFT, padx=5)
        
        # Output Format checkboxes frame
        format_frame = ttk.LabelFrame(frame, text="出力フォーマット", padding="5")
        format_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(format_frame, text="MIDI", variable=self.fmt_midi_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(format_frame, text="MusicXML", variable=self.fmt_musicxml_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(format_frame, text="Vsqx", variable=self.fmt_vsqx_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(format_frame, text="UST", variable=self.fmt_ust_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(format_frame, text="Ccs", variable=self.fmt_ccs_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(format_frame, text="Svp", variable=self.fmt_svp_var).pack(side=tk.LEFT, padx=5)
        
        # Execute button
        self.start_btn = ttk.Button(frame, text="変換開始", command=self.start_conversion)
        self.start_btn.pack(pady=10)
        
        # Log Text
        ttk.Label(frame, text="ログ出力:").pack(anchor=tk.W)
        self.log_text = scrolledtext.ScrolledText(frame, height=15, state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def on_model_select(self, event):
        cb = event.widget
        if cb.get() == "URLを指定してモデル追加":
            self.ask_custom_model_url(cb)
            
    def ask_custom_model_url(self, cb):
        top = tk.Toplevel(self.root)
        top.title("カスタムモデルの追加")
        top.geometry("400x150")
        top.transient(self.root)
        top.grab_set()
        
        ttk.Label(top, text="Hugging FaceのURLを入力してください:\n(例: https://huggingface.co/author/model)").pack(pady=10)
        url_var = tk.StringVar()
        entry = ttk.Entry(top, textvariable=url_var, width=50)
        entry.pack(padx=10, pady=5)
        
        def submit():
            url = url_var.get().strip()
            import re
            match = re.search(r"huggingface\.co/([^/]+/[^/]+)", url)
            if match:
                model_name = match.group(1)
                model_name = model_name.split('/tree')[0]
                model_name = model_name.split('?')[0]
                
                values = list(cb['values'])
                values.insert(-1, model_name)
                cb['values'] = values
                cb.set(model_name)
                top.destroy()
            else:
                messagebox.showerror("エラー", "正しいHugging FaceのURLを入力してください。", parent=top)
                
        def cancel():
            cb.set(cb['values'][0])
            top.destroy()
            
        btn_frame = ttk.Frame(top)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="追加", command=submit).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="キャンセル", command=cancel).pack(side=tk.LEFT, padx=5)

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="音声ファイルを選択",
            filetypes=(("Audio/Video files", "*.wav *.mp3 *.m4a *.mp4 *.flac *.ogg"), ("All files", "*.*"))
        )
        if filename:
            self.audio_file_path.set(filename)
            base_path = os.path.splitext(filename)[0]
            self.output_base_path_var.set(base_path)

    def browse_output(self):
        current_path = self.output_base_path_var.get()
        initial_dir = os.path.dirname(current_path) if current_path else ""
        initial_file = os.path.basename(current_path) if current_path else ""
        
        filename = filedialog.asksaveasfilename(
            title="出力ベースパスを選択",
            initialdir=initial_dir,
            initialfile=initial_file,
            defaultextension="",
            filetypes=(("USTファイル", "*.ust"), ("All files", "*.*"))
        )
        if filename:
            if filename.lower().endswith(".ust"):
                filename = filename[:-4]
            self.output_base_path_var.set(filename)

    def start_conversion(self):
        audio_file = self.audio_file_path.get()
        if not audio_file or not os.path.exists(audio_file):
            messagebox.showerror("エラー", "有効な音声ファイルを選択してください。")
            return
            
        tempo_str = self.tempo_var.get().strip()
        user_tempo = None
        if tempo_str:
            try:
                user_tempo = float(tempo_str)
            except ValueError:
                messagebox.showerror("エラー", "BPMは数値を入力してください。")
                return
                
        duration_str = self.min_duration_var.get().strip()
        min_duration = 0.03
        if duration_str:
            try:
                min_duration = float(duration_str)
            except ValueError:
                messagebox.showerror("エラー", "最小ノート長は数値を入力してください。")
                return

        try:
            unvoiced_threshold_frames = int(self.unvoiced_threshold_var.get().strip())
            frame_period = float(self.frame_period_var.get().strip())
            low_pitch_threshold = int(self.low_pitch_threshold_var.get().strip())
            low_pitch_drop_amount = int(self.low_pitch_drop_amount_var.get().strip())
            top_db = float(self.top_db_var.get().strip())
            skip_b_cost = float(self.skip_b_cost_var.get().strip())
            last_mora_ratio = float(self.last_mora_ratio_var.get().strip())
            pyworld_silence_threshold = float(self.pyworld_silence_threshold_var.get().strip())
            pitch_split_threshold_ms = float(self.pitch_split_threshold_ms_var.get().strip())
            pitch_split_fluctuation = float(self.pitch_split_fluctuation_var.get().strip())
            absorb_max_ms = float(self.absorb_max_ms_var.get().strip())
            enable_pitch_split = self.enable_pitch_split_var.get()
        except ValueError:
            messagebox.showerror("エラー", "詳細設定の各項目には正しい数値を入力してください。")
            return
            
        whisper_model_name = self.whisper_model_var.get().strip()
        w2v2_model_name = self.w2v2_model_var.get().strip()
        f0_model = self.f0_model_var.get().strip()
                
        export_hybrid = self.export_hybrid_var.get()
        export_w2v2 = self.export_w2v2_var.get()
        export_whisper = self.export_whisper_var.get()
        
        if not (export_hybrid or export_w2v2 or export_whisper):
            messagebox.showerror("エラー", "少なくとも1つの出力データソースを選択してください。")
            return
            
        output_formats = []
        if self.fmt_ust_var.get(): output_formats.append("ust")
        if self.fmt_musicxml_var.get(): output_formats.append("musicxml")
        if self.fmt_svp_var.get(): output_formats.append("svp")
        if self.fmt_vsqx_var.get(): output_formats.append("vsqx")
        if self.fmt_ccs_var.get(): output_formats.append("ccs")
        if self.fmt_midi_var.get(): output_formats.append("midi")
        
        if not output_formats:
            messagebox.showerror("エラー", "少なくとも1つの出力フォーマットを選択してください。")
            return
            
        output_base = self.output_base_path_var.get().strip()
        if not output_base:
            messagebox.showerror("エラー", "出力ベースパスを指定してください。")
            return
            
        existing_files = []
        for src, is_selected in [("hybrid", export_hybrid), ("wav2vec2", export_w2v2), ("whisper", export_whisper)]:
            if is_selected:
                for fmt in output_formats:
                    ext = ".musicxml" if fmt == "musicxml" else ".mid" if fmt == "midi" else f".{fmt}"
                    filepath = f"{output_base}_{src}{ext}"
                    if os.path.exists(filepath):
                        existing_files.append(filepath)
            
        if existing_files:
            msg = "以下のファイルが既に存在します。上書きしますか？\n\n" + "\n".join(existing_files)
            if not messagebox.askyesno("上書き確認", msg):
                return
                
        self.start_btn.config(state=tk.DISABLED)
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')
        print(f"変換処理を開始します: {audio_file}")
        
        # 別スレッドで処理を実行（GUIのフリーズ防止）
        threading.Thread(target=self.run_conversion_thread, args=(audio_file, output_base, user_tempo, min_duration, export_hybrid, export_w2v2, export_whisper,
                                                                   unvoiced_threshold_frames, frame_period, low_pitch_threshold, low_pitch_drop_amount, top_db, skip_b_cost, last_mora_ratio,
                                                                   whisper_model_name, w2v2_model_name, f0_model, output_formats, pyworld_silence_threshold, pitch_split_threshold_ms, pitch_split_fluctuation, absorb_max_ms, enable_pitch_split), daemon=True).start()

    def run_conversion_thread(self, audio_file, output_base, user_tempo, min_duration, export_hybrid, export_w2v2, export_whisper,
                              unvoiced_threshold_frames, frame_period, low_pitch_threshold, low_pitch_drop_amount, top_db, skip_b_cost, last_mora_ratio,
                              whisper_model_name, w2v2_model_name, f0_model, output_formats, pyworld_silence_threshold, pitch_split_threshold_ms, pitch_split_fluctuation, absorb_max_ms, enable_pitch_split):
        try:
            run_conversion(audio_file, output_base, user_tempo, min_duration, export_hybrid, export_w2v2, export_whisper,
                           unvoiced_threshold_frames=unvoiced_threshold_frames, frame_period=frame_period,
                           low_pitch_threshold=low_pitch_threshold, low_pitch_drop_amount=low_pitch_drop_amount,
                           top_db=top_db, skip_b_cost=skip_b_cost, last_mora_ratio=last_mora_ratio,
                           whisper_model_name=whisper_model_name, w2v2_model_name=w2v2_model_name, f0_model=f0_model,
                           output_formats=output_formats, pyworld_silence_threshold=pyworld_silence_threshold, pitch_split_threshold_ms=pitch_split_threshold_ms, pitch_split_fluctuation=pitch_split_fluctuation, absorb_max_ms=absorb_max_ms, enable_pitch_split=enable_pitch_split)
        except Exception as e:
            import traceback
            print(f"\nエラーが発生しました:\n{traceback.format_exc()}")
        finally:
            self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))

if __name__ == "__main__":
    root = tk.Tk()
    app = OToVoApp(root)
    root.mainloop()