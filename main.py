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

# ==========================================
# 2. ピッチ連続データを取得 (PyWorld / CREPE)
# ==========================================
def get_pitch_contour(audio_path, frame_period=10.0, f0_model="PyWorld"):
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
        confidence = (f0 > 0).astype(float)
    
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
def segment_and_align_notes(char_segments, time_array, midi_contour, confidence_array, min_duration=0.03, unvoiced_threshold_frames=10, low_pitch_threshold=47, low_pitch_drop_amount=18):
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
        for idx in range(start_idx, end_idx):
            if idx < len(timeline_texts):
                timeline_texts[idx] = {"text": char_info["text"], "id": i}

    # ギャップ（Wav2Vec2で検出されなかった有声音区間）を「直前の文字」で埋める
    # これにより、ピッチが同じであればWav2Vec2の検出タイミングと結合され、1つの長いノートになる
    last_info = None
    for i in range(len(timeline_texts)):
        if timeline_texts[i] is not None:
            last_info = timeline_texts[i]
        else:
            if last_info is not None and confidence_array[i] > 0:
                timeline_texts[i] = last_info

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
                note_pitch = int(round(np.median(valid_pitches)))
            else:
                note_pitch = current_midi
                
            raw_notes.append({
                "lyric": current_info["text"] if current_info else "R",
                "start": current_start,
                "end": time_array[i],
                "pitch": note_pitch if current_info else 60,
                "pitch_curve": segment_pitches.tolist() if current_info else []
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
        note_pitch = int(round(np.median(valid_pitches)))
    else:
        note_pitch = current_midi

    raw_notes.append({
        "lyric": current_info["text"] if current_info else "R",
        "start": current_start,
        "end": time_array[-1],
        "pitch": note_pitch if current_info else 60,
        "pitch_curve": segment_pitches.tolist() if current_info else []
    })

    # 短いノートの統合（ノイズ除去）は撤廃し、すべてのノートをそのまま出力する
    merged_raw_notes = raw_notes

    # 3. データの整形と「ー」の割当
    final_notes = []
    last_lyric_base = None
    
    for note in merged_raw_notes:
        lyric = note["lyric"]
        processed_lyric = lyric
        
        # 同じ文字の連続（ピッチ変動による分割）の場合、2回目以降は「ー」にする
        if lyric != "R":
            if lyric == last_lyric_base:
                processed_lyric = "ー"
            else:
                last_lyric_base = lyric
        else:
            last_lyric_base = None
            
        final_notes.append({
            "text": processed_lyric,
            "original_text": lyric,
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
        print(f"成功: {output_path} が生成されました（重なり強制排除済み）。")
    except Exception as e:
        print(f"UST書き出し中にエラーが発生しました: {e}")

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
                   whisper_model_name="large-v3", w2v2_model_name="vumichien/wav2vec2-large-xlsr-japanese-hiragana", f0_model="PyWorld"):
    # Normalize paths to avoid mixed slashes on Windows
    audio_file = os.path.normpath(audio_file)
    input_dir = os.path.dirname(audio_file)
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    
    # User specified output base path
    output_base_path = os.path.normpath(output_base_path)
    
    output_ust = f"{output_base_path}_hybrid.ust"
    output_wav2vec2_ust = f"{output_base_path}_wav2vec2.ust"
    output_whisper_ust = f"{output_base_path}_whisper.ust"
    
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
        
        # 1.5. Wav2Vec2による純粋な文字起こしとタイミング検出 (デバッグ用)
        char_segments_w2v2 = process_wav2vec2_ctc_chunk(temp_audio_file, model_w2v2, processor_w2v2, device_w2v2, offset_seconds)
        print(f"  -> Wav2Vec2検出数: {len(char_segments_w2v2)}")
        
        # 2. ピッチ推移の取得
        time_array, midi_contour, confidence_array = get_pitch_contour(temp_audio_file, frame_period=frame_period, f0_model=f0_model)
        
        # タイムアレイにオフセットを加算
        if len(time_array) > 0:
            time_array = time_array + offset_seconds
            
            # 1.5. Wav2Vec2のアライメント結果（歌詞マッピング）を先に生成
            
            # WhisperX側の小書き文字を直前の文字とマージして1モーラ化する（例: "ふ" + "ぉ" -> "ふぉ"）
            char_segments_merged = merge_small_chars_in_segments(char_segments)
            
            # Wav2Vec2側も同様に小書き文字をマージする（「しょ」等が2つのノートに分裂するのを防ぐため）
            char_segments_w2v2_filtered = filter_short_w2v2_segments(char_segments_w2v2, min_duration=min_duration)
            char_segments_w2v2_merged = merge_small_chars_in_segments(char_segments_w2v2_filtered)
            
            aligned_w2v2_chars = align_lyrics_to_timings(char_segments_merged, char_segments_w2v2_merged, skip_b_cost=skip_b_cost, last_mora_ratio=last_mora_ratio)
            
            # 3. データの結合とノート化 (ハイブリッド出力用)
            # Wav2Vec2の正確なタイミングをベースに、隙間をピッチ追従の「ー」で埋める
            final_notes = segment_and_align_notes(aligned_w2v2_chars, time_array, midi_contour, confidence_array, min_duration=min_duration, unvoiced_threshold_frames=unvoiced_threshold_frames, low_pitch_threshold=low_pitch_threshold, low_pitch_drop_amount=low_pitch_drop_amount)
            all_final_notes.extend(final_notes)
            
            # Whisperデバッグ用: is_skipped な文字を "R" (休符) に置き換える
            aligned_whisper_chars = []
            for seg in aligned_w2v2_chars:
                new_seg = dict(seg)
                if new_seg.get("is_skipped", False):
                    new_seg["text"] = "R"
                aligned_whisper_chars.append(new_seg)
            
            final_notes_whisper = segment_and_align_notes(aligned_whisper_chars, time_array, midi_contour, confidence_array, min_duration=min_duration, unvoiced_threshold_frames=unvoiced_threshold_frames, low_pitch_threshold=low_pitch_threshold, low_pitch_drop_amount=low_pitch_drop_amount)
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
    
    # 5. USTファイルとして保存
    if export_hybrid:
        export_to_ust(all_final_notes, output_ust, tempo=target_tempo)
    if export_w2v2:
        export_to_ust(all_final_notes_w2v2, output_wav2vec2_ust, tempo=target_tempo)
    if export_whisper:
        export_to_ust(all_final_notes_whisper, output_whisper_ust, tempo=target_tempo)
        
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
        self.root.title("O-to-Vo (Audio to UST Converter)")
        self.root.geometry("650x600")
        
        self.audio_file_path = tk.StringVar()
        self.output_base_path_var = tk.StringVar()
        self.tempo_var = tk.StringVar()
        self.min_duration_var = tk.StringVar(value="0.03")
        self.unvoiced_threshold_var = tk.StringVar(value="10")
        self.frame_period_var = tk.StringVar(value="10.0")
        self.low_pitch_threshold_var = tk.StringVar(value="47")
        self.low_pitch_drop_amount_var = tk.StringVar(value="18")
        self.top_db_var = tk.StringVar(value="40")
        self.skip_b_cost_var = tk.StringVar(value="0.5")
        self.last_mora_ratio_var = tk.StringVar(value="0.7")
        self.whisper_model_var = tk.StringVar(value="large-v3")
        self.w2v2_model_var = tk.StringVar(value="vumichien/wav2vec2-large-xlsr-japanese-hiragana")
        self.f0_model_var = tk.StringVar(value="PyWorld")
        self.export_hybrid_var = tk.BooleanVar(value=True)
        self.export_w2v2_var = tk.BooleanVar(value=False)
        self.export_whisper_var = tk.BooleanVar(value=False)
        
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
        
        # Export checkboxes frame
        export_frame = ttk.LabelFrame(frame, text="出力するUSTファイル", padding="5")
        export_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(export_frame, text="Hybrid (推奨)", variable=self.export_hybrid_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(export_frame, text="Wav2Vec2", variable=self.export_w2v2_var).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(export_frame, text="WhisperX", variable=self.export_whisper_var).pack(side=tk.LEFT, padx=5)
        
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
            messagebox.showerror("エラー", "少なくとも1つの出力USTファイルを選択してください。")
            return
            
        output_base = self.output_base_path_var.get().strip()
        if not output_base:
            messagebox.showerror("エラー", "出力ベースパスを指定してください。")
            return
            
        existing_files = []
        if export_hybrid and os.path.exists(f"{output_base}_hybrid.ust"):
            existing_files.append(f"{output_base}_hybrid.ust")
        if export_w2v2 and os.path.exists(f"{output_base}_wav2vec2.ust"):
            existing_files.append(f"{output_base}_wav2vec2.ust")
        if export_whisper and os.path.exists(f"{output_base}_whisper.ust"):
            existing_files.append(f"{output_base}_whisper.ust")
            
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
                                                                   whisper_model_name, w2v2_model_name, f0_model), daemon=True).start()

    def run_conversion_thread(self, audio_file, output_base, user_tempo, min_duration, export_hybrid, export_w2v2, export_whisper,
                              unvoiced_threshold_frames, frame_period, low_pitch_threshold, low_pitch_drop_amount, top_db, skip_b_cost, last_mora_ratio,
                              whisper_model_name, w2v2_model_name, f0_model):
        try:
            run_conversion(audio_file, output_base, user_tempo, min_duration, export_hybrid, export_w2v2, export_whisper,
                           unvoiced_threshold_frames=unvoiced_threshold_frames, frame_period=frame_period,
                           low_pitch_threshold=low_pitch_threshold, low_pitch_drop_amount=low_pitch_drop_amount,
                           top_db=top_db, skip_b_cost=skip_b_cost, last_mora_ratio=last_mora_ratio,
                           whisper_model_name=whisper_model_name, w2v2_model_name=w2v2_model_name, f0_model=f0_model)
        except Exception as e:
            import traceback
            print(f"\nエラーが発生しました:\n{traceback.format_exc()}")
        finally:
            self.root.after(0, lambda: self.start_btn.config(state=tk.NORMAL))

if __name__ == "__main__":
    root = tk.Tk()
    app = OToVoApp(root)
    root.mainloop()