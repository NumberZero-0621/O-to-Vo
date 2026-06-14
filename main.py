import os
import warnings

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
def load_whisperx_models():
    print("WhisperXモデルをロード中...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"使用デバイス: {device} ({compute_type})")

    model = whisperx.load_model("large-v3", device, compute_type=compute_type, language="ja")
    model_a, metadata = whisperx.load_align_model(language_code="ja", device=device, model_name="vumichien/wav2vec2-large-xlsr-japanese-hiragana")
    
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
def load_wav2vec2_ctc_model():
    print("Wav2Vec2(CTC)モデルをロード中...")
    model_name = "vumichien/wav2vec2-large-xlsr-japanese-hiragana"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
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
# 2. pyworldでピッチ連続データを取得
# ==========================================
def get_pyworld_pitch_contour(audio_path):
    print("pyworld (harvest) でピッチ解析を実行中...")
    
    # pyworldはfloat64形式の1D numpy配列を要求する
    sr, audio = wavfile.read(audio_path)
    
    # モノラル化
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
        
    # float64に変換し、必要に応じて正規化
    if audio.dtype == np.int16:
        audio = audio.astype(np.float64) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float64) / 2147483648.0
    else:
        audio = audio.astype(np.float64)
    
    # harvestでF0推定 (フレーム周期10ms)
    f0, time = pw.harvest(audio, sr, frame_period=10.0)
    
    # 無音や判定不能時の 0Hz を NaN にして hz_to_midi の警告を回避
    f0_safe = np.copy(f0)
    f0_safe[f0_safe == 0] = np.nan
    
    # 周波数(Hz)をMIDIノート番号（小数含む）に変換
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
    
    # 有声音フラグを返す（無音・休符の判定に使用）
    confidence = (f0 > 0).astype(float)
    
    return time, midi_contour, confidence

# ==========================================
# DPを用いたWhisperXとWav2Vec2の歌詞マッピング
# ==========================================
def align_lyrics_to_timings(whisper_chars, w2v2_chars):
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
    SKIP_B_COST = 0.5     # Wav2Vec2の不要な文字をスキップ（長音化）するコスト
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
        
        # 複数のモーラがある場合、最後のモーラを長めにする（例: 70%を最後に、残りを均等に）
        total_duration = seg_template["end"] - seg_template["start"]
        if len(moras) > 1:
            last_mora_ratio = 0.7
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
def segment_and_align_notes(char_segments, time_array, midi_contour, confidence_array, min_duration=0.03):
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
    unvoiced_threshold_frames = 10  # 100ms (速いパッセージに対応するため短縮)
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
    # 直前の有効なノートより1.5オクターブ（18半音）以上低く、かつB2（47）以下のノートを休符（R）にする
    last_valid_pitch = None
    for note in final_notes:
        if note["text"] == "R":
            continue
            
        if last_valid_pitch is not None:
            if note["pitch"] <= 47 and note["pitch"] <= last_valid_pitch - 18:
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
            
            # 各ノートの絶対的な開始・終了Tickを先に計算
            for note in ust_notes:
                note["start_tick"] = int(round(note["start"] * ticks_per_second))
                note["end_tick"] = int(round(note["end"] * ticks_per_second))
                
            # オーバーラップの解消（前のノートの後ろを削ることで、開始タイミングを死守する）
            for i in range(len(ust_notes) - 1):
                if ust_notes[i]["end_tick"] > ust_notes[i+1]["start_tick"]:
                    ust_notes[i]["end_tick"] = ust_notes[i+1]["start_tick"]
                    
            current_tick = 0
            prev_vowel = "あ"
            note_idx = 0

            # ノートの書き出し
            for i, note in enumerate(ust_notes):
                start_tick = note["start_tick"]
                end_tick = note["end_tick"]
                
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
if __name__ == "__main__":
    audio_file = "vocal.wav"
    output_ust = "output_hybrid.ust"
    output_wav2vec2_ust = "output_wav2vec2.ust"  # デバッグ用出力
    output_whisper_ust = "output_whisper.ust"    # Whisperデバッグ用出力
    
    # ユーザー手動設定用の変数（None の場合は自動推定を行う）
    # GUI化する際は、ここの値をUIから受け取るようにします
    user_specified_tempo = None 
    
    if not os.path.exists(audio_file):
        print(f"エラー: {audio_file} が見つかりません。")
        exit(1)
        
    if user_specified_tempo:
        target_tempo = user_specified_tempo
        print(f"ユーザー指定のBPMを使用します: {target_tempo}")
    else:
        target_tempo = estimate_tempo(audio_file)
        
    # モデルの事前ロード
    model, model_a, metadata, device = load_whisperx_models()
    model_w2v2, processor_w2v2, device_w2v2 = load_wav2vec2_ctc_model()
    
    sr, full_audio = wavfile.read(audio_file)
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
        
    intervals = librosa.effects.split(audio_float, top_db=40, frame_length=2048, hop_length=512)
    
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
            
        temp_audio_file = "temp_chunk.wav"
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
        time_array, midi_contour, confidence_array = get_pyworld_pitch_contour(temp_audio_file)
        
        # タイムアレイにオフセットを加算
        if len(time_array) > 0:
            time_array = time_array + offset_seconds
            
            # 1.5. Wav2Vec2のアライメント結果（歌詞マッピング）を先に生成
            
            # WhisperX側のみ、小書き文字を直前の文字とマージして1モーラ化する（例: "ふ" + "ぉ" -> "ふぉ"）
            char_segments_merged = merge_small_chars_in_segments(char_segments)
            
            # Wav2Vec2側はマージせず、極端に短いノート（30ms未満など）のみをノイズとして削除する
            char_segments_w2v2_filtered = filter_short_w2v2_segments(char_segments_w2v2, min_duration=0.03)
            
            aligned_w2v2_chars = align_lyrics_to_timings(char_segments_merged, char_segments_w2v2_filtered)
            
            # 3. データの結合とノート化 (ハイブリッド出力用)
            # Wav2Vec2の正確なタイミングをベースに、隙間をピッチ追従の「ー」で埋める
            final_notes = segment_and_align_notes(aligned_w2v2_chars, time_array, midi_contour, confidence_array)
            all_final_notes.extend(final_notes)
            
            # Whisperデバッグ用: is_skipped な文字を "R" (休符) に置き換える
            aligned_whisper_chars = []
            for seg in aligned_w2v2_chars:
                new_seg = dict(seg)
                if new_seg.get("is_skipped", False):
                    new_seg["text"] = "R"
                aligned_whisper_chars.append(new_seg)
            
            final_notes_whisper = segment_and_align_notes(aligned_whisper_chars, time_array, midi_contour, confidence_array)
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
    export_to_ust(all_final_notes, output_ust, tempo=target_tempo)
    export_to_ust(all_final_notes_w2v2, output_wav2vec2_ust, tempo=target_tempo)
    export_to_ust(all_final_notes_whisper, output_whisper_ust, tempo=target_tempo)
    print("すべての処理が完了しました。")