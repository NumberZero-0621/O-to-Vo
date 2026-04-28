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
        return list(hira_str)
    except Exception as e:
        print(f"pyopenjtalk変換エラー: {e}")
        # フォールバックとして元の文字列を文字ごとに分割して返す
        return list(text)

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

# ==========================================
# 1. Qwen3-ASRとForcedAlignerで高精度なタイムスタンプを取得
# ==========================================
def get_qwen3_lyrics_and_alignment(audio_path):
    print("Qwen3-ASR-1.7BとQwen3-ForcedAligner-0.6Bで音声認識とアライメントを実行中...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")

    try:
        from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
    except ImportError as e:
        print(f"qwen_asrライブラリのインポートに失敗しました。仮想環境の設定を確認してください: {e}")
        return []

    # モデルのロード (ASR + Forced Aligner)
    try:
        model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-1.7B", 
            trust_remote_code=True,
            device_map=device,
            forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
            forced_aligner_kwargs={"trust_remote_code": True, "device_map": device}
        )
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}")
        print("一時的な回避策として、HuggingFaceの接続やtransformersのバージョンを確認してください。")
        return []

    # 推論とアライメントの実行
    print("推論を開始します。これには少し時間がかかる場合があります。")
    res = model.transcribe(audio_path, language="Japanese", return_time_stamps=True)
    
    if not res or len(res) == 0:
        print("音声認識結果が得られませんでした。")
        return []

    transcription = res[0]
    print(f"認識されたテキスト: {transcription.text}")

    char_segments = []
    
    if not transcription.time_stamps or not hasattr(transcription.time_stamps, 'items'):
        print("警告: アライメント情報が取得できませんでした。")
        return []

    # ForcedAlignItem (text, start_time, end_time) からUST用の文字単位セグメントを構築
    for item in transcription.time_stamps.items:
        word_text = item.text
        start_t = float(item.start_time)
        end_t = float(item.end_time)
        
        # 読み仮名を取得し、文字（モーラ）単位に分割
        moras = get_word_moras(word_text)
        num_moras = len(moras)
        
        if num_moras == 0:
            continue
            
        # 1つの単語/文字の時間を、含まれるモーラ数で均等に分割する
        duration_per_mora = (end_t - start_t) / num_moras
        
        for i, mora in enumerate(moras):
            mora_start = start_t + i * duration_per_mora
            mora_end = start_t + (i + 1) * duration_per_mora
            
            char_segments.append({
                "text": mora,
                "start": mora_start,
                "end": mora_end
            })

    # メモリ解放
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

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
# 3. ノートの分割と文字の割り当て（ハイブリッド処理）
# ==========================================
def segment_and_align_notes(char_segments, time_array, midi_contour, confidence_array, min_duration=0.02):
    print("統合タイムライン方式によるノート生成を実行中...")
    
    if len(time_array) == 0:
        return []

    # 1. タイムライン（歌詞情報）の構築
    # 10ms単位の各スロットにどの文字が割り当てられているかを埋める
    timeline_texts = [None] * len(time_array)
    
    # 開始時間でソート（後の文字を優先的に上書きするようにし、整合性を保つ）
    char_segments_sorted = sorted(char_segments, key=lambda x: x["start"])
    
    # Wav2Vec2の特性上、文字が極端に短く予測され残りが空白トークンになるため、
    # 次の文字が始まるまで今の文字の長さを延長する（ギャップを埋める）
    for i in range(len(char_segments_sorted) - 1):
        char_segments_sorted[i]["end"] = char_segments_sorted[i+1]["start"]
    if len(char_segments_sorted) > 0:
        char_segments_sorted[-1]["end"] = max(char_segments_sorted[-1]["end"], char_segments_sorted[-1]["start"] + 2.0)
    
    for char_info in char_segments_sorted:
        # この文字の区間に対応するタイムラインのインデックス範囲を特定
        start_idx = np.searchsorted(time_array, char_info["start"])
        end_idx = np.searchsorted(time_array, char_info["end"])
        for idx in range(start_idx, end_idx):
            if idx < len(timeline_texts):
                timeline_texts[idx] = char_info["text"]

    # 無音（休符）の判定
    # ピッチが取れなかった（無声音/無音）区間が一定時間以上続いた場合、Rest (None) にする
    # 子音の無声化（k, s, t など）は通常短いため、短時間の無声音はそのまま歌詞を継続させる
    unvoiced_threshold_frames = 25  # 250ms
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
    current_text = timeline_texts[0]
    current_midi = int(round(midi_contour[0]))
    current_start = time_array[0]
    
    for i in range(1, len(time_array)):
        this_text = timeline_texts[i]
        this_midi = int(round(midi_contour[i]))
        
        # 歌詞が変わるか、歌詞がある区間内でピッチが変わった場合に区切る
        if this_text != current_text or (current_text is not None and this_midi != current_midi):
            raw_notes.append({
                "lyric": current_text if current_text else "R",
                "start": current_start,
                "end": time_array[i],
                "pitch": current_midi if current_text else 60
            })
            current_text = this_text
            current_midi = this_midi
            current_start = time_array[i]
            
    # 最後の区間を追加
    raw_notes.append({
        "lyric": current_text if current_text else "R",
        "start": current_start,
        "end": time_array[-1],
        "pitch": current_midi if current_text else 60
    })

    # 3. データの整形と「ー」の割当
    final_notes = []
    last_lyric_base = None
    
    for note in raw_notes:
        duration = note["end"] - note["start"]
        
        # フィルタリング:
        # 1. 歌詞が同じで、現在のノートが短すぎる場合は前のノートに統合
        # 2. 歌詞が同じで、前のノートが短すぎる（最小長さに満たない）場合も統合して長さを稼ぐ
        # ただし、歌詞が変わった場合は、たとえ短くても統合せずに新しいノートとして開始する（発音の変化を優先）
        if len(final_notes) > 0 and note["lyric"] == final_notes[-1]["original_text"]:
            last_note = final_notes[-1]
            last_duration = last_note["end"] - last_note["start"]
            if duration < min_duration or last_duration < min_duration:
                # 長い方のノートのピッチを優先する（しゃくりやビブラートの端点ではなく、本体のピッチを拾うため）
                if duration > last_duration:
                    last_note["pitch"] = note["pitch"]
                last_note["end"] = note["end"]
                continue
            
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
            "pitch": note["pitch"]
        })
        
    return final_notes

# ==========================================
# 4. USTファイルの書き出し
# ==========================================
def export_to_ust(ust_notes, output_path, tempo=120):
    print(f"USTファイルを書き出し中: {output_path}")
    if not ust_notes:
        print("警告: 書き出せるノートがありません。")
        return

    try:
        with open(output_path, "w", encoding="shift_jis") as f:
            f.write("[#SETTING]\n")
            f.write(f"Tempo={tempo}\n")
            f.write("Tracks=1\n")
            f.write("ProjectName=O-to-Vo_Output\n")
            
            ticks_per_second = (tempo * 480) / 60
            current_tick = 0
            prev_vowel = "あ"
            note_idx = 0

            # 1. 音声の冒頭が 0秒から始まっていない場合、休符を挿入
            first_note_start_tick = int(round(ust_notes[0]["start"] * ticks_per_second))
            if first_note_start_tick > 0:
                f.write(f"[#{str(note_idx).zfill(4)}]\n")
                f.write(f"Length={first_note_start_tick}\n")
                f.write("Lyric=R\n")
                f.write("NoteNum=60\n")
                f.write("PreUtterance=0\n")
                f.write("VoiceOverlap=0\n")
                note_idx += 1
                current_tick = first_note_start_tick

            # 2. ノートの書き出し
            for i, note in enumerate(ust_notes):
                # 絶対座標（ティック）を計算し、前との差分を Length とする
                end_tick = int(round(note["end"] * ticks_per_second))
                length_ticks = end_tick - current_tick
                
                # 重複や逆転を防止
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
                # 重なりを物理的に排除するため PreUtterance と VoiceOverlap を 0 に固定
                f.write("PreUtterance=0\n")
                f.write("VoiceOverlap=0\n")
                
                note_idx += 1
                current_tick = end_tick
                
            f.write("[#TRACKEND]\n")
        print(f"成功: {output_path} が生成されました（重なり強制排除済み）。")
    except Exception as e:
        print(f"UST書き出し中にエラーが発生しました: {e}")

# ==========================================
# メイン処理（実行フロー）
# ==========================================
if __name__ == "__main__":
    audio_file = "vocal.wav"
    output_ust = "output_hybrid.ust"
    
    # 1. Qwen3を用いた高精度音声認識および強制アライメント
    char_segments = get_qwen3_lyrics_and_alignment(audio_file)
    
    # 2. pyworldで10msごとのピッチ推移を取得
    time_array, midi_contour, confidence_array = get_pyworld_pitch_contour(audio_file)
    
    # 3. データの結合（文字の時間枠内でピッチが動いたら分割する）
    final_notes = segment_and_align_notes(char_segments, time_array, midi_contour, confidence_array)
    
    # 4. 確認表示
    # for note in final_notes:
    #     print(f"[{note['start']:.2f}s - {note['end']:.2f}s] {note['text']} (MIDI: {note['pitch']})")
    
    # 5. USTファイルとして保存
    export_to_ust(final_notes, output_ust)
    print("すべての処理が完了しました。")