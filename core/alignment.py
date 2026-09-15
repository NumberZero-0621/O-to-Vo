import numpy as np
from core.text_utils import group_into_moras, get_vowel

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
def segment_and_align_notes(char_segments, time_array, midi_contour, confidence_array, min_duration=0.03, unvoiced_threshold_frames=10, low_pitch_threshold=47, low_pitch_drop_amount=18, pitch_split_threshold_frames=10, pitch_split_fluctuation=0.2, absorb_max_frames=10, enable_pitch_split=False, volume_contour=None):
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
            final_volume_curve = []
            note_volume = 0.5
            
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
                
                if volume_contour is not None:
                    segment_volumes = volume_contour[start_frame:i]
                    final_volume_curve = segment_volumes.tolist()
                    if len(segment_volumes) > 0:
                        note_volume = float(np.mean(segment_volumes))
                
            raw_notes.append({
                "lyric": current_info["text"] if current_info else "R",
                "start": current_start,
                "end": time_array[i],
                "pitch": note_pitch if current_info else 60,
                "pitch_curve": final_pitch_curve,
                "volume": note_volume,
                "volume_curve": final_volume_curve
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
    final_volume_curve = []
    note_volume = 0.5
    
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
        
        if volume_contour is not None:
            segment_volumes = volume_contour[start_frame:]
            final_volume_curve = segment_volumes.tolist()
            if len(segment_volumes) > 0:
                note_volume = float(np.mean(segment_volumes))

    raw_notes.append({
        "lyric": current_info["text"] if current_info else "R",
        "start": current_start,
        "end": time_array[-1],
        "pitch": note_pitch if current_info else 60,
        "pitch_curve": final_pitch_curve,
        "volume": note_volume,
        "volume_curve": final_volume_curve
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
            "pitch_curve": note.get("pitch_curve", []),
            "volume": note.get("volume", 0.5),
            "volume_curve": note.get("volume_curve", [])
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

