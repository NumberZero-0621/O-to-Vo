import os
import numpy as np
from core.text_utils import get_vowel
from core.alignment import segment_and_align_notes

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
                if "volume" in note:
                    intensity = max(0, min(200, int(note["volume"] * 200)))
                    f.write(f"Intensity={intensity}\n")
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
                        "text": sub_note["text"],
                        "volume": note.get("volume", 0.5)
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
                chunk["volume"] = event.get("volume", 0.5)
                
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
            if chunk["type"] == "note" and chunk["is_start"] and "volume" in chunk:
                dyn_val = max(0, min(140, int(chunk["volume"] * 140)))
                xml_str.append('      <direction placement="below">')
                xml_str.append(f'        <sound dynamics="{dyn_val}"/>')
                xml_str.append('      </direction>')
                
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
            note_vol = note.get("volume", 0.787)
            velocity = max(1, min(127, int(note_vol * 127)))
            
            for sub_note in sub_notes:
                p = sub_note["pitch"]
                p = max(0, min(127, int(p)))
                events.append((sub_note["start_tick"], 'note_on', p, velocity))
                events.append((sub_note["end_tick"], 'note_off', p, 0))
                
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
            track_data.extend(bytes([0x90, e[2], e[3]]))
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
    loudness_points = []
    
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
                    
        volume_curve = note.get("volume_curve", [])
        if len(volume_curve) > 0:
            total_vol_points = len(volume_curve)
            for i, v in enumerate(volume_curve):
                pt_blick = start_blick + int((i / total_vol_points) * duration_blick)
                # Map 0.0~1.0 to -12.0~12.0 dB
                loudness_points.extend([pt_blick, float((v * 24.0) - 12.0)])
                    
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
                        },
                        "loudness": {
                            "mode": "linear",
                            "points": loudness_points
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
    
    part_name = os.path.splitext(os.path.basename(output_path))[0] or "O-to-Vo Part"
    vsPart = sub(vsTrack, "vsPart")
    sub(vsPart, "t", "1920")
    playTime = sub(vsPart, "playTime", "0")
    sub(vsPart, "name", part_name, True)
    sub(vsPart, "comment", part_name, True)
    
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
    cc_dyn_events = {}
    
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
            
        volume_curve = note.get("volume_curve", [])
        if len(volume_curve) > 0:
            total_vol_points = len(volume_curve)
            for i, v in enumerate(volume_curve):
                pt_tick = start_tick + int((i / total_vol_points) * dur_tick)
                dyn_val = max(0, min(127, int(v * 127)))
                cc_dyn_events[pt_tick] = dyn_val

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
        
    for t in sorted(cc_dyn_events.keys()):
        cc_d = ET.Element("cc")
        sub(cc_d, "t", t)
        elem = sub(cc_d, "v", str(cc_dyn_events[t]))
        elem.set("id", "D")
        vsPart.append(cc_d)

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
    vol_elem = sub(param, "VOL")
    vol_data = {}

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
                        
        volume_curve = note.get("volume_curve", [])
        if len(volume_curve) > 0:
            total_vol_points = len(volume_curve)
            for i, v in enumerate(volume_curve):
                time_sec = note["start"] + (i / total_vol_points) * (note["end"] - note["start"])
                frame_idx = int(round(time_sec / frame_period))
                # Map 0.0~1.0 to -127~127
                vol_data[frame_idx] = int(max(-127, min(127, (v * 254) - 127)))

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

    if vol_data:
        max_idx = max(vol_data.keys())
        vol_elem.set("Length", str(max_idx + 1))
        
        sorted_indices = sorted(vol_data.keys())
        last_idx = -2
        for idx in sorted_indices:
            val = vol_data[idx]
            if idx == last_idx + 1:
                sub(vol_elem, "Data", text=str(val))
            else:
                sub(vol_elem, "Data", text=str(val), Index=str(idx))
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
def consume_lyrics(whisper_text, remaining_lyrics):
    if not whisper_text:
        return "", remaining_lyrics
    if not remaining_lyrics:
        return "", ""
        
    N = len(whisper_text)
    M = min(len(remaining_lyrics), N * 2 + 10)
    
    dp = np.full((N + 1, M + 1), float('inf'))
    dp[0, 0] = 0
    for j in range(1, M + 1):
        dp[0, j] = j * 0.1
        
    for i in range(1, N + 1):
        dp[i, 0] = i
        for j in range(1, M + 1):
            cost = 0 if whisper_text[i-1] == remaining_lyrics[j-1] else 1
            dp[i, j] = min(dp[i-1, j-1] + cost, dp[i-1, j] + 1, dp[i, j-1] + 1)
            
    best_j = np.argmin(dp[N, :])
    if best_j == 0:
        best_j = min(N, len(remaining_lyrics))
        
    consumed = remaining_lyrics[:best_j]
    return consumed, remaining_lyrics[best_j:]

def run_conversion(audio_file, output_base_path, user_specified_tempo, min_duration=0.03, export_hybrid=True, export_w2v2=False, export_whisper=False,
                   unvoiced_threshold_frames=10, frame_period=10.0, low_pitch_threshold=47, low_pitch_drop_amount=18, top_db=40, skip_b_cost=0.5, last_mora_ratio=0.7,
                   whisper_model_name="large-v3", w2v2_model_name="vumichien/wav2vec2-large-xlsr-japanese-hiragana", f0_model="PyWorld",
                   output_formats=None, pyworld_silence_threshold=-40.0, pitch_split_threshold_ms=100.0, pitch_split_fluctuation=0.2, absorb_max_ms=100.0, enable_pitch_split=False,
                   predefined_lyrics=None, convert_to_vocaloid=False, extract_vocals=False, transpose=0, reflect_dynamics=False, dynamics_sensitivity=100):
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
    
    print(f"変換処理を開始します: {process_audio_file}")
    
    if extract_vocals:
        print("Demucsを用いたボーカル抽出を開始します...")
        try:
            import subprocess
            demucs_out_dir = os.path.join(os.path.dirname(output_base_path), "demucs_out")
            cmd = [sys.executable, "-m", "demucs.separate", "-n", "htdemucs", "--two-stems=vocals", process_audio_file, "-o", demucs_out_dir]
            subprocess.run(cmd, check=True)
            
            input_basename = os.path.splitext(os.path.basename(process_audio_file))[0]
            vocals_path = os.path.join(demucs_out_dir, "htdemucs", input_basename, "vocals.wav")
            if os.path.exists(vocals_path):
                process_audio_file = vocals_path
                print(f"ボーカル抽出完了: {process_audio_file}")
            else:
                print("抽出されたボーカルファイルが見つかりません。元の音声で続行します。")
        except Exception as e:
            print(f"ボーカル抽出に失敗しました: {e}。元の音声で続行します。")
    
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
    
    global_vol_times, global_vol_values = None, None
    if reflect_dynamics:
        print("音声全般のダイナミクス（音量推移）を解析中...")
        global_vol_times, global_vol_values = compute_global_dynamics(process_audio_file, sensitivity=dynamics_sensitivity, min_volume_ratio=0.2)
    
    remaining_lyrics = None
    if predefined_lyrics is not None:
        if convert_to_vocaloid:
            # ボカロ語に変換（pyopenjtalkでひらがな化＋は->わ等）
            lyrics_chars = get_word_moras(predefined_lyrics)
            remaining_lyrics = "".join(lyrics_chars)
        else:
            # 律儀にそのまま（ただし不要な記号や空白は除去）
            exclude_chars = "’、っ。！?？（）() 　.,'\"-\n\r\t "
            remaining_lyrics = "".join([c for c in predefined_lyrics if c not in exclude_chars])
            
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
        
        if remaining_lyrics is not None:
            whisper_text = "".join([seg["text"] for seg in char_segments])
            consumed_text, remaining_lyrics = consume_lyrics(whisper_text, remaining_lyrics)
            if char_segments and consumed_text:
                chunk_start = char_segments[0]["start"]
                chunk_end = char_segments[-1]["end"]
                dur = (chunk_end - chunk_start) / len(consumed_text)
                new_char_segments = []
                for idx, c in enumerate(consumed_text):
                    new_char_segments.append({"text": c, "start": chunk_start + idx*dur, "end": chunk_start + (idx+1)*dur})
                char_segments = new_char_segments
            else:
                char_segments = []
                
        # 認識されたひらがなをコンソールに表示
        chunk_lyric = "".join([seg["text"] for seg in char_segments])
        print(f"  -> 認識・アサイン結果: {chunk_lyric} (文字数: {len(char_segments)})")
        
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
            
            # 2.6 ダイナミクスの取得
            volume_contour = None
            if reflect_dynamics and global_vol_times is not None and len(global_vol_times) > 0:
                volume_contour = np.interp(time_array, global_vol_times, global_vol_values)
                
            # 3. データの結合とノート化 (ハイブリッド出力用)
            # Wav2Vec2の正確なタイミングをベースに、隙間をピッチ追従の「ー」で埋める
            final_notes = segment_and_align_notes(aligned_w2v2_chars, time_array, midi_contour, confidence_array, min_duration=min_duration, unvoiced_threshold_frames=unvoiced_threshold_frames, low_pitch_threshold=low_pitch_threshold, low_pitch_drop_amount=low_pitch_drop_amount, pitch_split_threshold_frames=pitch_split_threshold_frames, pitch_split_fluctuation=pitch_split_fluctuation, absorb_max_frames=absorb_max_frames, enable_pitch_split=enable_pitch_split, volume_contour=volume_contour)
            all_final_notes.extend(final_notes)
            
            # Whisperデバッグ用: is_skipped な文字を "R" (休符) に置き換える
            aligned_whisper_chars = []
            for seg in aligned_w2v2_chars:
                new_seg = dict(seg)
                if new_seg.get("is_skipped", False):
                    new_seg["text"] = "R"
                aligned_whisper_chars.append(new_seg)
            
            final_notes_whisper = segment_and_align_notes(aligned_whisper_chars, time_array, midi_contour, confidence_array, min_duration=min_duration, unvoiced_threshold_frames=unvoiced_threshold_frames, low_pitch_threshold=low_pitch_threshold, low_pitch_drop_amount=low_pitch_drop_amount, pitch_split_threshold_frames=pitch_split_threshold_frames, pitch_split_fluctuation=pitch_split_fluctuation, absorb_max_frames=absorb_max_frames, volume_contour=volume_contour)
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
        if transpose != 0:
            for note in notes_data:
                note["pitch"] += transpose
                if note.get("pitch_curve"):
                    new_curve = []
                    for p in note["pitch_curve"]:
                        if p is None or (isinstance(p, (int, float)) and np.isnan(p)):
                            new_curve.append(p)
                        else:
                            new_curve.append(p + transpose)
                    note["pitch_curve"] = new_curve

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

