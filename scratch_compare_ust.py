import re
import jaconv

def parse_ust(file_path):
    with open(file_path, 'rb') as f:
        content_bytes = f.read()
    content = content_bytes.decode('shift_jis', errors='ignore')
    
    tempo_match = re.search(r'Tempo=([\d.]+)', content)
    tempo = float(tempo_match.group(1)) if tempo_match else 120.0
    
    notes = []
    blocks = re.findall(r'\[#(\d+)\]\r?\n(.*?)(?=\[#|\Z)', content, re.DOTALL)
    
    current_time_ticks = 0
    for num, body in blocks:
        if num in ["SETTING", "VERSION", "TRACKEND"]: continue
        
        length_match = re.search(r'Length=(\d+)', body)
        lyric_match = re.search(r'Lyric=(.*)', body)
        notenum_match = re.search(r'NoteNum=(\d+)', body)
        
        if length_match and lyric_match:
            length = int(length_match.group(1))
            lyric = lyric_match.group(1).strip()
            notenum = int(notenum_match.group(1)) if notenum_match else 60
            
            duration_sec = (length / 480.0) * (60.0 / tempo)
            start_sec = (current_time_ticks / 480.0) * (60.0 / tempo)
            
            notes.append({
                "lyric": jaconv.kata2hira(lyric),
                "start": start_sec,
                "end": start_sec + duration_sec,
                "duration": duration_sec,
                "pitch": notenum
            })
            current_time_ticks += length
            
    return notes, tempo

def compare_robust(correct, output):
    # Match lyrics by finding corresponding notes
    print(f"{'C_Idx':<5} | {'O_Idx':<5} | {'Lyric':<10} | {'T_Diff':<8} | {'Dur_Diff'}")
    print("-" * 60)
    
    c_i = 0
    o_i = 0
    
    # Find the first non-R match to anchor
    while c_i < len(correct) and correct[c_i]['lyric'] == 'r': c_i += 1
    while o_i < len(output) and output[o_i]['lyric'] == 'r': o_i += 1
    
    if c_i >= len(correct) or o_i >= len(output):
        print("No matching notes found.")
        return

    anchor_c = correct[c_i]
    anchor_o = output[o_i]
    offset = anchor_o['start'] - anchor_c['start']
    
    last_o_match = o_i
    
    matched_count = 0
    total_t_diff = 0
    
    for i in range(c_i, len(correct)):
        c_note = correct[i]
        if c_note['lyric'] == 'r': continue
        
        # Search for this lyric in output near the expected time
        best_o = -1
        min_dist = 999
        
        expected_t = c_note['start'] + offset
        
        # Look ahead in output
        for j in range(last_o_match, min(last_o_match + 20, len(output))):
            o_note = output[j]
            if o_note['lyric'] == c_note['lyric']:
                dist = abs(o_note['start'] - expected_t)
                if dist < min_dist:
                    min_dist = dist
                    best_o = j
        
        if best_o != -1 and min_dist < 2.0: # Match within 2 seconds
            o_note = output[best_o]
            t_diff = (o_note['start'] - offset) - c_note['start']
            d_diff = o_note['duration'] - c_note['duration']
            print(f"{i:<5} | {best_o:<5} | {c_note['lyric']:<10} | {t_diff:+.3f} | {d_diff:+.3f}")
            
            last_o_match = best_o + 1
            matched_count += 1
            total_t_diff += abs(t_diff)
        else:
            # print(f"{i:<5} | {'MISS':<5} | {c_note['lyric']:<10} | {'-':<8} | {'-'}")
            pass
            
    if matched_count > 0:
        print(f"\nAverage Timing Error: {total_t_diff/matched_count:.3f}s")
        print(f"Matched {matched_count} notes.")

if __name__ == "__main__":
    correct, _ = parse_ust('correct.ust')
    output, _ = parse_ust('output_hybrid.ust')
    compare_robust(correct, output)
