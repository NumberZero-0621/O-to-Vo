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
                "lyric": lyric,
                "start": start_sec,
                "end": start_sec + duration_sec,
                "duration": duration_sec,
                "pitch": notenum
            })
            current_time_ticks += length
            
    return notes, tempo

def compare_sequences(correct, output):
    # Try to align sequences by lyric
    # Use a simple matching to find where the output starts compared to correct
    
    print(f"{'Idx':<4} | {'Correct':<10} | {'Output':<10} | {'T_Start':<8} | {'T_End':<8} | {'Dur':<8} | {'Pitch'}")
    print("-" * 80)
    
    c_i = 0
    o_i = 0
    
    # Skip initial rests
    while c_i < len(correct) and correct[c_i]['lyric'] == 'R': c_i += 1
    while o_i < len(output) and output[o_i]['lyric'] == 'R': o_i += 1
    
    offset = output[o_i]['start'] - correct[c_i]['start']
    
    for _ in range(100):
        if c_i >= len(correct) or o_i >= len(output): break
        
        c = correct[c_i]
        o = output[o_i]
        
        c_lyric = c['lyric']
        o_lyric = o['lyric']
        
        # Check if lyrics match (roughly)
        # Note: correct might have Katakana, output is Hiragana
        c_hira = jaconv.kata2hira(c_lyric)
        o_hira = jaconv.kata2hira(o_lyric)
        
        s_diff = (o['start'] - offset) - c['start']
        e_diff = (o['end'] - offset) - c['end']
        d_diff = o['duration'] - c['duration']
        p_diff = o['pitch'] - c['pitch']
        
        match_str = "OK" if c_hira == o_hira else "!!"
        
        print(f"{c_i:<4} | {c_lyric:<10} | {o_lyric:<10} | {s_diff:+.3f} | {e_diff:+.3f} | {d_diff:+.3f} | {p_diff:+d} {match_str}")
        
        # Advance logic: if they don't match, try to see if one is split or merged
        if c_hira == o_hira:
            c_i += 1
            o_i += 1
        else:
            # If output is 'R' but correct is not, output might have missed a note
            if o_lyric == 'R':
                o_i += 1
            elif c_lyric == 'R':
                c_i += 1
            else:
                # Just advance both for now
                c_i += 1
                o_i += 1

if __name__ == "__main__":
    correct, _ = parse_ust('correct.ust')
    output, _ = parse_ust('output_hybrid.ust')
    compare_sequences(correct, output)
