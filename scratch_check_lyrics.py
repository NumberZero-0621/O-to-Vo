import re
import jaconv

def get_lyrics(file_path):
    with open(file_path, 'rb') as f:
        content = f.read().decode('shift_jis', errors='ignore')
    lyrics = re.findall(r'Lyric=(.*)', content)
    return [jaconv.kata2hira(l.strip()) for l in lyrics if l.strip() != 'R']

c_lyrics = get_lyrics('correct.ust')
o_lyrics = get_lyrics('output_hybrid.ust')

with open('lyrics_comparison.txt', 'w', encoding='utf-8') as f:
    f.write("Correct Lyrics (first 100):\n")
    f.write(" ".join(c_lyrics[:100]) + "\n\n")
    f.write("Output Lyrics (first 100):\n")
    f.write(" ".join(o_lyrics[:100]) + "\n")
