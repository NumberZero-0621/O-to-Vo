import re
import pyworld as pw
import pyopenjtalk
import jaconv
import g2p_en

# --- 英語からボカロ風ひらがなへの変換 ---
g2p_instance = None

def preload_models():
    """アプリ起動時にモデルや辞書をキャッシュし、変換時のフリーズを防ぐ"""
    global g2p_instance
    if g2p_instance is None:
        try:
            g2p_instance = g2p_en.G2p()
        except Exception as e:
            print(f"g2p_en preload error: {e}")
            
    try:
        # 初回実行でpyopenjtalkの辞書をロード
        pyopenjtalk.run_frontend("あ")
    except Exception as e:
        print(f"pyopenjtalk preload error: {e}")

VOWEL_GROUPS = {
    'AA': ('A', ''), 'AE': ('A', ''), 'AH': ('A', ''),
    'AO': ('O', ''), 'AW': ('A', 'う'), 'AY': ('A', 'い'),
    'EH': ('E', ''), 'ER': ('A', 'ー'), 'EY': ('E', 'い'),
    'IH': ('I', ''), 'IY': ('I', 'ー'),
    'OW': ('O', 'ー'), 'OY': ('O', 'い'),
    'UH': ('U', ''), 'UW': ('U', 'ー')
}

ARPABET_CONSONANTS_ALONE = {
    'B': 'ぶ', 'CH': 'ち', 'D': 'ど', 'DH': 'ず', 'F': 'ふ', 'G': 'ぐ',
    'HH': 'は', 'JH': 'じ', 'K': 'く', 'L': 'る', 'M': 'む', 'N': 'ん',
    'NG': 'ん', 'P': 'ぷ', 'R': 'あー', 'S': 'す', 'SH': 'し', 'T': 'と',
    'TH': 'す', 'V': 'ぶ', 'W': 'う', 'Y': 'い', 'Z': 'ず', 'ZH': 'じ'
}

CV_MAP = {
    'B': {'A': 'ば', 'I': 'び', 'U': 'ぶ', 'E': 'べ', 'O': 'ぼ'},
    'CH': {'A': 'ちゃ', 'I': 'ち', 'U': 'ちゅ', 'E': 'ちぇ', 'O': 'ちょ'},
    'D': {'A': 'だ', 'I': 'でぃ', 'U': 'どぅ', 'E': 'で', 'O': 'ど'},
    'DH': {'A': 'ざ', 'I': 'じ', 'U': 'ず', 'E': 'ぜ', 'O': 'ぞ'},
    'F': {'A': 'ふぁ', 'I': 'ふぃ', 'U': 'ふ', 'E': 'ふぇ', 'O': 'ふぉ'},
    'G': {'A': 'が', 'I': 'ぎ', 'U': 'ぐ', 'E': 'げ', 'O': 'ご'},
    'HH': {'A': 'は', 'I': 'ひ', 'U': 'ふ', 'E': 'へ', 'O': 'ほ'},
    'JH': {'A': 'じゃ', 'I': 'じ', 'U': 'じゅ', 'E': 'じぇ', 'O': 'じょ'},
    'K': {'A': 'か', 'I': 'き', 'U': 'く', 'E': 'け', 'O': 'こ'},
    'L': {'A': 'ら', 'I': 'り', 'U': 'る', 'E': 'れ', 'O': 'ろ'},
    'M': {'A': 'ま', 'I': 'み', 'U': 'む', 'E': 'め', 'O': 'も'},
    'N': {'A': 'な', 'I': 'に', 'U': 'ぬ', 'E': 'ね', 'O': 'の'},
    'NG': {'A': 'が', 'I': 'ぎ', 'U': 'ぐ', 'E': 'げ', 'O': 'ご'}, 
    'P': {'A': 'ぱ', 'I': 'ぴ', 'U': 'ぷ', 'E': 'ぺ', 'O': 'ぽ'},
    'R': {'A': 'ら', 'I': 'り', 'U': 'る', 'E': 'れ', 'O': 'ろ'},
    'S': {'A': 'さ', 'I': 'し', 'U': 'す', 'E': 'せ', 'O': 'そ'},
    'SH': {'A': 'しゃ', 'I': 'し', 'U': 'しゅ', 'E': 'しぇ', 'O': 'しょ'},
    'T': {'A': 'た', 'I': 'てぃ', 'U': 'とぅ', 'E': 'て', 'O': 'と'},
    'TH': {'A': 'さ', 'I': 'し', 'U': 'す', 'E': 'せ', 'O': 'そ'},
    'V': {'A': 'ば', 'I': 'び', 'U': 'ぶ', 'E': 'べ', 'O': 'ぼ'},
    'W': {'A': 'わ', 'I': 'うぃ', 'U': 'う', 'E': 'うぇ', 'O': 'を'},
    'Y': {'A': 'や', 'I': 'い', 'U': 'ゆ', 'E': 'いぇ', 'O': 'よ'},
    'Z': {'A': 'ざ', 'I': 'じ', 'U': 'ず', 'E': 'ぜ', 'O': 'ぞ'},
    'ZH': {'A': 'じゃ', 'I': 'じ', 'U': 'じゅ', 'E': 'じぇ', 'O': 'じょ'},
}

def convert_english_segment(text):
    global g2p_instance
    if g2p_instance is None:
        g2p_instance = g2p_en.G2p()
        
    phonemes = g2p_instance(text)
    cleaned_phonemes = []
    for p in phonemes:
        p_base = ''.join([c for c in p if c.isalpha()])
        if p_base in VOWEL_GROUPS or p_base in ARPABET_CONSONANTS_ALONE:
            cleaned_phonemes.append(p_base)
            
    result = ""
    i = 0
    while i < len(cleaned_phonemes):
        p1 = cleaned_phonemes[i]
        
        # Sokuon check (consecutive identical consonants, except N)
        if i + 1 < len(cleaned_phonemes):
            p2 = cleaned_phonemes[i+1]
            if p1 == p2 and p1 in ARPABET_CONSONANTS_ALONE and p1 != 'N':
                result += "っ"
                i += 1
                continue
                
        # Is it a vowel?
        if p1 in VOWEL_GROUPS:
            v_type, suffix = VOWEL_GROUPS[p1]
            v_map = {'A': 'あ', 'I': 'い', 'U': 'う', 'E': 'え', 'O': 'お'}
            result += v_map[v_type] + suffix
            i += 1
            continue
            
        # It's a consonant
        if i + 1 < len(cleaned_phonemes):
            p2 = cleaned_phonemes[i+1]
            if p2 in VOWEL_GROUPS:
                # CV combination
                v_type, suffix = VOWEL_GROUPS[p2]
                if p1 in CV_MAP and v_type in CV_MAP[p1]:
                    result += CV_MAP[p1][v_type] + suffix
                else:
                    result += ARPABET_CONSONANTS_ALONE.get(p1, '') + VOWEL_GROUPS[p2][0] + suffix
                i += 2
                continue
                
        # Consonant alone
        result += ARPABET_CONSONANTS_ALONE.get(p1, '')
        i += 1
        
    return result

def convert_english_to_vocaloid_hiragana(text):
    if not text:
        return text
    pattern = r'[A-Za-z0-9\']+(?:[ \t\.,\!\?]+[A-Za-z0-9\']+)*'
    def replacer(m):
        return convert_english_segment(m.group(0))
    return re.sub(pattern, replacer, text)

# 漢字・カタカナ等をひらがなのモーラに分割する関数
def get_word_moras(text):
    """pyopenjtalkとjaconvを用いてテキストをひらがなのモーラ（文字）リストに変換する"""
    if not text or text.strip() == "":
        return []
        
    text = convert_english_to_vocaloid_hiragana(text)
    
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
        exclude_chars = "’、っ。！?？（）() 　.,'\"-\n\r\t "
        return [c for c in text if c not in exclude_chars]

def get_vocaloid_preview_text(text):
    """プレビュー用にスペースや改行などのレイアウトを保持しつつ、ふりがなに変換する"""
    if not text:
        return text
    text = convert_english_to_vocaloid_hiragana(text)
    
    try:
        features = pyopenjtalk.run_frontend(text)
    except Exception:
        return text
        
    preview = ""
    orig_ptr = 0
    
    for f in features:
        node_str = f.get('string', '')
        if not node_str or node_str.strip('　 \t\n\r') == '':
            continue
            
        start_idx = text.find(node_str, orig_ptr)
        if start_idx != -1:
            preview += text[orig_ptr:start_idx]
            pron = f.get('pron', '')
            if not pron or pron == '*':
                pron = node_str
            hira_pron = jaconv.kata2hira(pron)
            hira_pron = hira_pron.replace('’', '').replace("'", "")
            preview += hira_pron
            orig_ptr = start_idx + len(node_str)
            
    preview += text[orig_ptr:]
    return preview

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
