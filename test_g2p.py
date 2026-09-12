import g2p_en

# Create G2P instance
g2p = g2p_en.G2p()

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

def english_to_vocaloid_hiragana(text):
    phonemes = g2p(text)
    # Strip stress numbers and keep valid phonemes
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

import sys
sys.stdout.reconfigure(encoding='utf-8')
print(english_to_vocaloid_hiragana('Apple'))
print(english_to_vocaloid_hiragana('When I'))
print(english_to_vocaloid_hiragana('want to'))
