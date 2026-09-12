import numpy as np
import librosa
from scipy.io import wavfile
import pyworld as pw
import warnings
import torch

# 2. ピッチ連続データを取得 (PyWorld / CREPE)
# ==========================================
def get_pitch_contour(audio_path, frame_period=10.0, f0_model="PyWorld", pyworld_silence_threshold=-40.0):
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
        
        # 音量（RMS）を計算し、F0が検出できなくても音量が閾値以上なら有声とする
        hop_length = int(sr * (frame_period / 1000.0))
        rms = librosa.feature.rms(y=audio, frame_length=hop_length*2, hop_length=hop_length)[0]
        if len(rms) < len(f0):
            rms = np.pad(rms, (0, len(f0) - len(rms)), mode='edge')
        else:
            rms = rms[:len(f0)]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        confidence = ((f0 > 0) | (rms_db > pyworld_silence_threshold)).astype(float)
    
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

def compute_global_dynamics(audio_path, sensitivity=100.0, min_volume_ratio=0.2):
    sr, audio = wavfile.read(audio_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
        
    audio_f32 = audio.astype(np.float32)
    if audio.dtype == np.int16:
        audio_f32 /= 32768.0
    elif audio.dtype == np.int32:
        audio_f32 /= 2147483648.0
        
    hop_length = max(64, int(sr * 0.005))
    frame_length = hop_length * 4
    
    rms = librosa.feature.rms(y=audio_f32, frame_length=frame_length, hop_length=hop_length)[0]
    
    db = librosa.amplitude_to_db(rms, ref=np.max)
    db_max = np.max(db)
    
    active_mask = db > -45.0
    if np.any(active_mask):
        center_db = np.mean(db[active_mask])
    else:
        center_db = -12.0
        
    range_db = max(6.0, float(db_max - center_db))
    
    norm_vol = 0.5 + 0.5 * (db - center_db) / range_db
    
    sens_factor = max(0.0, min(100.0, float(sensitivity))) / 100.0
    sens_vol = 0.5 + (norm_vol - 0.5) * sens_factor
    
    final_vol = np.clip(sens_vol, min_volume_ratio, 1.0)
    
    rms_times = librosa.frames_to_time(np.arange(len(final_vol)), sr=sr, hop_length=hop_length)
    
    return rms_times, final_vol

def get_volume_contour(audio_path, time_array, sensitivity=100.0, min_volume_ratio=0.2):
    rms_times, final_vol = compute_global_dynamics(audio_path, sensitivity, min_volume_ratio)
    if len(rms_times) > 1:
        return np.interp(time_array, rms_times, final_vol)
    else:
        return np.full_like(time_array, final_vol[0] if len(final_vol)>0 else 0.5)

