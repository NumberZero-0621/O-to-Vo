import tkinter as tk
import customtkinter as ctk
import numpy as np
import math
import threading
import librosa
import os
import pygame
import sounddevice as sd
import soundfile as sf

class VocalEditor(ctk.CTkFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # State
        self.notes = []
        self.global_pitch_curve = {} # time(sec) -> pitch
        self.global_dyn_curve = {}   # time(sec) -> dyn (0.0 - 1.0)
        self.waveform_data = None    # envelope data for drawing
        self.audio_file_path = None
        
        pygame.mixer.init()
        self.is_playing = False
        self.play_start_time = 0
        self.current_playback_time = 0
        
        self.is_recording = False
        self.recorded_audio = []
        self.record_stream = None
        
        self.scroll_offset_x = 0.0
        self.scroll_h = 1000
        
        self.history = []
        self.history_idx = -1
        
        self.tempo = 120.0
        self.pixels_per_second = 100
        self.pixels_per_pitch = 12
        self.min_pitch = 36
        self.max_pitch = 84
        
        self.selected_note_indices = set()
        self.drag_mode = None
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.selection_rect = None # [x1, y1, x2, y2]
        self.active_layer = "note" # "note", "pitch", "dyn"
        
        # ----------------
        # Left Panel
        # ----------------
        self.left_panel = ctk.CTkFrame(self, width=120)
        self.left_panel.grid(row=0, column=0, sticky="ns", padx=2, pady=2)
        self.left_panel.grid_propagate(False)
        
        self.tool_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.tool_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.tool_var = tk.StringVar(value="select")
        
        tools = [
            ("select", "↖", "選択/移動"), 
            ("pen", "✏", "ペン"), 
            ("eraser", "🗑", "消しゴム"), 
            ("line", "╱", "直線描画"),
            ("link", "🔗", "結合/分割")
        ]
        
        for i, (t_val, t_text, t_tooltip) in enumerate(tools):
            row = i // 2
            col = i % 2
            btn = ctk.CTkRadioButton(self.tool_frame, text=t_text, variable=self.tool_var, value=t_val, width=40)
            btn.grid(row=row, column=col, padx=2, pady=2)
            
        ctk.CTkFrame(self.left_panel, height=2, fg_color="#555555").pack(fill=tk.X, pady=5, padx=5)
        
        # Layers Toggles & Buttons
        self.show_note_var = ctk.BooleanVar(value=True)
        self.show_pitch_var = ctk.BooleanVar(value=True)
        self.show_dyn_var = ctk.BooleanVar(value=True)
        
        self.layer_frames = {}
        for layer_id, label_text, var in [("note", "ノート", self.show_note_var), 
                                          ("pitch", "ピッチ", self.show_pitch_var), 
                                          ("dyn", "ダイナミクス", self.show_dyn_var)]:
            f = ctk.CTkFrame(self.left_panel, fg_color="transparent")
            f.pack(fill=tk.X, padx=5, pady=2)
            cb = ctk.CTkCheckBox(f, text="", variable=var, command=self.redraw, width=20)
            cb.pack(side=tk.LEFT)
            lbl = ctk.CTkLabel(f, text=label_text, cursor="hand2")
            lbl.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
            # Click label to activate layer
            lbl.bind("<Button-1>", lambda e, lid=layer_id: self.set_active_layer(lid))
            self.layer_frames[layer_id] = f

        self.set_active_layer("note")
        
        ctk.CTkFrame(self.left_panel, height=2, fg_color="#555555").pack(fill=tk.X, pady=5, padx=5)
        
        # Zoom Controls
        ctk.CTkLabel(self.left_panel, text="横ズーム").pack(anchor="w", padx=10)
        self.zoom_x_slider = ctk.CTkSlider(self.left_panel, from_=50, to=500, command=self.on_zoom_x)
        self.zoom_x_slider.set(100)
        self.zoom_x_slider.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        ctk.CTkLabel(self.left_panel, text="縦ズーム").pack(anchor="w", padx=10)
        self.zoom_y_slider = ctk.CTkSlider(self.left_panel, from_=8, to=40, command=self.on_zoom_y)
        self.zoom_y_slider.set(12)
        self.zoom_y_slider.pack(fill=tk.X, padx=10, pady=(0, 10))

        # -----------------
        # Right Panel
        # -----------------
        self.right_panel = ctk.CTkFrame(self, fg_color="transparent")
        self.right_panel.grid(row=0, column=1, sticky="nsew")
        self.right_panel.grid_rowconfigure(1, weight=1)
        self.right_panel.grid_columnconfigure(0, weight=1)
        
        # Top Playback & Global Controls
        self.top_ctrl_frame = ctk.CTkFrame(self.right_panel, height=35)
        self.top_ctrl_frame.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        
        self.play_btn = ctk.CTkButton(self.top_ctrl_frame, text="▶", width=30, command=self.toggle_playback)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        self.stop_btn = ctk.CTkButton(self.top_ctrl_frame, text="■", width=30, command=self.stop_playback)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        self.mic_btn = ctk.CTkButton(self.top_ctrl_frame, text="○", width=30, text_color="white", fg_color="#cc0000", hover_color="#aa0000", command=self.toggle_recording)
        self.mic_btn.pack(side=tk.LEFT, padx=2) 
        
        ctk.CTkFrame(self.top_ctrl_frame, width=2, height=20, fg_color="#555555").pack(side=tk.LEFT, padx=10)
        
        self.metronome_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(self.top_ctrl_frame, text="メトロノーム", variable=self.metronome_var).pack(side=tk.LEFT, padx=2)
        
        ctk.CTkFrame(self.top_ctrl_frame, width=2, height=20, fg_color="#555555").pack(side=tk.LEFT, padx=10)
        
        ctk.CTkLabel(self.top_ctrl_frame, text="移調:").pack(side=tk.LEFT, padx=2)
        self.trans_var = tk.StringVar(value="0")
        ctk.CTkEntry(self.top_ctrl_frame, textvariable=self.trans_var, width=30).pack(side=tk.LEFT, padx=2)
        ctk.CTkButton(self.top_ctrl_frame, text="適用", width=40, command=self.apply_transpose).pack(side=tk.LEFT, padx=2)
        
        ctk.CTkFrame(self.top_ctrl_frame, width=2, height=20, fg_color="#555555").pack(side=tk.LEFT, padx=10)
        
        ctk.CTkLabel(self.top_ctrl_frame, text="BPM:").pack(side=tk.LEFT, padx=2)
        self.bpm_label = ctk.CTkLabel(self.top_ctrl_frame, text="120")
        self.bpm_label.pack(side=tk.LEFT, padx=2)
        
        ctk.CTkLabel(self.top_ctrl_frame, text="拍子: 4/4").pack(side=tk.LEFT, padx=10)
        
        ctk.CTkLabel(self.top_ctrl_frame, text="クオンタイズ:").pack(side=tk.LEFT, padx=2)
        self.quantize_var = tk.StringVar(value="None")
        ctk.CTkOptionMenu(self.top_ctrl_frame, variable=self.quantize_var, values=["None", "1/4", "1/8", "1/16", "1/32", "1/64"], width=70, command=self.on_quantize_change).pack(side=tk.LEFT, padx=2)
        
        self.show_wave_var = ctk.BooleanVar(value=True)
        self.show_orig_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self.top_ctrl_frame, text="元の音声", variable=self.show_orig_var, command=self.redraw).pack(side=tk.RIGHT, padx=5)
        ctk.CTkCheckBox(self.top_ctrl_frame, text="変換後の波形", variable=self.show_wave_var, command=self.redraw).pack(side=tk.RIGHT, padx=5)
        
        # Workspace (Ruler + Canvas)
        self.workspace = ctk.CTkFrame(self.right_panel)
        self.workspace.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.workspace.grid_rowconfigure(1, weight=1)
        self.workspace.grid_columnconfigure(0, weight=1)
        
        self.ruler_canvas = tk.Canvas(self.workspace, bg="#222222", height=25, highlightthickness=0)
        self.ruler_canvas.grid(row=0, column=0, sticky="ew")
        
        self.canvas = tk.Canvas(self.workspace, bg="#2b2b2b", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        
        self.h_scroll = ctk.CTkScrollbar(self.workspace, orientation="horizontal", command=self._on_hscroll)
        self.h_scroll.grid(row=2, column=0, sticky="ew")
        
        self.v_scroll = ctk.CTkScrollbar(self.workspace, orientation="vertical", command=self.canvas.yview)
        self.v_scroll.grid(row=1, column=1, sticky="ns")
        
        self.canvas.configure(yscrollcommand=self.v_scroll.set)
        
        # Bindings
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Double-1>", self.on_double_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        
        self.canvas.bind("<Button-1>", lambda e: self.canvas.focus_set(), add="+")
        self.canvas.bind("<Delete>", self.on_delete)
        self.canvas.bind("<BackSpace>", self.on_delete)
        
        # Undo / Redo bindings
        self.canvas.bind("<Control-z>", self.undo)
        self.canvas.bind("<Control-y>", self.redo)
        self.canvas.bind("<Control-Shift-Z>", self.redo)
        
        # Mousewheel
        self.canvas.bind("<MouseWheel>", self._on_mousewheel_v)
        self.canvas.bind("<Shift-MouseWheel>", self._on_mousewheel_h)
        
        self.canvas.bind("<Configure>", lambda e: self.update_canvas_scrollregion())
        self.ruler_canvas.bind("<Configure>", lambda e: self.redraw_ruler())
        
        self._init_metronome()
        self._setup_canvas()

    def set_active_layer(self, layer_id):
        self.active_layer = layer_id
        for lid, f in self.layer_frames.items():
            if lid == layer_id:
                f.configure(fg_color="#334455") # Highlight
            else:
                f.configure(fg_color="transparent")
        
        if layer_id == "note" and not self.show_note_var.get():
            self.show_note_var.set(True)
        elif layer_id == "pitch" and not self.show_pitch_var.get():
            self.show_pitch_var.set(True)
        elif layer_id == "dyn" and not self.show_dyn_var.get():
            self.show_dyn_var.set(True)
            
        if hasattr(self, 'canvas'):
            self.redraw()
            
    def _init_metronome(self):
        sample_rate = 44100
        t = np.linspace(0, 0.05, int(sample_rate * 0.05), False)
        high = np.int16(np.sin(880 * 2 * np.pi * t) * 16000)
        low = np.int16(np.sin(440 * 2 * np.pi * t) * 16000)
        
        import scipy.io.wavfile as wav
        h_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "click_h.wav")
        l_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "click_l.wav")
        
        wav.write(h_file, sample_rate, high)
        wav.write(l_file, sample_rate, low)
        
        try:
            self.click_high = pygame.mixer.Sound(h_file)
            self.click_low = pygame.mixer.Sound(l_file)
        except Exception as e:
            print(f"Metronome init error: {e}")
            self.click_high = None
            self.click_low = None
        self.last_beat_played = -1

    def _save_state(self):
        import copy
        state = {
            "notes": copy.deepcopy(self.notes),
            "pitch_curve": copy.deepcopy(self.global_pitch_curve),
            "dyn_curve": copy.deepcopy(self.global_dyn_curve)
        }
        if self.history_idx < len(self.history) - 1:
            self.history = self.history[:self.history_idx + 1]
            
        self.history.append(state)
        if len(self.history) > 50:
            self.history.pop(0)
        else:
            self.history_idx += 1
            
    def _load_state(self):
        if self.history_idx < 0 or self.history_idx >= len(self.history): return
        import copy
        state = self.history[self.history_idx]
        self.notes = copy.deepcopy(state["notes"])
        self.global_pitch_curve = copy.deepcopy(state["pitch_curve"])
        self.global_dyn_curve = copy.deepcopy(state["dyn_curve"])
        self.redraw()
        
    def undo(self, event=None):
        if self.history_idx > 0:
            self.history_idx -= 1
            self._load_state()
            
    def redo(self, event=None):
        if self.history_idx < len(self.history) - 1:
            self.history_idx += 1
            self._load_state()
            
    def apply_transpose(self):
        try:
            shift = int(self.trans_var.get())
            if shift == 0: return
        except: return
        
        target_indices = self.selected_note_indices if self.selected_note_indices else range(len(self.notes))
        for idx in target_indices:
            if self.notes[idx]["text"] != "R":
                self.notes[idx]["pitch"] = max(self.min_pitch, min(self.max_pitch, self.notes[idx]["pitch"] + shift))
        self._save_state()
        self.redraw()
        
    def _on_mousewheel_v(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def _on_mousewheel_h(self, event):
        self.scroll_offset_x += int(-1*(event.delta/120)) * 50
        self.update_canvas_scrollregion()
        self.redraw()

    def _on_hscroll(self, action, *args):
        canvas_w = self.canvas.winfo_width()
        max_time = max([n["end"] for n in self.notes] + [300.0]) if self.notes else 300.0
        if self.current_playback_time > max_time - 10:
            max_time = self.current_playback_time + 60.0
        total_w = max_time * self.pixels_per_second
        
        if action == 'moveto':
            self.scroll_offset_x = float(args[0]) * total_w
        elif action == 'scroll':
            amount = int(args[0])
            unit = args[1]
            if unit == 'units':
                self.scroll_offset_x += amount * 50
            elif unit == 'pages':
                self.scroll_offset_x += amount * canvas_w * 0.9
                
        self.update_canvas_scrollregion()
        self.redraw()

    def _setup_canvas(self):
        self.update_canvas_scrollregion()
        self.redraw()
        
    def update_canvas_scrollregion(self):
        canvas_w = self.canvas.winfo_width()
        if canvas_w <= 1: canvas_w = 1000
        
        max_time = max([n["end"] for n in self.notes] + [300.0]) if self.notes else 300.0
        if self.current_playback_time > max_time - 10:
            max_time = self.current_playback_time + 60.0
            
        total_w = max_time * self.pixels_per_second
        self.scroll_h = (self.max_pitch - self.min_pitch + 1) * self.pixels_per_pitch
        
        # Configure canvas to NOT scroll natively in X
        self.canvas.configure(scrollregion=(0, 0, canvas_w, self.scroll_h))
        self.ruler_canvas.configure(scrollregion=(0, 0, canvas_w, 25))
        
        # Clamp and update virtual scrollbar
        if total_w <= canvas_w:
            self.h_scroll.set(0.0, 1.0)
            self.scroll_offset_x = 0
        else:
            self.scroll_offset_x = max(0, min(self.scroll_offset_x, total_w - canvas_w))
            first = self.scroll_offset_x / total_w
            last = (self.scroll_offset_x + canvas_w) / total_w
            self.h_scroll.set(first, last)

    def on_zoom_x(self, val):
        self.pixels_per_second = float(val)
        self.update_canvas_scrollregion()
        self.redraw()
        
    def on_zoom_y(self, val):
        self.pixels_per_pitch = float(val)
        self.update_canvas_scrollregion()
        self.redraw()

    def set_data(self, notes, audio_file=None):
        self.notes = notes
        self.global_pitch_curve = {}
        for n in self.notes:
            if "pitch_curve" in n and n["pitch_curve"]:
                step = (n["end"] - n["start"]) / max(1, len(n["pitch_curve"]))
                for j, p in enumerate(n["pitch_curve"]):
                    if p is not None and not np.isnan(p):
                        t = n["start"] + j * step
                        self.global_pitch_curve[t] = p
                        
        try:
            tempo_str = self.master.master.app_vars["tempo_var"].get()
            if tempo_str: self.tempo = float(tempo_str)
        except: pass
            
        self.bpm_label.configure(text=f"{self.tempo}")
        self.update_canvas_scrollregion()
        self.redraw()
        
        # Save initial state
        self.history = []
        self.history_idx = -1
        self._save_state()
        
        if audio_file != self.audio_file_path:
            self.audio_file_path = audio_file
            if audio_file:
                threading.Thread(target=self._load_waveform, args=(audio_file,), daemon=True).start()
            else:
                self.waveform_data = None
                self.redraw()

    def _load_waveform(self, audio_file):
        try:
            # Load with low SR for visualization
            y, sr = librosa.load(audio_file, sr=4000)
            frame_length = max(1, len(y) // 2000) # Max 2000 points
            
            envelope = []
            for i in range(0, len(y), frame_length):
                chunk = y[i:i+frame_length]
                envelope.append(float(np.max(np.abs(chunk))))
                
            self.waveform_data = {
                "envelope": envelope,
                "sr": sr,
                "frame_length": frame_length,
                "duration": len(y) / sr
            }
            # Trigger redraw safely
            self.after(0, self.redraw)
        except Exception as e:
            print(f"Waveform load error: {e}")
            self.waveform_data = None
            
    def toggle_playback(self):
        if self.is_playing:
            self.stop_playback()
        else:
            if self.audio_file_path and os.path.exists(self.audio_file_path):
                try:
                    pygame.mixer.music.load(self.audio_file_path)
                    self.play_start_time = max(0.0, self.x_to_time(0))
                    
                    pygame.mixer.music.play(start=self.play_start_time)
                    self.is_playing = True
                    
                    beat_len = 60.0 / max(1.0, self.tempo)
                    self.last_beat_played = int(self.play_start_time / beat_len) - 1
                    
                    self.play_btn.configure(text="⏸")
                    self._playback_loop()
                except Exception as e:
                    print(f"Playback error: {e}")

    def stop_playback(self):
        self.is_playing = False
        pygame.mixer.music.stop()
        self.play_btn.configure(text="▶")
        self.canvas.delete("playhead")
        self.ruler_canvas.delete("playhead")

    def _playback_loop(self):
        if self.is_playing:
            if pygame.mixer.music.get_busy():
                pos_ms = pygame.mixer.music.get_pos()
                if pos_ms >= 0:
                    self.current_playback_time = self.play_start_time + (pos_ms / 1000.0)
                    self.update_playhead()
                    
                    if self.metronome_var.get() and self.click_high:
                        beat_len = 60.0 / max(1.0, self.tempo)
                        current_beat = int(self.current_playback_time / beat_len)
                        if current_beat > self.last_beat_played:
                            if current_beat % 4 == 0:
                                self.click_high.play()
                            else:
                                self.click_low.play()
                            self.last_beat_played = current_beat
                            
                    self.after(30, self._playback_loop)
                else:
                    self.stop_playback()
            else:
                self.stop_playback()

    def update_playhead(self):
        self.canvas.delete("playhead")
        self.ruler_canvas.delete("playhead")
        x = self.time_to_x(self.current_playback_time)
        
        canvas_w = self.canvas.winfo_width()
        if x > canvas_w - 50:
            self.scroll_offset_x += (x - (canvas_w - 50))
            self.update_canvas_scrollregion()
            self.redraw()
            x = self.time_to_x(self.current_playback_time)
        
        self.canvas.create_line(x, 0, x, 2000, fill="#00ff00", tags="playhead", width=2)
        self.ruler_canvas.create_line(x, 0, x, 25, fill="#00ff00", tags="playhead", width=2)

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
            
    def start_recording(self):
        try:
            self.is_recording = True
            self.mic_btn.configure(text="■")
            self.recorded_audio = []
            
            def callback(indata, frames, time, status):
                if status: print(status)
                self.recorded_audio.append(indata.copy())
                
            self.record_stream = sd.InputStream(samplerate=44100, channels=1, callback=callback)
            self.record_stream.start()
        except Exception as e:
            print(f"録音開始エラー: {e}")
            self.stop_recording()
        
    def stop_recording(self):
        self.is_recording = False
        # default colors
        self.mic_btn.configure(text="○")
        
        if self.record_stream:
            self.record_stream.stop()
            self.record_stream.close()
            self.record_stream = None
            
        if self.recorded_audio:
            audio_data = np.concatenate(self.recorded_audio, axis=0)
            out_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recorded_audio.wav")
            try:
                sf.write(out_file, audio_data, 44100)
                print(f"録音完了: {out_file}")
                
                # AppVars update
                try:
                    app = self.master.master
                    app.app_vars["audio_file_path"].set(out_file)
                    app.app_vars["output_base_path"].set(os.path.splitext(out_file)[0])
                except Exception as e:
                    print(f"AppVars update error: {e}")
                    
                self.set_data(self.notes, out_file)
            except Exception as e:
                print(f"録音保存エラー: {e}")
        
    def on_quantize_change(self, val):
        self.redraw()
        
    def get_quantize_step(self):
        q = self.quantize_var.get()
        if q == "None": return None
        beat_len = 60.0 / max(1.0, self.tempo)
        if q == "1/4": return beat_len
        elif q == "1/8": return beat_len / 2.0
        elif q == "1/16": return beat_len / 4.0
        elif q == "1/32": return beat_len / 8.0
        elif q == "1/64": return beat_len / 16.0
        return None

    def snap_time(self, time_val):
        step = self.get_quantize_step()
        if step is None: return time_val
        return round(time_val / step) * step

    def pitch_to_y(self, pitch):
        return (self.max_pitch - pitch) * self.pixels_per_pitch
        
    def y_to_pitch(self, y):
        return self.max_pitch - ((y - self.pixels_per_pitch/2) / self.pixels_per_pitch)
        
    def dyn_to_y(self, dyn):
        h = (self.max_pitch - self.min_pitch + 1) * self.pixels_per_pitch
        return h - (dyn * 200) # Draw over bottom 200px
        
    def y_to_dyn(self, y):
        h = (self.max_pitch - self.min_pitch + 1) * self.pixels_per_pitch
        return max(0.0, min(1.0, (h - y) / 200.0))
        
    def time_to_x(self, time):
        return time * self.pixels_per_second - self.scroll_offset_x
        
    def x_to_time(self, x):
        return (x + self.scroll_offset_x) / self.pixels_per_second

    def redraw_ruler(self):
        self.ruler_canvas.delete("all")
        canvas_w = self.canvas.winfo_width()
        
        beat_len = 60.0 / max(1.0, self.tempo)
        start_time = max(0.0, self.x_to_time(0))
        end_time = self.x_to_time(canvas_w)
        
        start_beat = int(start_time / beat_len)
        end_beat = int(end_time / beat_len) + 1
        
        for b in range(start_beat, end_beat):
            x = self.time_to_x(b * beat_len)
            measure = (b // 4) + 1
            beat_in_measure = (b % 4) + 1
            
            if beat_in_measure == 1:
                self.ruler_canvas.create_line(x, 0, x, 25, fill="#888888", width=2)
                self.ruler_canvas.create_text(x + 3, 2, text=f"{measure}.{beat_in_measure}", anchor=tk.NW, fill="#ffffff", font=("Arial", 10, "bold"))
            else:
                self.ruler_canvas.create_line(x, 15, x, 25, fill="#555555")
                self.ruler_canvas.create_text(x + 2, 10, text=f"{measure}.{beat_in_measure}", anchor=tk.NW, fill="#888888", font=("Arial", 8))

    def _draw_curve(self, curve_dict, color, tag, val_to_y_func):
        if not curve_dict: return
        sorted_times = sorted(curve_dict.keys())
        chunk = []
        for i, t in enumerate(sorted_times):
            p = curve_dict[t]
            x, y = self.time_to_x(t), val_to_y_func(p)
            
            if not chunk:
                chunk.extend([x, y])
            else:
                prev_x = chunk[-2]
                if x - prev_x > self.pixels_per_second * 0.2: # break line if > 0.2s gap
                    if len(chunk) >= 4:
                        self.canvas.create_line(chunk, fill=color, smooth=False, width=2, tags=tag)
                    chunk = [x, y]
                else:
                    chunk.extend([x, y])
        if len(chunk) >= 4:
            self.canvas.create_line(chunk, fill=color, smooth=False, width=2, tags=tag)

    def redraw(self):
        if not hasattr(self, 'canvas'): return
        self.canvas.delete("all")
        self.redraw_ruler()
        
        canvas_w = self.canvas.winfo_width()
        
        start_time = max(0.0, self.x_to_time(0))
        end_time = self.x_to_time(canvas_w)
        
        # Grid lines (Pitch) - draw across visible width
        for p in range(self.min_pitch, self.max_pitch + 1):
            y = self.pitch_to_y(p)
            color = "#222222" if p % 12 in [1, 3, 6, 8, 10] else "#333333"
            self.canvas.create_rectangle(0, y, canvas_w, y + self.pixels_per_pitch, fill=color, outline="")
            if p % 12 not in [1, 3, 6, 8, 10]:
                self.canvas.create_line(0, y, canvas_w, y, fill="#444444")
            
        # Grid lines (Time)
        beat_len = 60.0 / max(1.0, self.tempo)
        start_beat = int(start_time / beat_len)
        end_beat = int(end_time / beat_len) + 1
        for b in range(start_beat, end_beat):
            x = self.time_to_x(b * beat_len)
            width = 2 if b % 4 == 0 else 1
            color = "#666666" if b % 4 == 0 else "#444444"
            self.canvas.create_line(x, 0, x, self.pitch_to_y(self.min_pitch), fill=color, width=width)
                
        q_step = self.get_quantize_step()
        if q_step and q_step < beat_len:
            start_s = int(start_time / q_step)
            end_s = int(end_time / q_step) + 1
            for s in range(start_s, end_s):
                t = s * q_step
                if abs((t % beat_len) / beat_len) > 0.01 and abs((t % beat_len) / beat_len) < 0.99:
                    x = self.time_to_x(t)
                    self.canvas.create_line(x, 0, x, self.pitch_to_y(self.min_pitch), fill="#3a3a3a")

        # Draw Waveform (Original Audio)
        if self.show_orig_var.get() and self.waveform_data:
            env = self.waveform_data["envelope"]
            max_env = max(env) if env and max(env) > 0 else 1.0
            y_center = (self.max_pitch - self.min_pitch + 1) * self.pixels_per_pitch / 2
            y_scale = y_center * 0.8 / max_env
            
            points = []
            # top half
            for i, val in enumerate(env):
                t = i * self.waveform_data["frame_length"] / self.waveform_data["sr"]
                x = self.time_to_x(t)
                y = y_center - (val * y_scale)
                points.extend([x, y])
            # bottom half
            for i in range(len(env)-1, -1, -1):
                val = env[i]
                t = i * self.waveform_data["frame_length"] / self.waveform_data["sr"]
                x = self.time_to_x(t)
                y = y_center + (val * y_scale)
                points.extend([x, y])
                
            if len(points) > 4:
                self.canvas.create_polygon(points, fill="white", stipple="gray25", outline="")

        # Draw Notes
        if self.show_note_var.get():
            for i, note in enumerate(self.notes):
                if note["end"] < start_time or note["start"] > end_time: continue
                if note["text"] == "R": continue
                x1 = self.time_to_x(note["start"])
                x2 = self.time_to_x(note["end"])
                y1 = self.pitch_to_y(note["pitch"])
                y2 = y1 + self.pixels_per_pitch
                
                fill_color = "#144870" if i in self.selected_note_indices else "#1f6aa5"
                outline_color = "#88ccff" if i in self.selected_note_indices else "#3a8cdb"
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline=outline_color, tags=("note", f"note_{i}"))
                self.canvas.create_text(x1 + 4, y1 + min(2, self.pixels_per_pitch/4), text=note["text"], anchor=tk.NW, fill="white", tags=("note", f"note_{i}"))
            
        # Draw Pitch Curve
        if self.show_pitch_var.get():
            # Use red
            self._draw_curve(self.global_pitch_curve, "#e65c5c", "pitch_curve", lambda p: self.pitch_to_y(p) + self.pixels_per_pitch/2)
            
        # Draw Dyn Curve
        if self.show_dyn_var.get():
            # Use yellow
            self._draw_curve(self.global_dyn_curve, "#e6e65c", "dyn_curve", self.dyn_to_y)
            
        # Draw Selection Rect or Line
        if self.selection_rect:
            x1, y1, x2, y2 = self.selection_rect
            if self.drag_mode == "draw_line":
                color = "#e65c5c" if self.active_layer == "pitch" else "#e6e65c"
                self.canvas.create_line(x1, y1, x2, y2, fill=color, width=2, tags="temp_line")
            else:
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="#ffffff", fill="white", stipple="gray25", dash=(4, 4), tags="sel_rect")

    def get_note_at(self, x, y):
        time_x = self.x_to_time(x)
        pitch_y = round(self.y_to_pitch(y))
        for i, note in enumerate(self.notes):
            if note["text"] == "R": continue
            if note["pitch"] == pitch_y and note["start"] <= time_x <= note["end"]:
                return i
        return None

    def get_notes_in_rect(self, x1, y1, x2, y2):
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        min_time, max_time = self.x_to_time(min_x), self.x_to_time(max_x)
        max_pitch, min_pitch = round(self.y_to_pitch(min_y)), round(self.y_to_pitch(max_y))
        
        indices = set()
        for i, note in enumerate(self.notes):
            if note["text"] == "R": continue
            if note["start"] < max_time and note["end"] > min_time and min_pitch <= note["pitch"] <= max_pitch:
                indices.add(i)
        return indices

    def on_mouse_move(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        tool = self.tool_var.get()
        layer = self.active_layer
        
        if tool == "select" and layer == "note":
            idx = self.get_note_at(x, y)
            if idx is not None:
                note = self.notes[idx]
                nx1 = self.time_to_x(note["start"])
                nx2 = self.time_to_x(note["end"])
                if x - nx1 < 5 or nx2 - x < 5:
                    self.canvas.config(cursor="sb_h_double_arrow")
                else:
                    self.canvas.config(cursor="fleur")
            else:
                self.canvas.config(cursor="")
        elif tool == "pen" or tool == "line":
            self.canvas.config(cursor="crosshair")
        elif tool == "eraser":
            self.canvas.config(cursor="X_cursor")
        elif tool == "link" and layer == "note":
            idx = self.get_note_at(x, y)
            if idx is not None:
                note = self.notes[idx]
                nx1 = self.time_to_x(note["start"])
                nx2 = self.time_to_x(note["end"])
                if x - nx1 < 10 or nx2 - x < 10:
                    self.canvas.config(cursor="sb_h_double_arrow") # join mode
                else:
                    self.canvas.config(cursor="sizing") # split mode
            else:
                self.canvas.config(cursor="")

    def on_press(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        tool = self.tool_var.get()
        layer = self.active_layer
        
        self.drag_start_x = x
        self.drag_start_y = y
        self.drag_mode = None
        self.selection_rect = None
        
        if layer == "note":
            idx = self.get_note_at(x, y)
            if tool == "select":
                if idx is not None:
                    if idx not in self.selected_note_indices:
                        self.selected_note_indices = {idx}
                    
                    self.drag_mode = "move_notes"
                    note = self.notes[idx]
                    nx1 = self.time_to_x(note["start"])
                    nx2 = self.time_to_x(note["end"])
                    
                    if x - nx1 < 5:
                        self.drag_mode = "resize_left"
                        self.selected_note_indices = {idx}
                    elif nx2 - x < 5:
                        self.drag_mode = "resize_right"
                        self.selected_note_indices = {idx}
                    
                    # Store original positions for moving
                    self.drag_original_notes = {i: dict(self.notes[i]) for i in self.selected_note_indices}
                else:
                    self.drag_mode = "rect_select"
                    self.selected_note_indices.clear()
                    
            elif tool == "pen":
                if idx is None:
                    start_time = self.snap_time(self.x_to_time(x))
                    pitch = round(self.y_to_pitch(y))
                    step = self.get_quantize_step() or 0.25
                    new_note = {"text": "あ", "start": start_time, "end": start_time + step, "pitch": pitch}
                    self.notes.append(new_note)
                    self.selected_note_indices = {len(self.notes) - 1}
                    self.drag_mode = "resize_right"
                    self.drag_start_x = self.time_to_x(start_time)
                    self._save_state() # save the creation
                    
            elif tool == "eraser":
                if idx is not None:
                    self.selected_note_indices = {idx}
                    self.on_delete(None)
                else:
                    self.drag_mode = "rect_select_erase"
                    self.selected_note_indices.clear()
                    
            elif tool == "link":
                if idx is not None:
                    note = self.notes[idx]
                    nx1 = self.time_to_x(note["start"])
                    nx2 = self.time_to_x(note["end"])
                    changed = False
                    if x - nx1 < 10:
                        changed = self._join_note_left(idx)
                    elif nx2 - x < 10:
                        changed = self._join_note_right(idx)
                    else:
                        changed = self._split_note(idx, self.snap_time(self.x_to_time(x)))
                        
                    if changed:
                        self._save_state()
                        self.redraw()

        elif layer in ["pitch", "dyn"]:
            if tool == "pen":
                self.drag_mode = "draw_curve"
                self._update_curve(x, y, layer)
            elif tool == "line":
                self.drag_mode = "draw_line"
            elif tool == "eraser":
                self.drag_mode = "erase_curve"
                self._erase_curve(x, y, layer)
            elif tool == "select":
                 self.drag_mode = "rect_select"
                
        self.redraw()
        
    def _join_note_left(self, idx):
        target_start = self.notes[idx]["start"]
        for i, n in enumerate(self.notes):
            if i != idx and abs(n["end"] - target_start) < 0.01:
                self.notes[idx]["start"] = n["start"]
                del self.notes[i]
                return True
        return False
                
    def _join_note_right(self, idx):
        target_end = self.notes[idx]["end"]
        for i, n in enumerate(self.notes):
            if i != idx and abs(n["start"] - target_end) < 0.01:
                self.notes[idx]["end"] = n["end"]
                if i < idx:
                    del self.notes[i]
                else:
                    del self.notes[i]
                return True
        return False

    def _split_note(self, idx, split_time):
        note = self.notes[idx]
        if note["start"] < split_time < note["end"]:
            new_note = dict(note)
            new_note["start"] = split_time
            note["end"] = split_time
            self.notes.append(new_note)
            return True
        return False

    def _update_curve(self, x, y, layer):
        t = self.x_to_time(x)
        if layer == "pitch":
            val = self.y_to_pitch(y)
            self.global_pitch_curve[t] = val
        elif layer == "dyn":
            val = self.y_to_dyn(y)
            self.global_dyn_curve[t] = val
            
    def _erase_curve(self, x, y, layer):
        t = self.x_to_time(x)
        radius = self.x_to_time(10) # 10 pixels radius
        target_dict = self.global_pitch_curve if layer == "pitch" else self.global_dyn_curve
        keys_to_delete = [kt for kt in target_dict.keys() if abs(kt - t) <= radius]
        for kt in keys_to_delete:
            del target_dict[kt]

    def on_drag(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        layer = self.active_layer
        
        if self.drag_mode == "rect_select" or self.drag_mode == "rect_select_erase":
            self.selection_rect = [self.drag_start_x, self.drag_start_y, x, y]
            if self.active_layer == "note":
                x1, y1, x2, y2 = self.selection_rect
                self.selected_note_indices = self.get_notes_in_rect(x1, y1, x2, y2)
            self.redraw()
            return
            
        if layer == "note":
            if self.drag_mode == "move_notes":
                dx_time = self.x_to_time(x - self.drag_start_x)
                dy_pitch = round((self.drag_start_y - y) / self.pixels_per_pitch)
                
                # Check bounds
                can_move = True
                for idx in self.selected_note_indices:
                    orig = self.drag_original_notes[idx]
                    new_start = self.snap_time(orig["start"] + dx_time)
                    if new_start < 0: can_move = False
                
                if can_move:
                    for idx in self.selected_note_indices:
                        orig = self.drag_original_notes[idx]
                        note = self.notes[idx]
                        new_start = self.snap_time(orig["start"] + dx_time)
                        note["start"] = new_start
                        note["end"] = new_start + (orig["end"] - orig["start"])
                        note["pitch"] = max(self.min_pitch, min(self.max_pitch, orig["pitch"] + dy_pitch))
                self.redraw()
                
            elif self.drag_mode == "resize_left" and self.selected_note_indices:
                idx = list(self.selected_note_indices)[0]
                note = self.notes[idx]
                new_start = self.snap_time(max(0, self.x_to_time(x)))
                if new_start < note["end"]: note["start"] = new_start
                self.redraw()
                
            elif self.drag_mode == "resize_right" and self.selected_note_indices:
                idx = list(self.selected_note_indices)[0]
                note = self.notes[idx]
                new_end = self.snap_time(max(note["start"] + 0.01, self.x_to_time(x)))
                note["end"] = new_end
                self.redraw()

        elif layer in ["pitch", "dyn"]:
            if self.drag_mode == "draw_curve":
                self._update_curve(x, y, layer)
                self.redraw()
            elif self.drag_mode == "draw_line":
                self.selection_rect = [self.drag_start_x, self.drag_start_y, x, y] # Use as temp line visual
                self.redraw()
            elif self.drag_mode == "erase_curve":
                self._erase_curve(x, y, layer)
                self.redraw()
            
    def on_release(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        layer = self.active_layer
        
        state_changed = False
        
        if self.drag_mode == "rect_select":
            self.selection_rect = None
            self.redraw()
            
        elif self.drag_mode == "rect_select_erase":
            if self.selection_rect:
                x1, y1, x2, y2 = self.selection_rect
                to_delete = sorted(list(self.get_notes_in_rect(x1, y1, x2, y2)), reverse=True)
                if to_delete:
                    for idx in to_delete:
                        del self.notes[idx]
                    state_changed = True
            self.selection_rect = None
            self.selected_note_indices.clear()
            self.redraw()
            
        elif self.drag_mode == "draw_line":
            # Generate points along the line
            steps = 50
            for i in range(steps + 1):
                px = self.drag_start_x + (x - self.drag_start_x) * (i / steps)
                py = self.drag_start_y + (y - self.drag_start_y) * (i / steps)
                self._update_curve(px, py, layer)
            self.selection_rect = None
            state_changed = True
            self.redraw()
            
        elif self.drag_mode in ["draw_curve", "erase_curve", "move_notes", "resize_left", "resize_right"]:
            state_changed = True
            
        if state_changed:
            self._save_state()
            
        self.drag_mode = None
        
    def on_double_click(self, event):
        if self.tool_var.get() == "select" and self.active_layer == "note":
            x = self.canvas.canvasx(event.x)
            y = self.canvas.canvasy(event.y)
            idx = self.get_note_at(x, y)
            if idx is not None:
                from customtkinter import CTkInputDialog
                note = self.notes[idx]
                dialog = CTkInputDialog(text="新しい歌詞を入力:", title="歌詞編集")
                new_text = dialog.get_input()
                if new_text is not None and new_text.strip():
                    note["text"] = new_text.strip()
                    self.redraw()

    def on_delete(self, event):
        if self.active_layer == "note" and self.selected_note_indices:
            for idx in sorted(list(self.selected_note_indices), reverse=True):
                del self.notes[idx]
            self.selected_note_indices.clear()
            self._save_state()
            self.redraw()
