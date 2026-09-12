import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import threading
import sys
import os
import json
import time

from ui.settings_window import SettingsWindow
from ui.vocal_editor import VocalEditor

class ThreadSafeTextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.after(0, self._write, string)

    def _write(self, string):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')

    def flush(self):
        pass

class OToVoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("O-To-Vo")
        self.root.geometry("1000x800")
        
        # 変数定義
        self.app_vars = {
            "audio_file_path": tk.StringVar(),
            "output_base_path": tk.StringVar(),
            "tempo_var": tk.StringVar(),
            "transpose_var": tk.IntVar(value=0),
            "extract_vocals_var": ctk.BooleanVar(value=False),
            "reflect_dynamics_var": ctk.BooleanVar(value=False),
            "dynamics_sensitivity_var": tk.IntVar(value=100),
            "use_predefined_lyrics_var": ctk.BooleanVar(value=False),
            "convert_to_vocaloid_var": ctk.BooleanVar(value=False),
            
            # 検索・置換用
            "search_var": tk.StringVar(),
            "replace_var": tk.StringVar(),
            
            # 設定ウィンドウ用
            "whisper_model_var": tk.StringVar(value="large-v3"),
            "w2v2_model_var": tk.StringVar(value="vumichien/wav2vec2-large-xlsr-japanese-hiragana"),
            "f0_model_var": tk.StringVar(value="PyWorld"),
            "unvoiced_threshold_var": tk.StringVar(value="10"),
            "frame_period_var": tk.StringVar(value="10.0"),
            "low_pitch_threshold_var": tk.StringVar(value="47"),
            "low_pitch_drop_amount_var": tk.StringVar(value="18"),
            "top_db_var": tk.StringVar(value="40"),
            "skip_b_cost_var": tk.StringVar(value="0.5"),
            "last_mora_ratio_var": tk.StringVar(value="0.7"),
            "min_duration_var": tk.StringVar(value="0.03"),
            "pyworld_silence_threshold_var": tk.StringVar(value="-40.0"),
            "pitch_split_threshold_ms_var": tk.StringVar(value="100"),
            "pitch_split_fluctuation_var": tk.StringVar(value="0.2"),
            "absorb_max_ms_var": tk.StringVar(value="100"),
            "enable_pitch_split_var": ctk.BooleanVar(value=False),
            
            "export_hybrid_var": ctk.BooleanVar(value=True),
            "export_w2v2_var": ctk.BooleanVar(value=False),
            "export_whisper_var": ctk.BooleanVar(value=False),
        }
        
        # 出力フォーマット
        self.fmt_vars = {
            "midi": ctk.BooleanVar(value=True),
            "musicxml": ctk.BooleanVar(value=False),
            "vsqx": ctk.BooleanVar(value=False),
            "ust": ctk.BooleanVar(value=False),
            "ccs": ctk.BooleanVar(value=False),
            "svp": ctk.BooleanVar(value=False)
        }
        
        self.current_notes = []
        
        self.create_widgets()
        
        # 標準出力と標準エラー出力をテキストボックスにリダイレクト
        sys.stdout = ThreadSafeTextRedirector(self.log_text)
        sys.stderr = ThreadSafeTextRedirector(self.log_text)
        
        self.load_settings()
        
        # モデルプリロードをバックグラウンドで実行
        threading.Thread(target=self._preload_models, daemon=True).start()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _preload_models(self):
        try:
            print("自然言語処理モデルのプリロードを開始します...")
            from core.text_utils import preload_models
            preload_models()
            print("自然言語処理モデルのプリロードが完了しました。")
        except Exception as e:
            print(f"プリロードエラー: {e}")

    def create_widgets(self):
        # 画面全体を上下に2分割 (上部=入力・設定・ログ, 下部=エディタ)
        self.root.grid_rowconfigure(0, weight=4) # Top Half
        self.root.grid_rowconfigure(1, weight=5) # Bottom Half (Editor)
        self.root.grid_columnconfigure(0, weight=1)
        
        top_half = ctk.CTkFrame(self.root, fg_color="transparent")
        top_half.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 5))
        
        bottom_half = ctk.CTkFrame(self.root, fg_color="transparent")
        bottom_half.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
        
        # Top Halfを左右に分割 (左=入力/歌詞等, 右=ログ/出力)
        top_half.grid_rowconfigure(0, weight=1)
        top_half.grid_columnconfigure(0, weight=5) # Left pane is wider
        top_half.grid_columnconfigure(1, weight=3) # Right pane
        
        left_pane = ctk.CTkFrame(top_half)
        left_pane.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        right_pane = ctk.CTkFrame(top_half)
        right_pane.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        self._build_left_pane(left_pane)
        self._build_right_pane(right_pane)
        self._build_bottom_half(bottom_half)

    def _build_left_pane(self, parent):
        # 入力ファイル・出力ファイル設定
        file_frame = ctk.CTkFrame(parent, fg_color="transparent")
        file_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        
        # Row 1: 入力音声
        in_frame = ctk.CTkFrame(file_frame, fg_color="transparent")
        in_frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(in_frame, text="入力音声:", width=80, anchor="e").pack(side=tk.LEFT, padx=5)
        ctk.CTkEntry(in_frame, textvariable=self.app_vars["audio_file_path"], state='readonly').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ctk.CTkButton(in_frame, text="参照...", command=self.browse_file, width=80).pack(side=tk.LEFT, padx=(0, 5))
        
        # Row 2: 出力ファイル
        out_frame = ctk.CTkFrame(file_frame, fg_color="transparent")
        out_frame.pack(fill=tk.X, pady=2)
        ctk.CTkLabel(out_frame, text="出力ファイル:", width=80, anchor="e").pack(side=tk.LEFT, padx=5)
        ctk.CTkEntry(out_frame, textvariable=self.app_vars["output_base_path"]).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ctk.CTkButton(out_frame, text="保存先...", command=self.browse_output, width=80).pack(side=tk.LEFT, padx=(0, 5))
        
        # 歌詞・設定オプション行
        lyrics_opt_frame = ctk.CTkFrame(parent, fg_color="transparent")
        lyrics_opt_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        
        ctk.CTkCheckBox(lyrics_opt_frame, text="歌詞を事前に入力する", variable=self.app_vars["use_predefined_lyrics_var"], command=self.on_use_predefined_lyrics_toggle).pack(side=tk.LEFT, padx=5)
        ctk.CTkCheckBox(lyrics_opt_frame, text="ボカロ語に変換", variable=self.app_vars["convert_to_vocaloid_var"], command=self.update_vocaloid_preview).pack(side=tk.LEFT, padx=15)

        
        # 設定ボタン
        ctk.CTkButton(lyrics_opt_frame, text="⚙ 設定", command=self.open_settings, width=60, fg_color="#444444", hover_color="#666666").pack(side=tk.RIGHT, padx=5)

        # 歌詞入力欄 (2分割)
        lyrics_container = ctk.CTkFrame(parent)
        lyrics_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 5))
        lyrics_container.grid_rowconfigure(1, weight=1)
        lyrics_container.grid_columnconfigure(0, weight=1)
        lyrics_container.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(lyrics_container, text="元の歌詞").grid(row=0, column=0, sticky="w", padx=5)
        ctk.CTkLabel(lyrics_container, text="変換プレビュー (編集不可)").grid(row=0, column=1, sticky="w", padx=5)
        
        self.predefined_lyrics_text = ctk.CTkTextbox(lyrics_container, state='disabled')
        self.predefined_lyrics_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        # Bind to internal textbox for sure key release capture
        self.predefined_lyrics_text._textbox.bind("<KeyRelease>", self.update_vocaloid_preview)
        
        self.vocaloid_lyrics_text = ctk.CTkTextbox(lyrics_container, state='disabled', fg_color="#333333")
        self.vocaloid_lyrics_text.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        # 検索・置換行
        search_frame = ctk.CTkFrame(parent, fg_color="transparent")
        search_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(search_frame, text="検索:").pack(side=tk.LEFT, padx=2)
        ctk.CTkEntry(search_frame, textvariable=self.app_vars["search_var"], width=100).pack(side=tk.LEFT, padx=2)
        
        ctk.CTkLabel(search_frame, text="置換:").pack(side=tk.LEFT, padx=(10, 2))
        ctk.CTkEntry(search_frame, textvariable=self.app_vars["replace_var"], width=100).pack(side=tk.LEFT, padx=2)
        
        ctk.CTkButton(search_frame, text="次を検索", width=70, fg_color="transparent", border_width=1, text_color=("black", "white"), command=self.find_next).pack(side=tk.LEFT, padx=2)
        ctk.CTkButton(search_frame, text="置換して次へ", width=90, fg_color="transparent", border_width=1, text_color=("black", "white"), command=self.replace_next).pack(side=tk.LEFT, padx=2)
        ctk.CTkButton(search_frame, text="すべて置換", width=80, fg_color="transparent", border_width=1, text_color=("black", "white"), command=self.replace_all).pack(side=tk.LEFT, padx=2)

        # 変換実行行
        conv_frame = ctk.CTkFrame(parent, fg_color="transparent")
        conv_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        
        ctk.CTkLabel(conv_frame, text="BPM (空欄で自動推定):").pack(side=tk.LEFT, padx=2)
        ctk.CTkEntry(conv_frame, textvariable=self.app_vars["tempo_var"], width=60).pack(side=tk.LEFT, padx=5)
        
        ctk.CTkCheckBox(conv_frame, text="変換前にボーカル抽出を行う(Demucs)", variable=self.app_vars["extract_vocals_var"]).pack(side=tk.LEFT, padx=15)
        
        self.convert_btn = ctk.CTkButton(conv_frame, text="データ変換", command=self.start_conversion, fg_color="transparent", border_width=2, hover_color="#222222", text_color=("black", "white"), font=("", 14, "bold"))
        self.convert_btn.pack(side=tk.RIGHT, padx=5)

    def _build_right_pane(self, parent):
        parent.grid_rowconfigure(0, weight=1) # ログ
        parent.grid_rowconfigure(1, weight=0) # 設定
        parent.grid_columnconfigure(0, weight=1)
        
        # 上部: 出力ログ
        log_frame = ctk.CTkFrame(parent, fg_color="transparent")
        log_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 5))
        
        ctk.CTkLabel(log_frame, text="出力ログ").pack(anchor="w")
        self.log_text = ctk.CTkTextbox(log_frame, state='disabled', fg_color="#1a1a1a")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # 下部: 出力設定とフォーマット
        out_set_frame = ctk.CTkFrame(parent)
        out_set_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
        
        out_set_frame.grid_columnconfigure(0, weight=1)
        out_set_frame.grid_columnconfigure(1, weight=1)
        
        # 出力設定 (左側)
        settings_side = ctk.CTkFrame(out_set_frame, fg_color="transparent")
        settings_side.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(settings_side, text="出力設定").pack(anchor="w")
        ctk.CTkCheckBox(settings_side, text="ダイナミクスを\n反映させる", variable=self.app_vars["reflect_dynamics_var"]).pack(anchor="w", pady=5)
        
        sens_frame = ctk.CTkFrame(settings_side, fg_color="transparent")
        sens_frame.pack(anchor="w")
        ctk.CTkLabel(sens_frame, text="感度(0-100):").pack(side=tk.LEFT)
        ctk.CTkEntry(sens_frame, textvariable=self.app_vars["dynamics_sensitivity_var"], width=40).pack(side=tk.LEFT, padx=5)
        
        # 出力フォーマット (右側)
        fmt_side = ctk.CTkFrame(out_set_frame, fg_color="transparent")
        fmt_side.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(fmt_side, text="出力フォーマット").pack(anchor="w")
        
        # 2列に分けて配置
        chk_container = ctk.CTkFrame(fmt_side, fg_color="transparent")
        chk_container.pack(fill=tk.X)
        
        col1 = ctk.CTkFrame(chk_container, fg_color="transparent")
        col1.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        col2 = ctk.CTkFrame(chk_container, fg_color="transparent")
        col2.pack(side=tk.LEFT, fill=tk.Y)
        
        formats = list(self.fmt_vars.items())
        for i, (fmt_name, var) in enumerate(formats):
            parent_col = col1 if i < 3 else col2
            ctk.CTkCheckBox(parent_col, text=fmt_name.upper(), variable=var).pack(anchor="w", pady=3)
            
        # ファイル出力ボタン
        self.export_btn = ctk.CTkButton(out_set_frame, text="ファイル出力", command=self.start_export, fg_color="transparent", border_width=2, hover_color="#222222", text_color=("black", "white"), font=("", 14, "bold"))
        self.export_btn.grid(row=1, column=0, columnspan=2, sticky="e", padx=10, pady=10)

    def _build_bottom_half(self, parent):
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        
        self.editor = VocalEditor(parent)
        self.editor.grid(row=0, column=0, sticky="nsew")

    def on_use_predefined_lyrics_toggle(self):
        if self.app_vars["use_predefined_lyrics_var"].get():
            self.predefined_lyrics_text.configure(state='normal')
        else:
            self.predefined_lyrics_text.configure(state='disabled')

    def update_vocaloid_preview(self, event=None):
        if not self.app_vars["use_predefined_lyrics_var"].get():
            return
            
        original_text = self.predefined_lyrics_text.get("1.0", tk.END).strip()
        if self.app_vars["convert_to_vocaloid_var"].get() and original_text:
            from core.text_utils import get_vocaloid_preview_text
            preview_text = get_vocaloid_preview_text(original_text)
        else:
            preview_text = original_text
            
        self.vocaloid_lyrics_text.configure(state='normal')
        self.vocaloid_lyrics_text.delete("1.0", tk.END)
        self.vocaloid_lyrics_text.insert("1.0", preview_text)
        self.vocaloid_lyrics_text.configure(state='disabled')

    def open_settings(self):
        SettingsWindow(self.root, self.app_vars)

    def _get_tk_text(self):
        # CTkTextbox内部のtk.Textウィジェットを取得
        return getattr(self.predefined_lyrics_text, "_textbox", self.predefined_lyrics_text)

    def find_next(self):
        search_str = self.app_vars["search_var"].get()
        if not search_str:
            return
            
        text_widget = self._get_tk_text()
        text_widget.tag_remove('search', '1.0', tk.END)
        
        start_pos = text_widget.index(tk.INSERT)
        pos = text_widget.search(search_str, start_pos, stopindex=tk.END)
        
        if not pos:
            # 最後までいったら最初から検索
            pos = text_widget.search(search_str, '1.0', stopindex=start_pos)
            
        if pos:
            end_pos = f"{pos}+{len(search_str)}c"
            text_widget.tag_add('search', pos, end_pos)
            text_widget.tag_config('search', background='yellow', foreground='black')
            text_widget.mark_set(tk.INSERT, end_pos)
            text_widget.see(tk.INSERT)
            self.predefined_lyrics_text.focus_set()
            text_widget.tag_remove("sel", "1.0", "end")
            text_widget.tag_add("sel", pos, end_pos)
        else:
            messagebox.showinfo("検索", f"「{search_str}」は見つかりませんでした。")

    def replace_next(self):
        if not self.app_vars["use_predefined_lyrics_var"].get():
            messagebox.showwarning("警告", "「歌詞を事前に入力する」にチェックを入れてください。")
            return
            
        search_str = self.app_vars["search_var"].get()
        replace_str = self.app_vars["replace_var"].get()
        if not search_str:
            return
            
        text_widget = self._get_tk_text()
        
        try:
            sel_start = text_widget.index("sel.first")
            sel_end = text_widget.index("sel.last")
            if text_widget.get(sel_start, sel_end) == search_str:
                text_widget.delete(sel_start, sel_end)
                text_widget.insert(sel_start, replace_str)
        except tk.TclError:
            pass
            
        self.find_next()

    def replace_all(self):
        if not self.app_vars["use_predefined_lyrics_var"].get():
            messagebox.showwarning("警告", "「歌詞を事前に入力する」にチェックを入れてください。")
            return
            
        search_str = self.app_vars["search_var"].get()
        replace_str = self.app_vars["replace_var"].get()
        if not search_str:
            return
            
        text_widget = self._get_tk_text()
        
        count = 0
        pos = '1.0'
        while True:
            pos = text_widget.search(search_str, pos, stopindex=tk.END)
            if not pos:
                break
            end_pos = f"{pos}+{len(search_str)}c"
            text_widget.delete(pos, end_pos)
            text_widget.insert(pos, replace_str)
            pos = f"{pos}+{len(replace_str)}c"
            count += 1
            
        if count > 0:
            messagebox.showinfo("置換完了", f"{count} 箇所の「{search_str}」を置換しました。")
        else:
            messagebox.showinfo("置換", f"「{search_str}」は見つかりませんでした。")

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="音声ファイルを選択",
            filetypes=(("Audio/Video files", "*.wav *.mp3 *.m4a *.mp4 *.flac *.ogg"), ("All files", "*.*"))
        )
        if filename:
            self.app_vars["audio_file_path"].set(filename)
            base_path = os.path.splitext(filename)[0]
            self.app_vars["output_base_path"].set(base_path)
            if hasattr(self, 'editor'):
                self.editor.set_data(self.current_notes, filename)

    def browse_output(self):
        current_path = self.app_vars["output_base_path"].get()
        initial_dir = os.path.dirname(current_path) if current_path else ""
        initial_file = os.path.basename(current_path) if current_path else ""
        
        filename = filedialog.asksaveasfilename(
            title="出力ベースパスを選択",
            initialdir=initial_dir,
            initialfile=initial_file,
            defaultextension=""
        )
        if filename:
            self.app_vars["output_base_path"].set(filename)

    def start_conversion(self):
        audio_file = self.app_vars["audio_file_path"].get()
        if not audio_file or not os.path.exists(audio_file):
            messagebox.showerror("エラー", "有効な音声ファイルを選択してください。")
            return
            
        self.convert_btn.configure(state="disabled")
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')
        print(f"データ変換処理を開始します: {audio_file}")
        
        threading.Thread(target=self._conversion_thread, daemon=True).start()
        
    def _conversion_thread(self):
        audio_file = self.app_vars["audio_file_path"].get()
        
        try:
            # 1. ボーカル抽出 (Demucs)
            if self.app_vars["extract_vocals_var"].get():
                print("Demucsによるボーカル抽出を実行中...")
                import subprocess
                out_dir = os.path.dirname(audio_file)
                cmd = ["demucs", "--two-stems=vocals", "-n", "htdemucs", "-o", out_dir, audio_file]
                try:
                    subprocess.run(cmd, check=True)
                    base_name = os.path.splitext(os.path.basename(audio_file))[0]
                    extracted_file = os.path.join(out_dir, "htdemucs", base_name, "vocals.wav")
                    if os.path.exists(extracted_file):
                        audio_file = extracted_file
                        print(f"ボーカル抽出完了: {audio_file}")
                    else:
                        print("ボーカル抽出ファイルの特定に失敗しました。元のファイルを使用します。")
                except Exception as e:
                    print(f"Demucsの実行に失敗しました: {e}\n元のファイルを使用します。")

            # 2. ピッチ・音量解析
            from core.audio_processor import get_pitch_contour, get_volume_contour
            
            frame_period = float(self.app_vars["frame_period_var"].get())
            f0_model = self.app_vars["f0_model_var"].get()
            pyworld_silence_threshold = float(self.app_vars["pyworld_silence_threshold_var"].get())
            
            time_array, midi_contour, confidence_array = get_pitch_contour(
                audio_file, 
                frame_period=frame_period, 
                f0_model=f0_model,
                pyworld_silence_threshold=pyworld_silence_threshold
            )
            
            volume_contour = None
            if self.app_vars["reflect_dynamics_var"].get():
                print("ダイナミクスを解析中...")
                sens = float(self.app_vars["dynamics_sensitivity_var"].get())
                volume_contour = get_volume_contour(audio_file, time_array, sensitivity=sens)

            # 3. テキスト抽出・変換
            from core.model_inference import load_wav2vec2_ctc_model, compute_forced_alignment, load_whisperx_models
            from core.text_utils import get_vocaloid_preview_text
            
            whisper_segments = []
            if self.app_vars["use_predefined_lyrics_var"].get():
                print("事前入力された歌詞を使用します...")
                raw_text = self.vocaloid_lyrics_text.get("1.0", tk.END).strip()
                whisper_segments = [{"text": raw_text}]
            else:
                print("WhisperXによる文字起こしを実行中...")
                w_model, w_model_a, w_metadata, w_device = load_whisperx_models(
                    whisper_model_name=self.app_vars["whisper_model_var"].get(),
                    w2v2_model_name=self.app_vars["w2v2_model_var"].get()
                )
                import whisperx
                audio = whisperx.load_audio(audio_file)
                result = w_model.transcribe(audio, batch_size=4)
                transcribed_text = "".join([seg["text"] for seg in result["segments"]])
                print(f"文字起こし結果: {transcribed_text}")
                
                vocaloid_text = get_vocaloid_preview_text(transcribed_text)
                print(f"ボカロ語変換後: {vocaloid_text}")
                whisper_segments = [{"text": vocaloid_text}]

            # 4. 強制アライメント
            print("Wav2Vec2による強制アライメントを実行中...")
            w2v2_model, processor, device = load_wav2vec2_ctc_model(
                w2v2_model_name=self.app_vars["w2v2_model_var"].get()
            )
            
            char_segments = compute_forced_alignment(
                audio_file, 
                whisper_segments, 
                w2v2_model, 
                processor, 
                device
            )
            
            if not char_segments:
                print("アライメントに失敗しました。")
                self.root.after(0, lambda: self.convert_btn.configure(state="normal"))
                return

            # 5. ノート生成
            print("ノートデータを生成中...")
            from core.alignment import segment_and_align_notes
            
            final_notes = segment_and_align_notes(
                char_segments=char_segments,
                time_array=time_array,
                midi_contour=midi_contour,
                confidence_array=confidence_array,
                min_duration=float(self.app_vars["min_duration_var"].get()),
                unvoiced_threshold_frames=int(self.app_vars["unvoiced_threshold_var"].get()),
                low_pitch_threshold=int(self.app_vars["low_pitch_threshold_var"].get()),
                low_pitch_drop_amount=int(self.app_vars["low_pitch_drop_amount_var"].get()),
                pitch_split_threshold_frames=int(self.app_vars["pitch_split_threshold_ms_var"].get()) // int(frame_period),
                pitch_split_fluctuation=float(self.app_vars["pitch_split_fluctuation_var"].get()),
                absorb_max_frames=int(self.app_vars["absorb_max_ms_var"].get()) // int(frame_period),
                enable_pitch_split=self.app_vars["enable_pitch_split_var"].get(),
                volume_contour=volume_contour
            )
            
            # 6. UI反映
            def on_complete():
                self.current_notes = final_notes
                self.editor.set_data(self.current_notes, self.app_vars["audio_file_path"].get())
                self.convert_btn.configure(state="normal")
                print("変換完了。下部のエディタで確認・編集してください。")
                
            self.root.after(0, on_complete)
            
        except Exception as e:
            import traceback
            print(f"エラーが発生しました: {e}")
            traceback.print_exc()
            self.root.after(0, lambda: self.convert_btn.configure(state="normal"))

    def start_export(self):
        if not self.current_notes:
            messagebox.showerror("エラー", "エクスポートするデータがありません。先にデータ変換を行ってください。")
            return
            
        output_base = self.app_vars["output_base_path"].get().strip()
        if not output_base:
            messagebox.showerror("エラー", "出力ベースパスを指定してください。")
            return
            
        self.export_btn.configure(state="disabled")
        self.log_text.configure(state='normal')
        
        threading.Thread(target=self._export_thread, daemon=True).start()

    def _export_thread(self):
        try:
            output_base = self.app_vars["output_base_path"].get().strip()
            
            tempo_str = self.app_vars["tempo_var"].get().strip()
            if tempo_str:
                try:
                    tempo = float(tempo_str)
                except ValueError:
                    print("BPMのパースに失敗しました。デフォルトの120を使用します。")
                    tempo = 120
            else:
                audio_file = self.app_vars["audio_file_path"].get()
                if audio_file and os.path.exists(audio_file):
                    from export.exporters import estimate_tempo
                    tempo = estimate_tempo(audio_file)
                else:
                    tempo = 120
                    
            from export.exporters import export_to_midi, export_to_musicxml, export_to_vsqx, export_to_ust, export_to_ccs, export_to_svp
            
            print(f"エクスポートを開始します... ベースパス: {output_base}")
            
            if self.fmt_vars["midi"].get():
                export_to_midi([dict(n) for n in self.current_notes], f"{output_base}.mid", tempo=tempo)
            if self.fmt_vars["musicxml"].get():
                export_to_musicxml([dict(n) for n in self.current_notes], f"{output_base}.musicxml", tempo=tempo)
            if self.fmt_vars["vsqx"].get():
                export_to_vsqx([dict(n) for n in self.current_notes], f"{output_base}.vsqx", tempo=tempo)
            if self.fmt_vars["ust"].get():
                export_to_ust([dict(n) for n in self.current_notes], f"{output_base}.ust", tempo=tempo)
            if self.fmt_vars["ccs"].get():
                export_to_ccs([dict(n) for n in self.current_notes], f"{output_base}.ccs", tempo=tempo)
            if self.fmt_vars["svp"].get():
                export_to_svp([dict(n) for n in self.current_notes], f"{output_base}.svp", tempo=tempo)
                
            print("すべてのエクスポート処理が完了しました。")
            
        except Exception as e:
            import traceback
            print(f"エクスポート中にエラーが発生しました: {e}")
            traceback.print_exc()
        finally:
            self.root.after(0, lambda: self.export_btn.configure(state="normal"))
            self.root.after(0, lambda: self.log_text.configure(state='disabled'))

    def load_settings(self):
        if os.path.exists('settings.json'):
            try:
                with open('settings.json', 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    for key, val in settings.items():
                        if key in self.app_vars:
                            try:
                                self.app_vars[key].set(val)
                            except:
                                pass
            except Exception as e:
                print(f"Failed to load settings: {e}")
        
        # UI状態の更新を強制
        self.on_use_predefined_lyrics_toggle()
        self.update_vocaloid_preview()

    def on_closing(self):
        settings = {k: v.get() for k, v in self.app_vars.items()}
        try:
            with open('settings.json', 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Failed to save settings: {e}")
        self.root.destroy()
