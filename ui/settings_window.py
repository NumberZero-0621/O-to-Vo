import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk

class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent, app_vars):
        super().__init__(parent)
        self.title("設定")
        self.geometry("500x600")
        self.resizable(False, False)
        
        # モーダル化
        self.transient(parent)
        self.grab_set()
        
        self.app_vars = app_vars
        self.create_widgets()
        
    def create_widgets(self):
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tabview = ctk.CTkTabview(main_frame)
        tabview.pack(fill=tk.BOTH, expand=True)
        
        tabview.add("モデル設定")
        tabview.add("詳細設定")
        tabview.add("出力する認識データ")
        
        # モデル設定タブ
        model_frame = tabview.tab("モデル設定")
        
        ctk.CTkLabel(model_frame, text="WhisperX (歌詞認識):").grid(row=0, column=0, sticky="w", pady=(10, 5), padx=5)
        whisper_models = ["large-v3", "large-v2", "medium", "small", "base", "tiny", "URLを指定してモデル追加"]
        self.whisper_cb = ctk.CTkComboBox(model_frame, variable=self.app_vars["whisper_model_var"], values=whisper_models, width=250)
        self.whisper_cb.grid(row=0, column=1, sticky="w", padx=5, pady=(10, 5))
        # CTkComboBox does not have bind("<<ComboboxSelected>>") directly, we use command
        self.whisper_cb.configure(command=self.on_model_select)
        
        ctk.CTkLabel(model_frame, text="Wav2Vec2 (タイミング):").grid(row=1, column=0, sticky="w", pady=5, padx=5)
        w2v2_models = ["vumichien/wav2vec2-large-xlsr-japanese-hiragana", "jonatasgrosman/wav2vec2-large-xlsr-53-japanese", "URLを指定してモデル追加"]
        self.w2v2_cb = ctk.CTkComboBox(model_frame, variable=self.app_vars["w2v2_model_var"], values=w2v2_models, width=250)
        self.w2v2_cb.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.w2v2_cb.configure(command=self.on_model_select)
        
        ctk.CTkLabel(model_frame, text="F0推定 (ピッチ):").grid(row=2, column=0, sticky="w", pady=5, padx=5)
        f0_models = ["PyWorld", "CREPE"]
        f0_cb = ctk.CTkComboBox(model_frame, variable=self.app_vars["f0_model_var"], values=f0_models, width=250)
        f0_cb.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # 詳細設定タブ
        advanced_frame = tabview.tab("詳細設定")
        
        params = [
            ("休符判定フレーム数:", "unvoiced_threshold_var"),
            ("ピッチ解析間隔(ms):", "frame_period_var"),
            ("低音ノイズ削除閾値(MIDI):", "low_pitch_threshold_var"),
            ("低音ノイズ落差(半音):", "low_pitch_drop_amount_var"),
            ("無音分割閾値(top_db):", "top_db_var"),
            ("文字スキップコスト:", "skip_b_cost_var"),
            ("最終モーラ時間比率:", "last_mora_ratio_var"),
            ("最小ノート長(秒):", "min_duration_var"),
            ("PyWorld有声閾値(dB):", "pyworld_silence_threshold_var"),
            ("ピッチ分割閾値(ms):", "pitch_split_threshold_ms_var"),
            ("ピッチ分割変動幅(半音):", "pitch_split_fluctuation_var"),
            ("吸収最大ノート長(ms):", "absorb_max_ms_var"),
        ]
        
        for i, (label, var_name) in enumerate(params):
            ctk.CTkLabel(advanced_frame, text=label).grid(row=i, column=0, sticky="w", pady=2, padx=10)
            ctk.CTkEntry(advanced_frame, textvariable=self.app_vars[var_name], width=80).grid(row=i, column=1, sticky="w", padx=5, pady=2)
            
        ctk.CTkCheckBox(advanced_frame, text="ピッチ推移による自動ノート分割・結合を有効にする", 
                        variable=self.app_vars["enable_pitch_split_var"]).grid(row=len(params), column=0, columnspan=2, sticky="w", pady=15, padx=10)
        
        # 出力ソースタブ
        export_frame = tabview.tab("出力する認識データ")
        
        ctk.CTkCheckBox(export_frame, text="Hybrid (推奨)", variable=self.app_vars["export_hybrid_var"]).pack(anchor="w", pady=10, padx=10)
        ctk.CTkCheckBox(export_frame, text="Wav2Vec2", variable=self.app_vars["export_w2v2_var"]).pack(anchor="w", pady=10, padx=10)
        ctk.CTkCheckBox(export_frame, text="WhisperX", variable=self.app_vars["export_whisper_var"]).pack(anchor="w", pady=10, padx=10)
        
        # 閉じるボタン
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill=tk.X, pady=10)
        ctk.CTkButton(btn_frame, text="閉じる", command=self.destroy, width=120).pack()

    def on_model_select(self, choice):
        # determine which cb triggered this by checking their values
        if choice == "URLを指定してモデル追加":
            if self.whisper_cb.get() == "URLを指定してモデル追加":
                self.ask_custom_model_url(self.whisper_cb)
            elif self.w2v2_cb.get() == "URLを指定してモデル追加":
                self.ask_custom_model_url(self.w2v2_cb)
            
    def ask_custom_model_url(self, cb):
        top = ctk.CTkToplevel(self)
        top.title("カスタムモデルの追加")
        top.geometry("400x200")
        top.transient(self)
        top.grab_set()
        
        ctk.CTkLabel(top, text="Hugging FaceのURLを入力してください:\n(例: https://huggingface.co/author/model)").pack(pady=15)
        url_var = tk.StringVar()
        entry = ctk.CTkEntry(top, textvariable=url_var, width=300)
        entry.pack(padx=10, pady=5)
        
        def submit():
            url = url_var.get().strip()
            import re
            match = re.search(r"huggingface\.co/([^/]+/[^/]+)", url)
            if match:
                model_name = match.group(1)
                model_name = model_name.split('/tree')[0]
                model_name = model_name.split('?')[0]
                
                values = list(cb.cget("values"))
                values.insert(-1, model_name)
                cb.configure(values=values)
                cb.set(model_name)
                top.destroy()
            else:
                messagebox.showerror("エラー", "正しいHugging FaceのURLを入力してください。", parent=top)
                
        def cancel():
            cb.set(cb.cget("values")[0])
            top.destroy()
            
        btn_frame = ctk.CTkFrame(top, fg_color="transparent")
        btn_frame.pack(pady=15)
        ctk.CTkButton(btn_frame, text="追加", command=submit, width=100).pack(side=tk.LEFT, padx=10)
        ctk.CTkButton(btn_frame, text="キャンセル", command=cancel, width=100, fg_color="#555555").pack(side=tk.LEFT, padx=10)
