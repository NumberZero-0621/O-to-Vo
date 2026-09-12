import os
import warnings
import customtkinter as ctk

# ==========================================
# ログ・警告の抑制設定
# ==========================================
# TensorFlowおよびoneDNNの情報ログ・警告を抑制
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# PyannoteやTransformersからのUserWarning等を無視
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="pyannote")

from ui.main_window import OToVoApp

if __name__ == "__main__":
    ctk.set_appearance_mode("Dark")  # テーマをダークに設定
    ctk.set_default_color_theme("blue")  # テーマカラーを設定
    root = ctk.CTk()
    app = OToVoApp(root)
    root.mainloop()