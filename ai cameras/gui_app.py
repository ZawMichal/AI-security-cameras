"""
Face ID Application - Modern GUI
Design implementation using customtkinter
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import threading
import time
import datetime
import csv
from pathlib import Path
import sys
import traceback

# AI Core imports - te będą działać gdy skopiujesz folder ai_core
try:
    from ai_core.engine import AIEngine
    from ai_core.config import config
    from ai_core.performance_monitor import PerformanceMonitor
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("[WARNING] ai_core not found - running in UI-only mode")

# Konfiguracja customtkinter
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# Kolory z designu
COLORS = {
    'bg': '#FFFFFF',
    'dark_btn': '#2D2D2D',
    'dark_btn_hover': '#404040',
    'accent_red': '#8B2332',
    'text_dark': '#2D2D2D',
    'text_gray': '#7F7F7F',
    'border': '#E0E0E0',
    'nav_active_bg': '#F5F5F5',
}


class SettingsWindow(ctk.CTkToplevel):
    """Okno ustawień"""
    def __init__(self, parent, engine, main_app):
        super().__init__(parent)
        self.engine = engine
        self.main_app = main_app

        self.title("Settings")
        self.geometry("500x550")
        self.resizable(False, False)

        # Center on parent
        self.transient(parent)
        self.grab_set()

        self.create_tabs()

    def create_tabs(self):
        # Tabview
        self.tabview = ctk.CTkTabview(self, width=480, height=450)
        self.tabview.pack(padx=10, pady=10, fill="both", expand=True)

        # Tabs
        self.tabview.add("GPU")
        self.tabview.add("Performance")
        self.tabview.add("Detection")

        self.create_gpu_tab()
        self.create_performance_tab()
        self.create_detection_tab()

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkButton(btn_frame, text="Apply", command=self.apply_settings,
                     fg_color="#27ae60", hover_color="#219a52", width=100).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Close", command=self.destroy,
                     fg_color="#95a5a6", hover_color="#7f8c8d", width=100).pack(side="left", padx=5)

    def create_gpu_tab(self):
        tab = self.tabview.tab("GPU")

        # Precision Mode
        ctk.CTkLabel(tab, text="GPU Precision Mode", font=("Arial", 14, "bold")).pack(anchor="w", pady=(10,5))

        if AI_AVAILABLE:
            self.quantization_var = ctk.StringVar(value=config.quantization_mode)
        else:
            self.quantization_var = ctk.StringVar(value="fp16")

        for text, value in [("FP32 (Full Precision)", "fp32"),
                           ("FP16 (Half Precision) - Recommended", "fp16"),
                           ("INT8 (8-bit Quantization)", "int8")]:
            ctk.CTkRadioButton(tab, text=text, variable=self.quantization_var, value=value).pack(anchor="w", pady=2)

        # GPU Features
        ctk.CTkLabel(tab, text="GPU Features", font=("Arial", 14, "bold")).pack(anchor="w", pady=(20,5))

        if AI_AVAILABLE:
            self.tensor_cores_var = ctk.BooleanVar(value=config.use_tensor_cores)
            self.optimize_mem_var = ctk.BooleanVar(value=config.optimize_memory)
        else:
            self.tensor_cores_var = ctk.BooleanVar(value=True)
            self.optimize_mem_var = ctk.BooleanVar(value=False)

        ctk.CTkCheckBox(tab, text="Enable Tensor Cores", variable=self.tensor_cores_var).pack(anchor="w", pady=2)
        ctk.CTkCheckBox(tab, text="Optimize Memory Layout", variable=self.optimize_mem_var).pack(anchor="w", pady=2)

    def create_performance_tab(self):
        tab = self.tabview.tab("Performance")

        ctk.CTkLabel(tab, text="Performance Monitoring", font=("Arial", 14, "bold")).pack(anchor="w", pady=(10,5))

        if AI_AVAILABLE:
            self.show_metrics_var = ctk.BooleanVar(value=config.show_performance_metrics)
            self.graph_mode_var = ctk.StringVar(value=config.graph_mode)
        else:
            self.show_metrics_var = ctk.BooleanVar(value=True)
            self.graph_mode_var = ctk.StringVar(value="off")

        ctk.CTkCheckBox(tab, text="Show Performance Metrics", variable=self.show_metrics_var).pack(anchor="w", pady=2)

        ctk.CTkLabel(tab, text="Graph Mode", font=("Arial", 14, "bold")).pack(anchor="w", pady=(20,5))

        for text, value in [("Off", "off"), ("Compact", "compact"), ("Full", "full")]:
            ctk.CTkRadioButton(tab, text=text, variable=self.graph_mode_var, value=value).pack(anchor="w", pady=2)

    def create_detection_tab(self):
        tab = self.tabview.tab("Detection")

        ctk.CTkLabel(tab, text="Detection Thresholds", font=("Arial", 14, "bold")).pack(anchor="w", pady=(10,5))

        # YOLO threshold
        frame1 = ctk.CTkFrame(tab, fg_color="transparent")
        frame1.pack(fill="x", pady=5)
        ctk.CTkLabel(frame1, text="YOLO confidence:").pack(side="left")

        if AI_AVAILABLE:
            self.yolo_conf_var = ctk.DoubleVar(value=config.yolo_confidence_threshold)
            self.face_det_var = ctk.DoubleVar(value=config.detection_threshold)
            self.face_conf_var = ctk.DoubleVar(value=config.confidence_threshold)
            self.face_topk_var = ctk.IntVar(value=getattr(config, 'face_top_k', 1))
        else:
            self.yolo_conf_var = ctk.DoubleVar(value=0.5)
            self.face_det_var = ctk.DoubleVar(value=0.9)
            self.face_conf_var = ctk.DoubleVar(value=0.6)
            self.face_topk_var = ctk.IntVar(value=1)

        ctk.CTkEntry(frame1, textvariable=self.yolo_conf_var, width=80).pack(side="right")

        # Face detection threshold
        frame2 = ctk.CTkFrame(tab, fg_color="transparent")
        frame2.pack(fill="x", pady=5)
        ctk.CTkLabel(frame2, text="Face detection:").pack(side="left")
        ctk.CTkEntry(frame2, textvariable=self.face_det_var, width=80).pack(side="right")

        # Face recognition threshold
        frame3 = ctk.CTkFrame(tab, fg_color="transparent")
        frame3.pack(fill="x", pady=5)
        ctk.CTkLabel(frame3, text="Face recognition:").pack(side="left")
        ctk.CTkEntry(frame3, textvariable=self.face_conf_var, width=80).pack(side="right")

        # Top-K
        frame4 = ctk.CTkFrame(tab, fg_color="transparent")
        frame4.pack(fill="x", pady=5)
        ctk.CTkLabel(frame4, text="Top labels (1-3):").pack(side="left")
        ctk.CTkEntry(frame4, textvariable=self.face_topk_var, width=80).pack(side="right")

    def apply_settings(self):
        if AI_AVAILABLE:
            new_quantization = self.quantization_var.get()
            config.show_performance_metrics = self.show_metrics_var.get()
            config.use_tensor_cores = self.tensor_cores_var.get()
            config.optimize_memory = self.optimize_mem_var.get()
            config.graph_mode = self.graph_mode_var.get()

            config.yolo_confidence_threshold = float(self.yolo_conf_var.get())
            config.detection_threshold = float(self.face_det_var.get())
            config.confidence_threshold = float(self.face_conf_var.get())
            config.face_top_k = int(self.face_topk_var.get())

            if new_quantization != config.quantization_mode:
                config.quantization_mode = new_quantization
                self.main_app._log(f"Switching to {new_quantization}...")

                def reload_faces():
                    try:
                        self.engine._face_system = None
                        _ = self.engine.face_system
                        self.main_app._log(f"Face system reloaded")
                    except Exception as e:
                        self.main_app._log(f"Error: {e}")

                threading.Thread(target=reload_faces, daemon=True).start()

        self.destroy()


class FaceIDApp(ctk.CTk):
    """Główna aplikacja Face ID"""

    def __init__(self):
        super().__init__()

        # Konfiguracja okna
        self.title("Face ID System")
        self.configure(fg_color=COLORS['bg'])

        # Własny pasek tytułowy z custom guzikami
        self.overrideredirect(True)
        self.use_custom_window_controls = True

        # Ustaw rozmiar i wycentruj na ekranie
        window_width = 1350
        window_height = 810
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Zmienne do przeciągania okna
        self._drag_x = 0
        self._drag_y = 0
        self._is_maximized = False
        self._normal_geometry = f"{window_width}x{window_height}+{x}+{y}"

        # Obsługa przywrócenia okna po minimalizacji
        self.bind("<Map>", self._on_map)

        # Minimalne wymiary
        self.minsize(1350, 810)
        self.maxsize(1350, 810)

        # Stan aplikacji
        self.current_tab = "START"
        self.is_running = True
        self.cap = None
        self.recording_mode = False
        self.target_person_name = ""
        self.preview_window = None
        self.preview_label = None
        self.preview_width = 640
        self.preview_height = 480
        self._camera_connecting = False
        self.recording_active = False
        self.video_writer = None
        self.current_evidence_file = None

        # Performance
        if AI_AVAILABLE:
            self.perf_monitor = PerformanceMonitor()
        else:
            self.perf_monitor = None

        self.frame_skip_counter = 0
        self.FRAME_SKIP = 2
        self._metrics_counter = 0

        # Inicjalizacja AI Engine PRZED create_ui
        if AI_AVAILABLE:
            self._log("Initializing AI Engine...")
            self.engine = AIEngine(log_callback=self._log)
            self._log("AI Engine ready")
            self.engine.enable_yolo = False
            self.engine.enable_faces = False
        else:
            self.engine = None
            self._log("Running in UI-only mode (ai_core not found)")

        # Ładowanie zasobów graficznych
        self.load_assets()

        # Przygotuj katalogi/logi detekcji
        self._ensure_detection_log()

        # Tworzenie UI
        self.create_ui()

        # Odśwież listę osób
        if AI_AVAILABLE and self.engine:
            self._refresh_db_list()

        # Start głównej pętli
        self._update_frame()

    def load_assets(self):
        """Ładuje zasoby graficzne"""
        assets_dir = Path(__file__).parent / "assets"

        self.assets = {}

        # Próbuj załadować grafiki
        try:
            if (assets_dir / "left_artwork.png").exists():
                img = Image.open(assets_dir / "left_artwork.png")
                # Skaluj proporcjonalnie do okna
                scale = 0.7
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.LANCZOS)
                self.assets['left_artwork'] = ctk.CTkImage(light_image=img, size=new_size)

            if (assets_dir / "right_lines.png").exists():
                img = Image.open(assets_dir / "right_lines.png")
                scale = 0.85  # Lekko zmniejszone żeby pasowało
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.LANCZOS)
                self.assets['right_lines'] = ctk.CTkImage(light_image=img, size=new_size)

            if (assets_dir / "corner_circle.png").exists():
                img = Image.open(assets_dir / "corner_circle.png")
                scale = 0.6
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.LANCZOS)
                self.assets['corner_circle'] = ctk.CTkImage(light_image=img, size=new_size)

            if (assets_dir / "avatar.png").exists():
                img = Image.open(assets_dir / "avatar.png")
                # Avatar jako okrągły
                img = img.resize((40, 40), Image.LANCZOS)
                self.assets['avatar'] = ctk.CTkImage(light_image=img, size=(40, 40))
        except Exception as e:
            print(f"Could not load some assets: {e}")

    def create_ui(self):
        """Tworzy główny interfejs"""

        # Główny kontener
        self.main_container = ctk.CTkFrame(self, fg_color=COLORS['bg'])
        self.main_container.pack(fill="both", expand=True)

        # Header z nawigacją
        self.create_header()

        # Obszar zawartości
        self.content_frame = ctk.CTkFrame(self.main_container, fg_color=COLORS['bg'])
        self.content_frame.pack(fill="both", expand=True, padx=0, pady=0)

        # Kontener na zawartość zakładek - NAJPIERW (pod spodem)
        self.tab_content = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.tab_content.place(relx=0.32, rely=0.08, relwidth=0.55, relheight=0.88)

        # Dekoracyjne grafiki - NA KOŃCU (na wierzchu)
        self.create_decorations()

        # Pokaż domyślną zakładkę
        self.show_tab("START")

    def create_header(self):
        """Tworzy header z nawigacją i przyciskami okna"""
        header = ctk.CTkFrame(self.main_container, fg_color=COLORS['bg'], height=80)
        header.pack(fill="x", padx=0, pady=(0, 0))
        header.pack_propagate(False)

        # Subtelny cień / linia pod headerem
        shadow = ctk.CTkFrame(header, height=1, fg_color="#E3E3E3")
        shadow.pack(side="bottom", fill="x")

        # Umożliwienie przeciągania okna za header
        header.bind("<Button-1>", self._start_drag)
        header.bind("<B1-Motion>", self._do_drag)

        # Avatar i ADMIN (lewa strona)
        user_frame = ctk.CTkFrame(header, fg_color="transparent")
        user_frame.pack(side="left", padx=(30, 0), pady=20)
        user_frame.bind("<Button-1>", self._start_drag)
        user_frame.bind("<B1-Motion>", self._do_drag)

        if 'avatar' in self.assets:
            avatar_label = ctk.CTkLabel(user_frame, image=self.assets['avatar'], text="")
            avatar_label.pack(side="left", padx=(0, 10))
        else:
            # Placeholder dla avatara
            avatar_canvas = tk.Canvas(user_frame, width=40, height=40, bg=COLORS['bg'], highlightthickness=0)
            avatar_canvas.create_oval(2, 2, 38, 38, fill="#DDD", outline="#BBB")
            avatar_canvas.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(user_frame, text="ADMIN", font=("Arial", 14, "bold"),
                    text_color=COLORS['text_dark']).pack(side="left")

        # Nawigacja (środek-lewo)
        nav_frame = ctk.CTkFrame(header, fg_color="transparent")
        nav_frame.pack(side="left", padx=(60, 0), pady=20)

        self.nav_buttons = {}
        for tab_name in ["START", "OPTIONS", "HELP", "WEBSITE"]:
            btn = ctk.CTkButton(
                nav_frame,
                text=tab_name,
                font=("Arial", 13),
                fg_color="transparent" if tab_name != "START" else COLORS['nav_active_bg'],
                text_color=COLORS['text_dark'],
                hover_color=COLORS['nav_active_bg'],
                width=100,
                height=35,
                corner_radius=5,
                command=lambda t=tab_name: self.show_tab(t)
            )
            btn.pack(side="left", padx=5)
            self.nav_buttons[tab_name] = btn

        # Przyciski okna (prawa strona) - NA TEJ SAMEJ WYSOKOŚCI co nawigacja
        if self.use_custom_window_controls:
            window_btns = ctk.CTkFrame(header, fg_color="transparent")
            window_btns.pack(side="right", padx=(0, 30), pady=20)

            for symbol, cmd in [("—", self._minimize_window), ("□", self._toggle_maximize), ("✕", self.on_close)]:
                btn = ctk.CTkButton(
                    window_btns, text=symbol, width=40, height=35,
                    fg_color="transparent", text_color=COLORS['text_dark'],
                    hover_color="#EEE", font=("Arial", 16),
                    corner_radius=5,
                    command=cmd
                )
                btn.pack(side="left", padx=2)

    def _start_drag(self, event):
        """Rozpoczyna przeciąganie okna"""
        self._drag_x = event.x
        self._drag_y = event.y

    def _do_drag(self, event):
        """Przeciąga okno"""
        x = self.winfo_x() + event.x - self._drag_x
        y = self.winfo_y() + event.y - self._drag_y
        self.geometry(f"+{x}+{y}")

    def _minimize_window(self):
        """Minimalizuje okno"""
        self.overrideredirect(False)
        self.iconify()

    def _toggle_maximize(self):
        """Przełącza maksymalizację"""
        if self._is_maximized:
            self.geometry(self._normal_geometry)
            self._is_maximized = False
        else:
            self._normal_geometry = self.geometry()
            self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}+0+0")
            self._is_maximized = True

    def _on_map(self, event):
        """Przywraca overrideredirect po minimalizacji"""
        if self.use_custom_window_controls:
            self.overrideredirect(True)

    def create_decorations(self):
        """Tworzy dekoracyjne elementy graficzne"""

        # Lewa grafika (sylwetka)
        if 'left_artwork' in self.assets:
            left_label = ctk.CTkLabel(self.content_frame, image=self.assets['left_artwork'], text="")
            left_label.place(relx=0, rely=0.3, anchor="nw")

        # Duże czerwone półkole w prawym górnym rogu - tylko dekoracja (bez przycisków)
        try:
            red_corner_path = Path(__file__).parent / "assets" / "red_corner.png"
            if red_corner_path.exists():
                red_img = Image.open(red_corner_path)
                self.red_corner_ctk = ctk.CTkImage(light_image=red_img, size=(280, 150))

                corner_label = ctk.CTkLabel(self.content_frame, image=self.red_corner_ctk,
                                           text="", fg_color="transparent")
                corner_label.place(relx=1, rely=0, anchor="ne")
        except Exception as e:
            print(f"Could not load red corner: {e}")

        # Prawa grafika (faliste linie) - NA WIERZCHU (po czerwonym półkolu)
        if 'right_lines' in self.assets:
            right_label = ctk.CTkLabel(self.content_frame, image=self.assets['right_lines'],
                                       text="", fg_color="transparent", bg_color="transparent")
            right_label.place(relx=1, rely=1, anchor="se", x=50)
            right_label.lift()  # Podnieś na wierzch

    def show_tab(self, tab_name):
        """Przełącza zakładkę"""
        self.current_tab = tab_name

        # Aktualizuj wygląd przycisków nawigacji
        for name, btn in self.nav_buttons.items():
            if name == tab_name:
                btn.configure(fg_color=COLORS['nav_active_bg'])
            else:
                btn.configure(fg_color="transparent")

        # Wyczyść zawartość
        for widget in self.tab_content.winfo_children():
            widget.destroy()

        # Pokaż odpowiednią zawartość
        if tab_name == "START":
            self.create_start_tab()
        elif tab_name == "OPTIONS":
            self.create_options_tab()
        elif tab_name == "HELP":
            self.create_help_tab()
        elif tab_name == "WEBSITE":
            self.create_website_tab()

    def create_start_tab(self):
        """Zakładka START - główne kontrolki z wyrównanymi przyciskami"""

        # Główny kontener z grid (kontrolki)
        grid_frame = ctk.CTkFrame(self.tab_content, fg_color="transparent")
        grid_frame.pack(fill="both", expand=True, padx=20)

        row = 0

        # ===== CAMERA SOURCE =====
        ctk.CTkLabel(grid_frame, text="CAMERA SOURCE", font=("Arial", 12),
                    text_color=COLORS['text_dark']).grid(row=row, column=0, sticky="w", pady=(0, 5))
        row += 1

        # Dropdown + przyciski w jednym wierszu
        self.camera_var = ctk.StringVar(value="0")
        self.camera_dropdown = ctk.CTkComboBox(
            grid_frame, values=["0", "1", "2", "3"],
            variable=self.camera_var, width=150,
            fg_color="white", border_color=COLORS['border'],
            button_color=COLORS['dark_btn'], button_hover_color=COLORS['dark_btn_hover']
        )
        self.camera_dropdown.grid(row=row, column=0, sticky="w", padx=(0, 15))

        ctk.CTkButton(grid_frame, text="CONNECT", command=self._connect_camera,
                     fg_color=COLORS['dark_btn'], hover_color=COLORS['dark_btn_hover'],
                     width=110, height=38, corner_radius=20).grid(row=row, column=1, padx=5)

        ctk.CTkButton(grid_frame, text="SETTINGS", command=self._open_settings,
                     fg_color=COLORS['dark_btn'], hover_color=COLORS['dark_btn_hover'],
                     width=110, height=38, corner_radius=20).grid(row=row, column=2, padx=5)
        row += 1

        # Spacer
        ctk.CTkLabel(grid_frame, text="").grid(row=row, column=0, pady=8)
        row += 1

        # ===== YOLO =====
        ctk.CTkLabel(grid_frame, text="YOLO", font=("Arial", 12),
                    text_color=COLORS['text_dark']).grid(row=row, column=0, sticky="w")

        self.btn_yolo_run = ctk.CTkButton(
            grid_frame, text="RUN", command=self._start_yolo,
            fg_color=COLORS['dark_btn'], hover_color=COLORS['dark_btn_hover'],
            width=110, height=38, corner_radius=20
        )
        self.btn_yolo_run.grid(row=row, column=1, padx=5)

        self.btn_yolo_stop = ctk.CTkButton(
            grid_frame, text="STOP", command=self._stop_yolo,
            fg_color=COLORS['dark_btn'], hover_color=COLORS['dark_btn_hover'],
            width=110, height=38, corner_radius=20, state="disabled"
        )
        self.btn_yolo_stop.grid(row=row, column=2, padx=5)
        row += 1

        # Spacer
        ctk.CTkLabel(grid_frame, text="").grid(row=row, column=0, pady=5)
        row += 1

        # ===== FACES RECOGNITION =====
        ctk.CTkLabel(grid_frame, text="FACES RECOGNITION", font=("Arial", 12),
                    text_color=COLORS['text_dark']).grid(row=row, column=0, sticky="w")

        self.btn_face_run = ctk.CTkButton(
            grid_frame, text="RUN", command=self._start_face,
            fg_color=COLORS['dark_btn'], hover_color=COLORS['dark_btn_hover'],
            width=110, height=38, corner_radius=20
        )
        self.btn_face_run.grid(row=row, column=1, padx=5)

        self.btn_face_stop = ctk.CTkButton(
            grid_frame, text="STOP", command=self._stop_face,
            fg_color=COLORS['dark_btn'], hover_color=COLORS['dark_btn_hover'],
            width=110, height=38, corner_radius=20, state="disabled"
        )
        self.btn_face_stop.grid(row=row, column=2, padx=5)
        row += 1

        # Black list filter checkbox
        self.blacklist_filter_var = ctk.BooleanVar(value=True)
        checkbox_frame = ctk.CTkFrame(grid_frame, fg_color="transparent")
        checkbox_frame.grid(row=row, column=0, columnspan=3, sticky="w", pady=(3, 0))

        ctk.CTkCheckBox(checkbox_frame, text="Black list filter",
                       variable=self.blacklist_filter_var,
                       command=self._update_face_opts,
                       text_color=COLORS['text_dark']).pack(anchor="w")
        ctk.CTkLabel(checkbox_frame, text="Description", font=("Arial", 10),
                    text_color=COLORS['text_gray']).pack(anchor="w", padx=(25, 0))
        row += 1

        # Spacer
        ctk.CTkLabel(grid_frame, text="").grid(row=row, column=0, pady=5)
        row += 1

        # ===== RECORDING =====
        ctk.CTkLabel(grid_frame, text="RECORDING", font=("Arial", 12),
                    text_color=COLORS['text_dark']).grid(row=row, column=0, sticky="w", pady=(0, 5))
        row += 1

        self.btn_recording = ctk.CTkButton(
            grid_frame, text="RECORDING OFF", command=self._toggle_recording,
            fg_color="#8B2332", hover_color="#6f1c28",
            width=235, height=38, corner_radius=20
        )
        self.btn_recording.grid(row=row, column=0, columnspan=3, sticky="w")
        row += 1

        # Spacer
        ctk.CTkLabel(grid_frame, text="").grid(row=row, column=0, pady=5)
        row += 1

        # ===== CAPTURE FACES =====
        ctk.CTkLabel(grid_frame, text="CAPTURE FACES", font=("Arial", 12),
                    text_color=COLORS['text_dark']).grid(row=row, column=0, sticky="w", pady=(0, 5))
        row += 1

        self.capture_name_entry = ctk.CTkEntry(
            grid_frame, placeholder_text="Enter name...",
            width=150, fg_color="white", border_color=COLORS['border']
        )
        self.capture_name_entry.grid(row=row, column=0, sticky="w", padx=(0, 15))

        self.btn_capture = ctk.CTkButton(
            grid_frame, text="CAPTURE", command=self._toggle_capture,
            fg_color=COLORS['dark_btn'], hover_color=COLORS['dark_btn_hover'],
            width=235, height=38, corner_radius=20
        )
        self.btn_capture.grid(row=row, column=1, columnspan=2, padx=5)
        row += 1

        # Spacer
        ctk.CTkLabel(grid_frame, text="").grid(row=row, column=0, pady=5)
        row += 1

        # ===== DELETE PERSON =====
        ctk.CTkLabel(grid_frame, text="DELETE PERSON", font=("Arial", 12),
                    text_color=COLORS['text_dark']).grid(row=row, column=0, sticky="w", pady=(0, 5))
        row += 1

        self.people_var = ctk.StringVar(value="Select person...")
        self.people_dropdown = ctk.CTkComboBox(
            grid_frame, values=["No people in database"],
            variable=self.people_var, width=150,
            fg_color="white", border_color=COLORS['border'],
            button_color=COLORS['dark_btn'], button_hover_color=COLORS['dark_btn_hover'],
            command=self._on_person_selected
        )
        self.people_dropdown.grid(row=row, column=0, sticky="w", padx=(0, 15))

        ctk.CTkButton(
            grid_frame, text="DELETE", command=self._delete_person,
            fg_color=COLORS['dark_btn'], hover_color=COLORS['dark_btn_hover'],
            width=235, height=38, corner_radius=20
        ).grid(row=row, column=1, columnspan=2, padx=5)
        row += 1

        # Black list filter checkbox pod DELETE PERSON
        self.delete_blacklist_var = ctk.BooleanVar(value=False)
        checkbox_frame2 = ctk.CTkFrame(grid_frame, fg_color="transparent")
        checkbox_frame2.grid(row=row, column=0, columnspan=3, sticky="w", pady=(3, 0))

        ctk.CTkCheckBox(checkbox_frame2, text="Black list filter",
                       variable=self.delete_blacklist_var,
                       command=self._update_delete_blacklist,
                       text_color=COLORS['text_dark']).pack(anchor="w")
        ctk.CTkLabel(checkbox_frame2, text="Description", font=("Arial", 10),
                    text_color=COLORS['text_gray']).pack(anchor="w", padx=(25, 0))

        # Odśwież listę osób
        if AI_AVAILABLE and hasattr(self, 'engine') and self.engine:
            self._refresh_db_list()

        # Sync recording button state
        self._sync_recording_button()

    def create_options_tab(self):
        """Zakładka OPTIONS - ustawienia YOLO"""

        # Tytuł
        ctk.CTkLabel(self.tab_content, text="YOLO", font=("Arial", 36, "bold"),
                    text_color=COLORS['text_dark']).pack(anchor="w", pady=(0, 5))

        ctk.CTkLabel(self.tab_content, text="Real-time object detection", font=("Arial", 16),
                    text_color=COLORS['text_gray']).pack(anchor="w", pady=(0, 30))

        # MODEL section
        ctk.CTkLabel(self.tab_content, text="MODEL", font=("Arial", 12),
                    text_color=COLORS['text_dark']).pack(anchor="w")

        model_frame = ctk.CTkFrame(self.tab_content, fg_color="transparent")
        model_frame.pack(fill="x", pady=(10, 20))

        # Lista modeli
        self.model_var = ctk.StringVar(value="nano")
        model_listbox = ctk.CTkFrame(model_frame, fg_color="white", border_width=1,
                                     border_color=COLORS['border'], corner_radius=5)
        model_listbox.pack(side="left", padx=(0, 30))

        for model in ["nano", "small", "medium", "high", "large"]:
            btn = ctk.CTkButton(
                model_listbox, text=model,
                fg_color="transparent" if self.model_var.get() != model else "#E8E8E8",
                text_color=COLORS['text_dark'],
                hover_color="#F0F0F0",
                anchor="w", width=120, height=30,
                command=lambda m=model: self._select_model(m)
            )
            btn.pack(fill="x", padx=5, pady=2)

        # XLARGE button
        xlarge_frame = ctk.CTkFrame(model_frame, fg_color="transparent")
        xlarge_frame.pack(side="left")

        ctk.CTkLabel(xlarge_frame, text="or just go extreme", font=("Arial", 11),
                    text_color=COLORS['text_gray']).pack(anchor="w")

        ctk.CTkLabel(xlarge_frame, text="XLARGE", font=("Arial", 20, "bold"),
                    text_color=COLORS['text_dark']).pack(anchor="w", pady=(5, 10))

        xlarge_row = ctk.CTkFrame(xlarge_frame, fg_color="transparent")
        xlarge_row.pack(anchor="w")

        ctk.CTkLabel(xlarge_row, text="XLARGE", font=("Arial", 12),
                    text_color=COLORS['text_dark']).pack(side="left", padx=(0, 15))

        ctk.CTkButton(
            xlarge_row, text="CLICK", command=lambda: self._select_model("xlarge"),
            fg_color=COLORS['dark_btn'], hover_color=COLORS['dark_btn_hover'],
            width=80, height=32, corner_radius=15
        ).pack(side="left")

        # Checkboxes
        checks_frame = ctk.CTkFrame(self.tab_content, fg_color="transparent")
        checks_frame.pack(fill="x", pady=(30, 0))

        self.var_warnings = ctk.BooleanVar(value=False)
        self.var_labels = ctk.BooleanVar(value=True)
        self.var_conf = ctk.BooleanVar(value=True)

        for var, text in [(self.var_warnings, "Warnings"),
                         (self.var_labels, "Labels"),
                         (self.var_conf, "Show Conf")]:
            check_col = ctk.CTkFrame(checks_frame, fg_color="transparent")
            check_col.pack(side="left", padx=(0, 40))

            ctk.CTkCheckBox(check_col, text=text, variable=var,
                           command=self._update_yolo_opts,
                           text_color=COLORS['text_dark']).pack(anchor="w")
            ctk.CTkLabel(check_col, text="Description", font=("Arial", 10),
                        text_color=COLORS['text_gray']).pack(anchor="w", padx=(25, 0))

        # ===== SETTINGS (merged) =====
        ctk.CTkLabel(self.tab_content, text="SETTINGS", font=("Arial", 16, "bold"),
                    text_color=COLORS['text_dark']).pack(anchor="w", pady=(30, 10))

        # GPU settings
        ctk.CTkLabel(self.tab_content, text="GPU", font=("Arial", 12, "bold"),
                    text_color=COLORS['text_dark']).pack(anchor="w")

        self.quantization_var = ctk.StringVar(value=config.quantization_mode)
        for text, value in [("FP32 (Full Precision)", "fp32"),
                           ("FP16 (Half Precision) - Recommended", "fp16"),
                           ("INT8 (8-bit Quantization)", "int8")]:
            ctk.CTkRadioButton(self.tab_content, text=text, variable=self.quantization_var,
                               value=value).pack(anchor="w", pady=2)

        self.tensor_cores_var = ctk.BooleanVar(value=config.use_tensor_cores)
        self.optimize_mem_var = ctk.BooleanVar(value=config.optimize_memory)
        ctk.CTkCheckBox(self.tab_content, text="Enable Tensor Cores",
                        variable=self.tensor_cores_var).pack(anchor="w", pady=2)
        ctk.CTkCheckBox(self.tab_content, text="Optimize Memory Layout",
                        variable=self.optimize_mem_var).pack(anchor="w", pady=2)

        # Performance settings
        ctk.CTkLabel(self.tab_content, text="Performance", font=("Arial", 12, "bold"),
                    text_color=COLORS['text_dark']).pack(anchor="w", pady=(20, 0))

        self.show_metrics_var = ctk.BooleanVar(value=config.show_performance_metrics)
        self.graph_mode_var = ctk.StringVar(value=config.graph_mode)
        self.compact_metric_var = ctk.StringVar(value=getattr(config, 'compact_metric', 'fps'))
        self.enable_csv_logging_var = ctk.BooleanVar(value=config.enable_csv_logging)

        ctk.CTkCheckBox(self.tab_content, text="Show Performance Metrics",
                        variable=self.show_metrics_var).pack(anchor="w", pady=2)

        for text, value in [("Off", "off"), ("Compact", "compact"), ("Full", "full")]:
            ctk.CTkRadioButton(self.tab_content, text=text, variable=self.graph_mode_var,
                               value=value).pack(anchor="w", pady=2)

        ctk.CTkLabel(self.tab_content, text="Compact metric:",
                    text_color=COLORS['text_gray']).pack(anchor="w", pady=(5, 0))
        ctk.CTkComboBox(self.tab_content, values=["fps", "inference"],
                        variable=self.compact_metric_var, width=120).pack(anchor="w", pady=2)

        ctk.CTkCheckBox(self.tab_content, text="Enable CSV Performance Log",
                        variable=self.enable_csv_logging_var).pack(anchor="w", pady=2)

        # Detection thresholds
        ctk.CTkLabel(self.tab_content, text="Detection", font=("Arial", 12, "bold"),
                    text_color=COLORS['text_dark']).pack(anchor="w", pady=(20, 0))

        self.yolo_conf_var = ctk.DoubleVar(value=config.yolo_confidence_threshold)
        self.face_det_var = ctk.DoubleVar(value=config.detection_threshold)
        self.face_conf_var = ctk.DoubleVar(value=config.confidence_threshold)
        self.face_topk_var = ctk.IntVar(value=getattr(config, 'face_top_k', 1))

        det_frame1 = ctk.CTkFrame(self.tab_content, fg_color="transparent")
        det_frame1.pack(fill="x", pady=2)
        ctk.CTkLabel(det_frame1, text="YOLO confidence:").pack(side="left")
        ctk.CTkEntry(det_frame1, textvariable=self.yolo_conf_var, width=80).pack(side="right")

        det_frame2 = ctk.CTkFrame(self.tab_content, fg_color="transparent")
        det_frame2.pack(fill="x", pady=2)
        ctk.CTkLabel(det_frame2, text="Face detection:").pack(side="left")
        ctk.CTkEntry(det_frame2, textvariable=self.face_det_var, width=80).pack(side="right")

        det_frame3 = ctk.CTkFrame(self.tab_content, fg_color="transparent")
        det_frame3.pack(fill="x", pady=2)
        ctk.CTkLabel(det_frame3, text="Face recognition:").pack(side="left")
        ctk.CTkEntry(det_frame3, textvariable=self.face_conf_var, width=80).pack(side="right")

        det_frame4 = ctk.CTkFrame(self.tab_content, fg_color="transparent")
        det_frame4.pack(fill="x", pady=2)
        ctk.CTkLabel(det_frame4, text="Top labels (1-3):").pack(side="left")
        ctk.CTkEntry(det_frame4, textvariable=self.face_topk_var, width=80).pack(side="right")

        # Recording
        ctk.CTkLabel(self.tab_content, text="Recording", font=("Arial", 12, "bold"),
                    text_color=COLORS['text_dark']).pack(anchor="w", pady=(20, 0))

        self.enable_recording_var = ctk.BooleanVar(value=getattr(config, 'enable_recording', False))
        self.record_on_alerts_var = ctk.BooleanVar(value=getattr(config, 'record_on_alerts', True))

        ctk.CTkCheckBox(self.tab_content, text="Enable MP4 Recording",
                        variable=self.enable_recording_var).pack(anchor="w", pady=2)
        ctk.CTkCheckBox(self.tab_content, text="Record only on alerts",
                        variable=self.record_on_alerts_var).pack(anchor="w", pady=2)

        # Apply button
        btn_frame = ctk.CTkFrame(self.tab_content, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(20, 0))
        ctk.CTkButton(btn_frame, text="APPLY SETTINGS",
                      command=self._apply_settings_from_options,
                      fg_color="#27ae60", hover_color="#219a52",
                      width=160, height=36).pack(anchor="w")

    def create_help_tab(self):
        """Zakładka HELP"""

        ctk.CTkLabel(self.tab_content, text="HELP", font=("Arial", 36, "bold"),
                    text_color=COLORS['text_dark']).pack(anchor="w", pady=(0, 5))

        ctk.CTkLabel(self.tab_content, text="call our team leader", font=("Arial", 16),
                    text_color=COLORS['text_gray']).pack(anchor="w", pady=(0, 50))

        ctk.CTkLabel(self.tab_content,
                    text="ASK EVERYTHING OUR BOSS HE KNEW EVERY ANWSER",
                    font=("Arial", 12, "italic"),
                    text_color=COLORS['text_dark']).pack(anchor="w", pady=(0, 20))

        ctk.CTkLabel(self.tab_content,
                    text="PS. THERE IS NO POLNISH NEVER EVER",
                    font=("Arial", 12, "italic"),
                    text_color=COLORS['text_dark']).pack(anchor="w")

    def create_website_tab(self):
        """Zakładka WEBSITE"""

        ctk.CTkLabel(self.tab_content, text="WEBSITE", font=("Arial", 36, "bold"),
                    text_color=COLORS['text_dark']).pack(anchor="w", pady=(0, 5))

        ctk.CTkLabel(self.tab_content, text="in making", font=("Arial", 16),
                    text_color=COLORS['text_gray']).pack(anchor="w", pady=(0, 80))

        ctk.CTkLabel(self.tab_content,
                    text="CURRENTLY IN DEVELOP",
                    font=("Arial", 12, "italic"),
                    text_color=COLORS['text_dark']).pack(anchor="w")

    # ============ LOGIKA APLIKACJI ============

    def _log(self, msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}")

    def _connect_camera(self):
        try:
            if self._camera_connecting:
                self._log("Camera connection already in progress")
                return
            cam_id = int(self.camera_var.get())
            self._ensure_preview_window()
            threading.Thread(target=self._start_camera_stream, args=(cam_id,), daemon=True).start()
        except ValueError:
            self._log("Invalid camera ID")

    def _start_camera_stream(self, idx):
        self._camera_connecting = True
        if self.cap:
            self.cap.release()
            self.cap = None

        self._log(f"Connecting to Camera {idx}...")

        try:
            self.cap = cv2.VideoCapture(idx)

            if not self.cap.isOpened():
                self._log(f"Camera {idx} not available")
                self.cap = None
                return

            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ret, test_frame = self.cap.read()
            if not ret:
                self._log(f"Camera {idx} cannot read frames")
                self.cap.release()
                self.cap = None
                return

            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)

            self._log(f"Camera {idx} ready: {w}x{h} @ {fps:.0f}fps")

        except Exception as e:
            self._log(f"Camera error: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
        finally:
            self._camera_connecting = False

    def _open_settings(self):
        self.show_tab("OPTIONS")

    def _start_yolo(self):
        if self.engine:
            self.engine.enable_yolo = True
        self.btn_yolo_run.configure(state="disabled")
        self.btn_yolo_stop.configure(state="normal")
        self._log("YOLO STARTED")

    def _stop_yolo(self):
        if self.engine:
            self.engine.enable_yolo = False
        self.btn_yolo_run.configure(state="normal")
        self.btn_yolo_stop.configure(state="disabled")
        self._log("YOLO STOPPED")

    def _start_face(self):
        if self.engine:
            self.engine.enable_faces = True
        self.btn_face_run.configure(state="disabled")
        self.btn_face_stop.configure(state="normal")
        self._log("Face Recognition STARTED")

    def _stop_face(self):
        if self.engine:
            self.engine.enable_faces = False
        self.btn_face_run.configure(state="normal")
        self.btn_face_stop.configure(state="disabled")
        self._log("Face Recognition STOPPED")

    def _update_face_opts(self):
        if AI_AVAILABLE:
            config.show_blacklist_alerts = self.blacklist_filter_var.get()

    def _toggle_recording(self):
        """Toggle MP4 evidence recording (alerts only)"""
        config.enable_recording = not getattr(config, "enable_recording", False)
        if config.enable_recording:
            config.record_on_alerts = True
        self._sync_recording_button()
        state = "ON" if config.enable_recording else "OFF"
        self._log(f"Recording {state} (alerts only)")

    def _sync_recording_button(self):
        if not hasattr(self, "btn_recording"):
            return
        if getattr(config, "enable_recording", False):
            self.btn_recording.configure(text="RECORDING ON", fg_color="#27ae60", hover_color="#219a52")
        else:
            self.btn_recording.configure(text="RECORDING OFF", fg_color="#8B2332", hover_color="#6f1c28")

    def _update_delete_blacklist(self):
        """Aktualizuje status blacklist dla wybranej osoby"""
        if self.engine and hasattr(self, 'delete_blacklist_var'):
            name = self.people_var.get()
            if name and name != "Select person..." and name != "No people in database":
                is_blacklisted = self.delete_blacklist_var.get()
                self.engine.manager.set_blacklist_status(name, is_blacklisted)
                self._log(f"Blacklist for {name}: {is_blacklisted}")

    def _update_yolo_opts(self):
        if self.engine and hasattr(self.engine, '_object_system') and self.engine._object_system:
            self.engine._object_system.show_conf = self.var_conf.get()
            self.engine._object_system.show_labels = self.var_labels.get()
        if AI_AVAILABLE:
            config.show_yolo_warnings = self.var_warnings.get()

    def _apply_settings_from_options(self):
        if not AI_AVAILABLE:
            messagebox.showinfo("Info", "Settings require ai_core module")
            return

        new_quantization = self.quantization_var.get()
        config.show_performance_metrics = self.show_metrics_var.get()
        config.use_tensor_cores = self.tensor_cores_var.get()
        config.optimize_memory = self.optimize_mem_var.get()
        config.graph_mode = self.graph_mode_var.get()
        config.compact_metric = self.compact_metric_var.get()
        config.enable_csv_logging = self.enable_csv_logging_var.get()

        config.yolo_confidence_threshold = float(self.yolo_conf_var.get())
        config.detection_threshold = float(self.face_det_var.get())
        config.confidence_threshold = float(self.face_conf_var.get())
        config.face_top_k = int(self.face_topk_var.get())

        config.enable_recording = self.enable_recording_var.get()
        config.record_on_alerts = self.record_on_alerts_var.get()

        # Sync recording button if present
        self._sync_recording_button()

        # Apply YOLO threshold immediately if already loaded
        if self.engine and getattr(self.engine, "_object_system", None) is not None:
            self.engine._object_system.conf = config.yolo_confidence_threshold

        # Start/stop CSV logging
        if config.enable_csv_logging and not hasattr(self, "csv_logger"):
            from ai_core.csv_logger import PerformanceCSVLogger
            self.csv_logger = PerformanceCSVLogger(config.csv_log_path)
            self.csv_logger.start()
            self._log(f"CSV logging started: {config.csv_log_path}")
        elif not config.enable_csv_logging and hasattr(self, "csv_logger"):
            self.csv_logger.stop()
            del self.csv_logger
            self._log("CSV logging stopped")

        if new_quantization != config.quantization_mode:
            config.quantization_mode = new_quantization
            self._log(f"Switching to {new_quantization}...")

            def reload_faces():
                try:
                    self.engine._face_system = None
                    _ = self.engine.face_system
                    self._log("Face system reloaded")
                except Exception as e:
                    self._log(f"Error: {e}")

            threading.Thread(target=reload_faces, daemon=True).start()

    def _select_model(self, model_name):
        self._log(f"Switching to {model_name}...")

        # Mapowanie nazw
        model_map = {
            "nano": "yolo11n",
            "small": "yolo11s",
            "medium": "yolo11m",
            "high": "yolo11l",
            "large": "yolo11l",
            "xlarge": "yolo11x"
        }

        if self.engine:
            def change():
                success = self.engine.change_yolo_model(model_map.get(model_name, "yolo11n"))
                if success:
                    self._log(f"Model changed to {model_name}")
            threading.Thread(target=change, daemon=True).start()

    def _toggle_capture(self):
        if not self.recording_mode:
            name = self.capture_name_entry.get().strip()
            if not name:
                messagebox.showwarning("Error", "Please enter a name first!")
                return

            self.target_person_name = name
            self.recording_mode = True
            self.btn_capture.configure(text="STOP CAPTURE", fg_color="#c0392b")
            self.capture_name_entry.configure(state="disabled")
            self._log(f"Capturing faces for: {name}")
        else:
            self.recording_mode = False
            self.btn_capture.configure(text="CAPTURE", fg_color=COLORS['dark_btn'])
            self.capture_name_entry.configure(state="normal")
            self.target_person_name = ""
            self._log("Capture finished")
            self._refresh_db_list()
            if self.engine:
                self.engine.reload_faces()

    def _delete_person(self):
        name = self.people_var.get()
        if not name or name == "Select person..." or name == "No people in database":
            return

        if messagebox.askyesno("Confirm", f"Delete {name}?"):
            if self.engine:
                success = self.engine.manager.delete_person(name)
                if success:
                    self._log(f"Deleted: {name}")
                    self._refresh_db_list()
                    self.engine.reload_faces()

    def _refresh_db_list(self):
        if self.engine:
            people = self.engine.manager.get_people_list()
            if people:
                self.people_dropdown.configure(values=people)
                self.people_var.set(people[0])
            else:
                self.people_dropdown.configure(values=["No people in database"])
                self.people_var.set("No people in database")

    def _on_person_selected(self, choice):
        """Aktualizuje checkbox blacklist gdy wybrano osobę"""
        if self.engine and hasattr(self, 'delete_blacklist_var'):
            if choice and choice != "Select person..." and choice != "No people in database":
                is_blacklisted = self.engine.manager.is_blacklisted(choice)
                self.delete_blacklist_var.set(is_blacklisted)

    def _update_frame(self):
        """Główna pętla aktualizacji"""
        if not self.is_running:
            return

        if self.perf_monitor:
            self.perf_monitor.frame_start()
            self._metrics_counter += 1
            if self._metrics_counter >= 30:
                self._metrics_counter = 0
                self.perf_monitor.update_system_metrics()

        if self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    frame = np.asarray(frame)

                    # Przetwarzanie AI
                    if self.engine and (self.engine.enable_yolo or self.engine.enable_faces):
                        h, w = frame.shape[:2]
                        target_w = 480
                        if w > target_w:
                            scale = target_w / w
                            process_frame = cv2.resize(frame, (target_w, int(h * scale)))
                        else:
                            process_frame = frame

                        processed = self.engine.process_frame(process_frame)
                    else:
                        processed = frame

                    # Nagrywanie twarzy
                    if self.recording_mode and self.target_person_name and self.engine:
                        if int(time.time() * 10) % 10 == 0:
                            self.engine.manager.save_training_photo(self.target_person_name, frame)

                    # Nagrywanie MP4 (alerty lub ciągłe)
                    self._handle_recording(processed)

                    # Podgląd w GUI
                    self._update_preview(processed)

            except Exception as e:
                self._log(f"Frame error: {e}")
        else:
            self._update_preview(None)

        if self.perf_monitor:
            self.perf_monitor.frame_end()

        if config.enable_csv_logging and hasattr(self, "csv_logger") and self.perf_monitor:
            try:
                metrics_dict = self.perf_monitor.get_metrics()
                state = {
                    'ai_running': int(self.engine.enable_yolo or self.engine.enable_faces) if self.engine else 0,
                    'face_loading': int(getattr(self.engine._face_system, '_loading', False) if self.engine and self.engine._face_system else False),
                    'objects_loading': int(getattr(self.engine._object_system, '_loading', False) if self.engine and self.engine._object_system else False),
                    'drawing_metrics': int(config.show_performance_metrics),
                    'drawing_graph': int(config.show_performance_metrics and config.graph_mode in ['compact', 'full'])
                }
                self.csv_logger.log_frame(config, state, metrics_dict)
            except Exception as e:
                self._log(f"CSV logging error: {e}")

        self.after(10, self._update_frame)

    def _update_preview(self, frame):
        """Aktualizuje podgląd kamery w GUI"""
        if not self.preview_window or not self.preview_label:
            return

        if not self.preview_window.winfo_exists():
            self.preview_window = None
            self.preview_label = None
            return

        if frame is None:
            self.preview_label.configure(image=None, text="No camera")
            return

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = img.resize((self.preview_width, self.preview_height), Image.LANCZOS)
            self._preview_image = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=self._preview_image, text="")
        except Exception as e:
            self._log(f"Preview error: {e}")

    def _ensure_detection_log(self):
        try:
            config.detections_dir.mkdir(parents=True, exist_ok=True)
            if not config.detections_csv_path.exists():
                with open(config.detections_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "alerts", "evidence_file"])
        except Exception as e:
            self._log(f"Detection log error: {e}")

    def _start_recording(self, frame, alerts_text):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        evidence_name = f"evidence_{timestamp}.mp4"
        evidence_path = config.detections_dir / evidence_name

        h, w = frame.shape[:2]
        fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 30
        if not fps or fps <= 0:
            fps = 30

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(str(evidence_path), fourcc, fps, (w, h))
        self.recording_active = True
        self.current_evidence_file = evidence_name

        try:
            with open(config.detections_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.datetime.now().isoformat(), alerts_text, evidence_name])
        except Exception as e:
            self._log(f"Detections log write error: {e}")

    def _stop_recording(self):
        if self.video_writer is not None:
            try:
                self.video_writer.release()
            except Exception:
                pass
        self.video_writer = None
        self.recording_active = False
        self.current_evidence_file = None

    def _handle_recording(self, frame):
        if not getattr(config, "enable_recording", False):
            if self.recording_active:
                self._stop_recording()
            return

        record_on_alerts = getattr(config, "record_on_alerts", True)
        alert_active = bool(getattr(self.engine, "alert_active", False)) if self.engine else False
        alerts_text = "|".join(getattr(self.engine, "current_alerts", [])) if self.engine else ""

        should_record = alert_active if record_on_alerts else True

        if should_record and not self.recording_active:
            self._start_recording(frame, alerts_text)
        elif not should_record and self.recording_active:
            self._stop_recording()

        if self.recording_active and self.video_writer is not None:
            try:
                self.video_writer.write(frame)
            except Exception:
                pass

    def _ensure_preview_window(self):
        if self.preview_window and self.preview_window.winfo_exists():
            self.preview_window.lift()
            return

        self.preview_window = ctk.CTkToplevel(self)
        self.preview_window.title("Camera Preview")
        self.preview_window.geometry(f"{self.preview_width}x{self.preview_height}")
        self.preview_window.resizable(False, False)
        self.preview_window.transient(self)
        self.preview_window.wm_attributes("-toolwindow", True)

        self.preview_label = tk.Label(
            self.preview_window,
            text="Connecting...",
            fg=COLORS['text_gray'],
            bg=COLORS['bg']
        )
        self.preview_label.pack(fill="both", expand=True)

        def on_close():
            self.preview_window.destroy()
            self.preview_window = None
            self.preview_label = None

        self.preview_window.protocol("WM_DELETE_WINDOW", on_close)

    def on_close(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        if self.recording_active:
            self._stop_recording()
        if hasattr(self, "csv_logger"):
            self.csv_logger.stop()
        self.destroy()


if __name__ == "__main__":
    app = FaceIDApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()