import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kdtree import KDTree
from extract import extract_features, segment_fixed_window, segment_by_silence, load_audio_simple

class AudioDBApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Database System - Chapter 8")
        self.geometry("1100x800")
        self.configure(bg="#1e1e2e")

        self.signal = None
        self.sr = 8000
        self.windows = []
        self.features = []
        self.segment_times = [] # [(start, end), ...]
        self.tree = None
        self.seg_mode = tk.StringVar(value="fixed")

        self._build_ui()

    def _build_ui(self):
        # Apply styles
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview", background="#313244", foreground="#cdd6f4", fieldbackground="#313244", borderwidth=0)
        
        # Toolbar
        toolbar = tk.Frame(self, bg="#313244", pady=10)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(toolbar, text="🎵 Load Synthetic Audio", bg="#89b4fa", command=self._load_synthetic).pack(side=tk.LEFT, padx=10)
        tk.Button(toolbar, text="📂 Load .wav File", bg="#b4befe", command=self._load_wav).pack(side=tk.LEFT, padx=10)
        
        # New: Mode Selection
        mode_frame = tk.Frame(toolbar, bg="#313244")
        mode_frame.pack(side=tk.LEFT, padx=20)
        tk.Label(mode_frame, text="Mode:", bg="#313244", fg="#cdd6f4").pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Fixed (100ms)", variable=self.seg_mode, value="fixed", 
                     bg="#313244", fg="#cdd6f4", selectcolor="#1e1e2e", activebackground="#313244").pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Silence-based", variable=self.seg_mode, value="silence", 
                     bg="#313244", fg="#cdd6f4", selectcolor="#1e1e2e", activebackground="#313244").pack(side=tk.LEFT)

        tk.Button(toolbar, text="🚀 Build Index (KD-Tree)", bg="#a6e3a1", command=self._build_index).pack(side=tk.LEFT, padx=10)

        # Main Content
        main_frame = tk.Frame(self, bg="#1e1e2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left: Features and Visualization
        left_frame = tk.Frame(main_frame, bg="#1e1e2e")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        header_frame = tk.Frame(left_frame, bg="#1e1e2e")
        header_frame.pack(fill=tk.X)

        tk.Label(header_frame, text="📈 Audio Features Visualization", bg="#1e1e2e", fg="#cdd6f4", font=("Helvetica", 12, "bold")).pack(side=tk.LEFT, pady=5)
        self.seg_count_label = tk.Label(header_frame, text="Total Segments: 0", bg="#1e1e2e", fg="#a6adc8", font=("Helvetica", 10))
        self.seg_count_label.pack(side=tk.RIGHT, pady=5, padx=10)
        
        self.viz_canvas = tk.Canvas(left_frame, bg="#181825", highlightthickness=0, height=300)
        self.viz_canvas.pack(fill=tk.X, expand=True, padx=5, pady=5)

        # Feature List
        self.feat_tree = ttk.Treeview(left_frame, columns=("ID", "Time", "I", "L", "P", "B"), show="headings", height=10)
        self.feat_tree.heading("ID", text="ID")
        self.feat_tree.heading("Time", text="Interval")
        self.feat_tree.heading("I", text="Intensity")
        self.feat_tree.heading("L", text="Loudness")
        self.feat_tree.heading("P", text="Pitch")
        self.feat_tree.heading("B", text="Brightness")
        self.feat_tree.column("ID", width=40)
        self.feat_tree.column("Time", width=100)
        for col in ("I", "L", "P", "B"): self.feat_tree.column(col, width=80)
        self.feat_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.feat_tree.bind("<Double-1>", self._on_tree_double_click)

        # Right: Search and Results
        right_frame = tk.LabelFrame(main_frame, text=" Similarity Search", bg="#1e1e2e", fg="#f9e2af", padx=10, pady=10, width=350)
        right_frame.pack_propagate(False)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)

        tk.Label(right_frame, text="Query Segment ID:", bg="#1e1e2e", fg="#cdd6f4").pack(anchor=tk.W)
        self.query_id_entry = tk.Entry(right_frame, bg="#313244", fg="#cdd6f4")
        self.query_id_entry.insert(0, "0")
        self.query_id_entry.pack(fill=tk.X, pady=5)
        tk.Label(right_frame, text="(Range: 0 to Total-1)", bg="#1e1e2e", fg="#a6adc8", font=("Helvetica", 8, "italic")).pack(anchor=tk.W)

        tk.Label(right_frame, text="K Neighbors (Results):", bg="#1e1e2e", fg="#cdd6f4").pack(anchor=tk.W, pady=(10, 0))
        self.k_entry = tk.Entry(right_frame, bg="#313244", fg="#cdd6f4")
        self.k_entry.insert(0, "3")
        self.k_entry.pack(fill=tk.X, pady=5)

        tk.Button(right_frame, text="Find Similar Sounds", bg="#f9e2af", command=self._search).pack(pady=10)
        tk.Button(right_frame, text="▶ Play Selected Segment", bg="#f5e0dc", command=self._play_query_segment).pack(fill=tk.X, pady=5)

        self.results_tree = ttk.Treeview(right_frame, columns=("ID", "Time", "Dist"), show="headings")
        self.results_tree.heading("ID", text="ID")
        self.results_tree.heading("Time", text="Time Range")
        self.results_tree.heading("Dist", text="Distance")
        self.results_tree.column("ID", width=60)
        self.results_tree.column("Time", width=120)
        self.results_tree.pack(fill=tk.BOTH, expand=True)

        # Status Bar
        self.status_var = tk.StringVar(value="Vui lòng nạp audio để bắt đầu.")
        tk.Label(self, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#313244", fg="#a6adc8").pack(side=tk.BOTTOM, fill=tk.X)

    def _load_synthetic(self):
        self.signal, self.sr = load_audio_simple(duration=2.0)
        self.status_var.set("✅ Đã tạo audio tổng hợp (2 giây).")
        self._process_audio()

    def _load_wav(self):
        f = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not f: return
        try:
            import wave
            with wave.open(f, 'rb') as w:
                self.sr = w.getframerate()
                n_frames = w.getnframes()
                frames = w.readframes(n_frames)
                self.signal = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            self.status_var.set(f"✅ Đã nạp: {os.path.basename(f)}")
            self._process_audio()
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))

    def _process_audio(self):
        if self.signal is None: return
        
        if self.seg_mode.get() == "fixed":
            self.windows = segment_fixed_window(self.signal, self.sr, window_ms=100)
            # Calculate times for fixed windows
            self.segment_times = []
            for i in range(len(self.windows)):
                self.segment_times.append((i * 0.1, (i + 1) * 0.1))
        else:
            self.windows = segment_by_silence(self.signal, self.sr)
            # Rough estimate of times for variable segments (hàm segment_by_silence cần trả về times để chính xác hơn, 
            # nhưng ở đây ta ước lượng dựa trên độ dài)
            self.segment_times = []
            current_time = 0.0
            for w in self.windows:
                duration = len(w) / self.sr
                self.segment_times.append((current_time, current_time + duration))
                current_time += duration

        self.features = [extract_features(w) for w in self.windows]
        
        # Update UI
        self.seg_count_label.config(text=f"Total Segments: {len(self.features)}")
        for item in self.feat_tree.get_children(): self.feat_tree.delete(item)
        for i, f in enumerate(self.features):
            t_start, t_end = self.segment_times[i]
            self.feat_tree.insert("", tk.END, iid=i, values=(i, f"[{t_start:.2f}-{t_end:.2f}s]", f"{f[0]:.4f}", f"{f[1]:.2f}", f"{f[2]:.4f}", f"{f[3]:.4f}"))
        
        self._draw_viz()

    def _draw_viz(self):
        self.viz_canvas.delete("all")
        if not self.features: return
        
        w = self.viz_canvas.winfo_width() or 600
        h = self.viz_canvas.winfo_height() or 300
        
        PAD_L, PAD_R, PAD_T, PAD_B = 50, 20, 40, 40
        graph_w = w - PAD_L - PAD_R
        graph_h = h - PAD_T - PAD_B
        
        # Draw Axes
        self.viz_canvas.create_line(PAD_L, h - PAD_B, w - PAD_R, h - PAD_B, fill="#cdd6f4", width=2) # X
        self.viz_canvas.create_line(PAD_L, PAD_T, PAD_L, h - PAD_B, fill="#cdd6f4", width=2)     # Y
        
        # Y labels (0.0 - 1.0)
        for i in range(5):
            val = i / 4.0
            y = h - PAD_B - val * graph_h
            self.viz_canvas.create_text(PAD_L - 10, y, text=f"{val:.1f}", fill="#a6adc8", anchor=tk.E, font=("Helvetica", 8))
            self.viz_canvas.create_line(PAD_L - 5, y, PAD_L, y, fill="#cdd6f4")

        # X labels (Time)
        n = len(self.features)
        total_time = n * 0.1 # assuming 100ms per window
        dx = graph_w / (n-1) if n > 1 else 0
        
        # Ticks every 0.5s or 1.0s
        step = max(1, int(0.5 / 0.1)) # 5 windows
        for i in range(0, n, step):
            x = PAD_L + i * dx
            time_val = i * 0.1
            self.viz_canvas.create_text(x, h - PAD_B + 15, text=f"{time_val:.1f}s", fill="#a6adc8", font=("Helvetica", 8))
            self.viz_canvas.create_line(x, h - PAD_B, x, h - PAD_B + 5, fill="#cdd6f4")

        # Plot Data
        pitches = [f[2] for f in self.features]
        intensities = [f[0] for f in self.features]
        max_p = max(pitches) if max(pitches) > 0 else 1
        max_i = max(intensities) if max(intensities) > 0 else 1
        
        for i in range(n - 1):
            x1 = PAD_L + i * dx
            x2 = PAD_L + (i+1) * dx
            
            # Pitch (Yellow)
            y1_p = h - PAD_B - (pitches[i] / max_p * graph_h)
            y2_p = h - PAD_B - (pitches[i+1] / max_p * graph_h)
            self.viz_canvas.create_line(x1, y1_p, x2, y2_p, fill="#f9e2af", width=2)
            
            # Intensity (Red Dash)
            y1_i = h - PAD_B - (intensities[i] / max_i * graph_h)
            y2_i = h - PAD_B - (intensities[i+1] / max_i * graph_h)
            self.viz_canvas.create_line(x1, y1_i, x2, y2_i, fill="#f38ba8", width=1, dash=(4,4))

        tk.Label(self.viz_canvas, text="— Pitch (Normalized)", fg="#f9e2af", bg="#181825", font=("Helvetica", 9)).place(x=PAD_L + 10, y=5)
        tk.Label(self.viz_canvas, text="— Intensity (Normalized)", fg="#f38ba8", bg="#181825", font=("Helvetica", 9)).place(x=PAD_L + 150, y=5)
        tk.Label(self.viz_canvas, text="Time (s)", fg="#cdd6f4", bg="#181825", font=("Helvetica", 9)).place(x=w - 60, y=h - PAD_B + 10)

    def _build_index(self):
        if not self.features:
            messagebox.showwarning("Lỗi", "Vui lòng nạp audio trước")
            return
        
        points = [f for f in self.features]
        metadata = list(range(len(self.features))) # Store window IDs as metadata
        self.tree = KDTree(points, metadata)
        self.status_var.set(f"✅ Đã xây dựng KD-Tree với {len(points)} segments.")

    def _search(self):
        if self.tree is None:
            messagebox.showwarning("Lỗi", "Vui lòng xây dựng mục lục trước")
            return
        
        try:
            qid = int(self.query_id_entry.get())
            k = int(self.k_entry.get())
            if qid < 0 or qid >= len(self.features): raise Exception("ID không hợp lệ")
        except Exception as e:
            messagebox.showerror("Lỗi", str(e))
            return

        target = self.features[qid]
        results = self.tree.query(target, k=k)

        for item in self.results_tree.get_children(): self.results_tree.delete(item)
        for dist, window_id in results:
            t_start, t_end = self.segment_times[window_id]
            self.results_tree.insert("", tk.END, iid=f"res_{window_id}", values=(f"Seg {window_id}", f"[{t_start:.2f}-{t_end:.2f}s]", f"{dist:.6f}"))
        
        self.status_var.set(f"🔍 Tìm thấy {len(results)} đoạn tương đồng nhất. Double-click để nghe.")
        self.results_tree.bind("<Double-1>", self._on_results_double_click)

    def _on_tree_double_click(self, event):
        item = self.feat_tree.selection()
        if item:
            sid = int(item[0])
            self._play_segment(sid)

    def _on_results_double_click(self, event):
        item = self.results_tree.selection()
        if item:
            # iid is "res_9" -> sid is 9
            sid = int(item[0].split('_')[1])
            self._play_segment(sid)

    def _play_query_segment(self):
        try:
            sid = int(self.query_id_entry.get())
            self._play_segment(sid)
        except:
            pass

    def _play_segment(self, sid):
        if not self.windows or sid < 0 or sid >= len(self.windows):
            return
        
        import wave
        import subprocess
        import tempfile
        
        # Tạo file tạm
        temp_wav = os.path.join(tempfile.gettempdir(), f"mmdb_segment_{sid}.wav")
        segment_data = self.windows[sid]
        
        # Convert back to 16-bit PCM for simple players
        pcm_data = (segment_data * 32767).astype(np.int16)
        
        with wave.open(temp_wav, 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2) # 16-bit
            w.setframerate(self.sr)
            w.writeframes(pcm_data.tobytes())
        
        # Thử các lệnh play phổ biến trên Linux
        self.status_var.set(f"🔊 Đang phát Segment {sid}...")
        try:
            # Thử ffplay (gọn, không hiện cửa sổ nếu dùng -nodisp)
            subprocess.Popen(["ffplay", "-nodisp", "-autoexit", temp_wav], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            try:
                # Thử aplay (standard ALSA)
                subprocess.Popen(["aplay", temp_wav], 
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                try:
                    # Thử paplay (PulseAudio)
                    subprocess.Popen(["paplay", temp_wav], 
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except FileNotFoundError:
                    messagebox.showerror("Lỗi", "Không tìm thấy chương trình phát âm thanh (ffplay, aplay, paplay).")


def run():
    app = AudioDBApp()
    app.mainloop()

if __name__ == "__main__":
    run()
