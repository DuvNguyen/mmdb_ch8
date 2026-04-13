import numpy as np

# --- BƯỚC 3: Feature Extraction ---

MU = 1.2      # mật độ không khí (kg/m³)
V  = 343.0    # vận tốc âm (m/s)
L0 = 1e-12    # ngưỡng nghe thấp nhất của tai người

def extract_intensity(window):
    """
    I = 2π² × f² × μ × a² × v
    f ước lượng bằng zero-crossing rate, a là RMS amplitude.
    """
    a = np.sqrt(np.mean(window ** 2))          # RMS ≈ biên độ hiệu dụng
    # Đếm số lần signal đổi dấu → ước lượng tần số
    zero_crossings = np.sum(np.diff(np.sign(window)) != 0)
    f = zero_crossings / (2 * len(window))     # tần số ước tính (Hz)
    f = max(f, 1e-6)                           # tránh chia cho 0
    I = 2 * (np.pi**2) * (f**2) * MU * (a**2) * V
    return float(I)

def extract_loudness(window):
    """
    L = 10 × log(I / L0)
    """
    I = extract_intensity(window)
    I = max(I, L0)                             # tránh log(0)
    L = 10 * np.log10(I / L0)
    return float(L)

def extract_pitch(window):
    """
    Pitch ≈ tần số cơ bản (fundamental frequency).
    Dùng autocorrelation để tìm chu kỳ lặp lại của sóng.
    """
    corr = np.correlate(window, window, mode='full')
    corr = corr[len(corr)//2:]                 # chỉ lấy nửa dương
    # Tìm đỉnh đầu tiên sau lag=1 (bỏ qua lag=0)
    d = np.diff(corr)
    peaks = np.where((d[:-1] > 0) & (d[1:] < 0))[0] + 1
    if len(peaks) == 0:
        return 0.0
    pitch = 1.0 / peaks[0] if peaks[0] > 0 else 0.0
    return float(pitch)

def extract_brightness(window):
    """
    Brightness = spectral centroid: trọng tâm của phổ tần số.
    Âm thanh "trong" (breaking glass) có centroid cao,
    âm thanh "đục" (muffled) có centroid thấp.
    """
    spectrum = np.abs(np.fft.rfft(window))     # biến đổi Fourier
    freqs    = np.fft.rfftfreq(len(window))
    if spectrum.sum() == 0:
        return 0.0
    centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
    return float(centroid)

def extract_features(window):
    """
    Gộp cả 4 features thành 1 vector [I, L, pitch, brightness].
    Đây chính là K=4 features trong công thức (K+3)-tuple của sách.
    """
    return np.array([
        extract_intensity(window),
        extract_loudness(window),
        extract_pitch(window),
        extract_brightness(window),
    ])

# --- Chạy thử trên các windows từ Bước 2 ---
def load_audio_simple(duration=1.0, sr=8000):
    t = np.linspace(0, duration, int(sr * duration))
    third = len(t) // 3
    return np.concatenate([
        0.8 * np.sin(2 * np.pi * 440 * t[:third]),
        0.2 * np.random.randn(third),
        0.9 * np.sin(2 * np.pi * 880 * t[third*2:]),
    ]), sr

def segment_fixed_window(signal, sr, window_ms=50):
    window_size = int(sr * window_ms / 1000)
    return [signal[i:i+window_size]
            for i in range(0, len(signal)-window_size+1, window_size)]

def segment_by_silence(signal, sr, threshold=0.01, min_silence_ms=100):
    """
    SOLUTION 2: Phân đoạn dựa trên khoảng lặng (Silence detection).
    Chia tín hiệu tại các vị trí có năng lượng thấp hơn threshold
    với khoảng thời gian ít nhất là min_silence_ms.
    """
    window_size = int(sr * 0.01) # cửa sổ 10ms để quét năng lượng
    energies = []
    for i in range(0, len(signal), window_size):
        w = signal[i:i+window_size]
        energies.append(np.sqrt(np.mean(w**2)))
    
    energies = np.array(energies)
    is_silent = energies < threshold
    
    # Gom cụm các đoạn silent
    segments = []
    current_start = 0
    min_silence_frames = max(1, int(min_silence_ms / 10))
    
    silent_count = 0
    for i, silent in enumerate(is_silent):
        if silent:
            silent_count += 1
        else:
            if silent_count >= min_silence_frames:
                # Cắt đoạn từ current_start đến point này
                end_idx = i * window_size
                segments.append(signal[current_start:end_idx])
                current_start = end_idx
            silent_count = 0
            
    # Thêm đoạn cuối cùng
    if current_start < len(signal):
        segments.append(signal[current_start:])
        
    return segments

if __name__ == "__main__":
    # Tạo audio có khoảng lặng ở giữa
    sr = 8000
    t = np.linspace(0, 0.5, int(sr * 0.5))
    s1 = 0.5 * np.sin(2 * np.pi * 440 * t)     # 0.5s tone
    silence = np.zeros(int(sr * 0.2))          # 0.2s silence
    s2 = 0.5 * np.sin(2 * np.pi * 880 * t)     # 0.5s tone
    signal = np.concatenate([s1, silence, s2])

    print("--- PHƯƠNG PHÁP 1: FIXED WINDOW ---")
    windows = segment_fixed_window(signal, sr, window_ms=200)
    print(f"Số lượng windows cố định (200ms): {len(windows)}")

    print("\n--- PHƯƠNG PHÁP 2: SILENCE-BASED ---")
    segments = segment_by_silence(signal, sr, threshold=0.01, min_silence_ms=100)
    print(f"Số lượng segments theo khoảng lặng: {len(segments)}")
    for i, seg in enumerate(segments):
        print(f"Segment {i+1}: Độ dài {len(seg)/sr:.2f}s")
