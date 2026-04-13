import numpy as np

# --- Segmentation ---

def load_audio_simple(duration=1.0, sr=8000):
    """
    Tạo một audio giả để test (sin wave ghép nhiều đoạn).
    Trong project thật sẽ dùng librosa.load() để đọc file .wav
    """
    t = np.linspace(0, duration, int(sr * duration))
    # Ghép 3 đoạn tín hiệu khác nhau để mô phỏng audio không đồng nhất
    third = len(t) // 3
    signal = np.concatenate([
        0.8 * np.sin(2 * np.pi * 440 * t[:third]),       # tiếng La 440Hz
        0.2 * np.random.randn(third),                      # tiếng ồn nhỏ
        0.9 * np.sin(2 * np.pi * 880 * t[third*2:]),      # tiếng La cao 880Hz
    ])
    return signal, sr


def segment_fixed_window(signal, sr, window_ms=50):
    """
    Solution 1: Chia signal thành các window có kích thước cố định.
    
    Args:
        signal: mảng numpy chứa dữ liệu audio
        sr: sample rate (số mẫu/giây)
        window_ms: kích thước window tính bằng milliseconds
    
    Returns:
        list các window (mỗi window là một mảng numpy)
    """
    window_size = int(sr * window_ms / 1000)  # đổi ms → số samples
    windows = []
    
    for start in range(0, len(signal), window_size):
        end = start + window_size
        window = signal[start:end]
        if len(window) == window_size:  # bỏ window cuối nếu không đủ
            windows.append(window)
    
    return windows


# --- Chạy thử ---
signal, sr = load_audio_simple(duration=1.0, sr=8000)
windows = segment_fixed_window(signal, sr, window_ms=50)

print(f"Độ dài signal: {len(signal)} samples ({len(signal)/sr:.1f} giây)")
print(f"Sample rate: {sr} Hz")
print(f"Kích thước mỗi window: {int(sr * 50/1000)} samples (50ms)")
print(f"Số windows tạo ra: {len(windows)}")
print(f"\nMỗi window là 1 bản ghi có dạng:")
print(f"  (audio_source, window_index, duration, [K features...])")
