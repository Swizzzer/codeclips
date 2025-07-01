import numpy as np
from scipy.io import wavfile
from scipy.ndimage import binary_closing
import argparse

MORSE_CODE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', '-----': '0', '.----': '1', '..---': '2',
    '...--': '3', '....-': '4', '.....': '5', '-....': '6',
    '--...': '7', '---..': '8', '----.': '9', '.-.-.-': '.',
    '--..--': ',', '..--..': '?', '-..-.': '/', '-....-': '-',
    '-.--.': '(', '-.--.-': ')', '.--.-.': '@'
}


def decode_morse_from_wav(filepath, config):
    """
    只适合非常纯净的morse code音频(没有背景噪音的那种)
    """
    try:
        samplerate, data = wavfile.read(filepath)
    except Exception as e:
        return f"Error reading WAV file: {e}"

    if data.ndim > 1:
        data = data.mean(axis=1)

    amplitude = np.abs(data)
    threshold = np.max(amplitude) * config['amplitude_threshold_ratio']
    is_signal = amplitude > threshold
    
    print(f"[*] Max amplitude: {np.max(amplitude)}, Threshold: {threshold:.2f}")
    print(f"[*] Signal ratio before smoothing: {np.sum(is_signal) / len(is_signal):.2%}")

    # 使用二值闭运算来填充信号中的短暂间隙从而避免毛刺
    gap_samples = int(samplerate * (config['gap_ms'] / 1000.0))
    if gap_samples > 0:
        print(f"[*] Smoothing signal by closing gaps up to {config['gap_ms']}ms ({gap_samples} samples)...")
        structure = np.ones(gap_samples)
        is_signal = binary_closing(is_signal, structure=structure)
        print(f"[*] Signal ratio after smoothing: {np.sum(is_signal) / len(is_signal):.2%}")

    events = []
    current_state = is_signal[0]
    count = 0
    for s in is_signal:
        if s == current_state:
            count += 1
        else:
            events.append((current_state, count))
            current_state = s
            count = 1
    events.append((current_state, count))

    signal_durations = [duration for state, duration in events if state]
    if not signal_durations:
        return "No signal detected at all. Try lowering the --amplitude-ratio."

    min_duration_samples = samplerate * (config['min_duration_ms'] / 1000.0)
    meaningful_signal_durations = [d for d in signal_durations if d > min_duration_samples]

    if not meaningful_signal_durations:
        return (f"No meaningful signal durations found.\n"
                f"All detected signals were shorter than {config['min_duration_ms']}ms, even after smoothing.\n"
                f"Try increasing --gap-ms or adjusting --amplitude-ratio.")
    
    dot_length = min(meaningful_signal_durations)

    dash_threshold = dot_length * config['dot_dash_ratio']
    char_space_threshold = dot_length * config['char_space_ratio']
    word_space_threshold = dot_length * config['word_space_ratio']

    morse_sequence = []
    for i, (state, duration) in enumerate(events):
        if state:
            if duration < dash_threshold:
                morse_sequence.append('.')
            else:
                morse_sequence.append('-')
        else:
            if i > 0 and events[i-1][0]:
                if duration > word_space_threshold:
                    morse_sequence.append(' / ')
                elif duration > char_space_threshold:
                    morse_sequence.append(' ')

    morse_string = "".join(morse_sequence).strip()
    
    print(f"[*] Base dot length: {dot_length / samplerate:.3f} seconds")
    print(f"[*] Detected Morse sequence: {morse_string}")

    decoded_text = []
    words = morse_string.split(' / ')
    for word in words:
        chars = word.split(' ')
        decoded_word = ""
        for char_code in chars:
            if char_code in MORSE_CODE_DICT:
                decoded_word += MORSE_CODE_DICT[char_code]
            elif char_code:
                decoded_word += '?'
        decoded_text.append(decoded_word)

    return " ".join(decoded_text)

def main():
    parser = argparse.ArgumentParser(description="Decode Morse code from a WAV file.")
    parser.add_argument("wav_file", help="Path to the WAV file to decode.")
    
    parser.add_argument("--amplitude-ratio", type=float, default=0.1,
                        help="Amplitude threshold ratio to distinguish signal from noise (default: 0.1)")
    # ARGUMENT FOR SMOOTHING
    parser.add_argument("--gap-ms", type=float, default=5.0,
                        help="Connect signal parts separated by a gap of up to this many milliseconds (default: 5.0)")
    parser.add_argument("--min-duration-ms", type=float, default=10.0,
                        help="Minimum duration in milliseconds for a signal to be considered a 'dot' (default: 10.0)")
    parser.add_argument("--dot-dash-ratio", type=float, default=2.0,
                        help="Ratio to distinguish a dot from a dash (default: 2.0)")
    parser.add_argument("--char-space-ratio", type=float, default=2.0,
                        help="Ratio for character space vs intra-character space (default: 2.0)")
    parser.add_argument("--word-space-ratio", type=float, default=5.0,
                        help="Ratio for word space vs character space (default: 5.0)")

    args = parser.parse_args()

    config = {
        'amplitude_threshold_ratio': args.amplitude_ratio,
        'gap_ms': args.gap_ms,
        'min_duration_ms': args.min_duration_ms,
        'dot_dash_ratio': args.dot_dash_ratio,
        'char_space_ratio': args.char_space_ratio,
        'word_space_ratio': args.word_space_ratio,
    }

    print(f"--- Decoding {args.wav_file} ---")
    result = decode_morse_from_wav(args.wav_file, config)
    print("\n--- Result ---")
    print(f"Decoded Text: {result}")
    print("--------------")

if __name__ == "__main__":
    main()
