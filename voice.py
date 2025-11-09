import numpy as np
import librosa
# import aubio  # For more advanced pitch detection, if needed - not used in current pitch detection logic
from scipy.signal import find_peaks
import collections
import pyaudio # For live audio recording
import wave    # For saving dummy audio file if needed

# --- Configuration ---
SAMPLE_RATE = 44100  # Standard audio sample rate
BUFFER_SIZE = 2048   # Size of audio chunks for processing (used by aubio, if active)
HOP_SIZE = 512       # Hop size for pitch detection (how often to analyze)

# Tolerance for pitch matching (in cents or MIDI steps)
PITCH_TOLERANCE_CENTS = 50  # +/- 50 cents (half a semitone) is a common tolerance
PITCH_TOLERANCE_MIDI_STEPS = PITCH_TOLERANCE_CENTS / 100

# --- Noise/Silence Thresholding Configuration ---
# RMS (Root Mean Square) energy threshold to consider a frame as "active" voice
# This is a critical parameter. Adjust based on your microphone and environment.
# Values between 0.005 and 0.02 are common starting points for normalized audio (-1.0 to 1.0).
RMS_THRESHOLD = 0.015
# The minimum number of consecutive "active" frames to consider a segment valid for pitch detection.
# Helps avoid analyzing short bursts of noise.
MIN_ACTIVE_FRAMES = 5

# --- Scale Generation Functions (from previous response) ---

def generate_scale(start_midi_note, interval_pattern):
    """
    Generates a scale (list of MIDI note numbers) given a starting MIDI note
    and an interval pattern (in semitones).
    """
    scale_notes = [start_midi_note]
    current_note = start_midi_note
    for interval in interval_pattern:
        current_note += interval
        scale_notes.append(current_note)
    return scale_notes

def get_common_scales():
    """
    Generates and returns a dictionary of common major and natural minor scales,
    with their notes represented as MIDI numbers.
    Each scale covers one octave.
    """
    # Define the 12 chromatic notes and their enharmonic equivalents
    # Starting MIDI note for C4 is 60. We'll use this as a reference point.
    chromatic_notes_info = {
        0: {"name": "C", "enharmonic": None},
        1: {"name": "C#", "enharmonic": "Db"},
        2: {"name": "D", "enharmonic": None},
        3: {"name": "D#", "enharmonic": "Eb"},
        4: {"name": "E", "enharmonic": None},
        5: {"name": "F", "enharmonic": None},
        6: {"name": "F#", "enharmonic": "Gb"},
        7: {"name": "G", "enharmonic": None},
        8: {"name": "G#", "enharmonic": "Ab"},
        9: {"name": "A", "enharmonic": None},
        10: {"name": "A#", "enharmonic": "Bb"},
        11: {"name": "B", "enharmonic": None},
    }

    # Scale interval patterns in semitones
    MAJOR_INTERVALS = [2, 2, 1, 2, 2, 2, 1]
    NATURAL_MINOR_INTERVALS = [2, 1, 2, 2, 1, 2, 2]

    all_scales = collections.OrderedDict()

    # Generate Major Scales
    for midi_offset in range(12):
        root_info = chromatic_notes_info[midi_offset]
        root_name = root_info["name"]
        start_note_midi = 60 + midi_offset # C4 (MIDI 60) for C Major, C#4 (MIDI 61) for C# Major, etc.

        major_scale_notes = generate_scale(start_note_midi, MAJOR_INTERVALS)
        all_scales[f"{root_name} Major"] = major_scale_notes

        if root_info["enharmonic"]:
            enharmonic_name = root_info["enharmonic"]
            all_scales[f"{enharmonic_name} Major"] = major_scale_notes

    # Generate Natural Minor Scales
    for midi_offset in range(12):
        root_info = chromatic_notes_info[midi_offset]
        root_name = root_info["name"]
        start_note_midi = 60 + midi_offset # Same as for Major scales

        minor_scale_notes = generate_scale(start_note_midi, NATURAL_MINOR_INTERVALS)
        all_scales[f"{root_name} Minor"] = minor_scale_notes

        if root_info["enharmonic"]:
            enharmonic_name = root_info["enharmonic"]
            all_scales[f"{enharmonic_name} Minor"] = minor_scale_notes

    return all_scales

# Load all common scales
SCALES = get_common_scales()

# --- Helper Functions for Musical Conversions ---

def freq_to_midi(freq):
    """Converts frequency (Hz) to MIDI note number."""
    if freq <= 0:
        return None  # Indicate no valid MIDI note
    return 12 * (np.log2(freq / 440)) + 69

def midi_to_freq(midi_note):
    """Converts MIDI note number to frequency (Hz)."""
    return 440 * (2 ** ((midi_note - 69) / 12))

def midi_to_note_name(midi_note_num):
    """Converts a MIDI note number to its common note name (e.g., 60 -> C4)."""
    if midi_note_num is None:
        return "N/A"
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi_note_num // 12) - 1 # MIDI note 0 is C-1, so C4 is octave 4 (MIDI 60)
    note_index = int(round(midi_note_num)) % 12 # Round to nearest integer for note name
    return f"{note_names[note_index]}{octave}"

def is_pitch_in_scale(pitch_midi, scale_midi_notes, tolerance_midi_steps):
    """
    Checks if a given pitch (MIDI) is within a specified scale,
    considering a tolerance.
    """
    for scale_note in scale_midi_notes:
        if abs(pitch_midi - scale_note) <= tolerance_midi_steps:
            return True
    return False

def calculate_scale_fit_score(detected_pitches_midi, scale_midi_notes, tolerance_midi_steps):
    """
    Calculates a fit score for how well detected pitches adhere to a given scale.
    Lower score means better fit (less deviation).
    """
    if not detected_pitches_midi:
        return float('inf') # Return infinity if no pitches detected

    total_deviation = 0
    in_scale_count = 0

    for pitch_midi in detected_pitches_midi:
        min_deviation = float('inf')
        is_in_scale = False
        for scale_note in scale_midi_notes:
            deviation = abs(pitch_midi - scale_note)
            if deviation < min_deviation:
                min_deviation = deviation
            if deviation <= tolerance_midi_steps:
                is_in_scale = True
        
        # If the pitch is within tolerance, its contribution to deviation is capped
        # This prevents extremely off-key notes from skewing the score too much for good scales
        if is_in_scale:
            total_deviation += min_deviation
            in_scale_count += 1
        else:
            # For out-of-scale notes, penalize more significantly
            total_deviation += (min_deviation + tolerance_midi_steps) # Add tolerance as a base penalty

    if in_scale_count == 0 and len(detected_pitches_midi) > 0:
        return float('inf') # No notes in scale, very bad fit

    # Score based on average deviation, favoring scales with more in-tune notes
    # We can refine this score based on desired weighting
    # A simple approach: average deviation of all detected pitches
    return total_deviation / len(detected_pitches_midi)

def find_best_matching_scale(all_detected_pitches):
    """
    Finds the scale that best matches the detected pitches.
    Returns the name of the best matching scale and its score.
    """
    if not all_detected_pitches:
        return "N/A", float('inf')

    # Extract just MIDI pitches for scoring
    detected_pitches_midi = [pitch_midi for hz, pitch_midi, name in all_detected_pitches]

    best_scale_name = None
    best_score = float('inf')

    for scale_name, scale_midi_notes in SCALES.items():
        score = calculate_scale_fit_score(detected_pitches_midi, scale_midi_notes, PITCH_TOLERANCE_MIDI_STEPS)
        if score < best_score:
            best_score = score
            best_scale_name = scale_name
    
    return best_scale_name, best_score


# --- Core Analysis Function ---

def analyze_vocal_segment(audio_segment, sr, target_scale_name=None):
    """
    Analyzes an audio segment for pitch, ignoring periods of silence/low noise.
    If a target_scale_name is provided, it also returns in-scale and out-of-scale notes for that target.
    Returns all detected notes, and optionally in-scale/out-of-scale for target.
    """
    # ADD THIS CHECK (already present, but good to re-emphasize its importance):
    if len(audio_segment) == 0:
        print("Warning: Audio segment is empty. Cannot perform analysis.")
        return [], [], []

    all_detected_notes = [] # List to store (original_hz, midi_pitch, note_name)

    # Run piptrack on the entire audio segment first
    # Add a try-except for piptrack as well, in case audio_segment causes issues there
    try:
        pitches, magnitudes = librosa.core.piptrack(
            y=audio_segment, sr=sr, fmin=75, fmax=600, hop_length=HOP_SIZE
        )
    except Exception as e:
        print(f"Error during piptrack: {e}. Returning empty results.")
        return [], [], []

    # Now, filter the detected pitches based on RMS activity
    for t in range(pitches.shape[1]):
        pitch_hz = pitches[:, t].max() # Get the dominant pitch at this time frame
        magnitude_at_pitch = magnitudes[pitches[:, t].argmax(), t]

        # Calculate RMS for the audio chunk corresponding to this pitch detection frame
        start_sample = t * HOP_SIZE
        end_sample = min(start_sample + HOP_SIZE, len(audio_segment))

        # IMPORTANT: Add a check here to ensure the frame is not empty
        if end_sample <= start_sample: # This means the frame is empty or invalid
            continue # Skip this iteration if the frame is not valid

        frame_for_rms = audio_segment[start_sample:end_sample]

        # Already added this check, but crucial for clarity:
        if len(frame_for_rms) == 0:
            continue

        # Calculate RMS - this is where the error occurred
        # Ensure frame_for_rms is not empty BEFORE calculating np.mean
        rms_for_frame = np.sqrt(np.mean(frame_for_rms**2))


        # Apply both RMS threshold and magnitude threshold
        if pitch_hz > 50 and magnitude_at_pitch > 0.1 and rms_for_frame > RMS_THRESHOLD:
            pitch_midi = freq_to_midi(pitch_hz)
            if pitch_midi is not None:
                all_detected_notes.append((pitch_hz, pitch_midi, midi_to_note_name(pitch_midi)))

    if not all_detected_notes:
        return [], [], [] # No significant pitch detected

    if target_scale_name:
        if target_scale_name not in SCALES:
            print(f"Error: Target Scale '{target_scale_name}' not found. Available scales: {list(SCALES.keys())}")
            return all_detected_notes, [], []

        target_scale_midi_notes = SCALES[target_scale_name]
        in_scale_notes_data = []
        out_of_scale_notes_data = []

        for hz, midi_pitch, note_name in all_detected_notes:
            if is_pitch_in_scale(midi_pitch, target_scale_midi_notes, PITCH_TOLERANCE_MIDI_STEPS):
                in_scale_notes_data.append((hz, midi_pitch, note_name))
            else:
                out_of_scale_notes_data.append((hz, midi_pitch, note_name))
        return all_detected_notes, in_scale_notes_data, out_of_scale_notes_data
    else:
        return all_detected_notes, [], []

def record_audio(duration, sr=SAMPLE_RATE):
    """
    Records audio from the microphone for a given duration.
    Requires PyAudio.
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1 # Mono

    p = pyaudio.PyAudio()

    stream = None # Initialize stream to None
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=sr,
                        input=True,
                        frames_per_buffer=CHUNK)
    except Exception as e:
        print(f"Error opening audio stream: {e}")
        p.terminate()
        return np.array([], dtype=np.float32) # Return empty array if stream fails to open

    print(f"Recording for {duration} seconds...")

    frames = []
    for _ in range(0, int(sr / CHUNK * duration)):
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        except IOError as e:
            print(f"IOError during recording: {e}")
            # Consider breaking or logging, but for robustness, we'll continue with what we have
            break # Exit loop if there's an issue with the stream

    print("Recording finished.")

    if stream: # Only stop and close if stream was successfully opened
        stream.stop_stream()
        stream.close()
    p.terminate()

    if not frames: # If no frames were recorded
        print("Warning: No audio frames were recorded.")
        return np.array([], dtype=np.float32) # Return empty array

    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    audio_data_float = audio_data.astype(np.float32) / 32768.0 # Normalize to -1.0 to 1.0

    return audio_data_float

# --- Main Program Logic Functions ---

def print_scale_analysis_results(all_detected_data, in_scale_data, out_of_scale_data, target_scale_name, best_matching_scale, best_match_score):
    """Helper function to print detailed analysis results."""
    total_pitches = len(all_detected_data)

    if total_pitches > 0:
        print(f"\n--- Analysis Summary ---")
        print(f"Total detected pitch frames (after noise filtering): {total_pitches}")

        if target_scale_name:
            if total_pitches > 0:
                in_scale_percentage = (len(in_scale_data) / total_pitches) * 100
                print(f"Target Scale: {target_scale_name}")
                print(f"Pitches in target scale: {len(in_scale_data)} ({in_scale_percentage:.2f}%)")
                print(f"Pitches out of target scale: {len(out_of_scale_data)}")

                if in_scale_percentage >= 80: # Arbitrary threshold for "good"
                    print("Verdict (Target Scale): Your singing generally stays within the target scale. Good job!")
                else:
                    print("Verdict (Target Scale): Consider practicing more to stay consistently within the target scale.")
            else:
                print("No significant vocal pitches detected for target scale analysis.")
        
        print(f"Best Matching Overall Scale: {best_matching_scale} (Fit Score: {best_match_score:.4f} MIDI deviation)")
        print(f"  A lower fit score indicates a closer match to the scale's notes.")

        print(f"\n--- All Detected Notes by Occurrence (filtered) ---")
        if all_detected_data:
            all_note_counts = collections.Counter(item[2] for item in all_detected_data)
            print(f"{'Note Name':<15} {'Occurrences':<15}")
            print(f"{'-'*15:<15} {'-'*15:<15}")
            for note_name, count in all_note_counts.most_common():
                print(f"{note_name:<15} {count:<15}")
        else:
            print("  No significant notes detected after filtering.")

        if target_scale_name:
            print(f"\n--- In-Target-Scale Notes (within {PITCH_TOLERANCE_CENTS} cents of a target scale note, filtered) ---")
            if in_scale_data:
                in_scale_counts = collections.Counter(item[2] for item in in_scale_data)
                print(f"{'Note Name':<15} {'Occurrences':<15}")
                print(f"{'-'*15:<15} {'-'*15:<15}")
                for note_name, count in in_scale_counts.most_common():
                    print(f"{note_name:<15} {count:<15}")
            else:
                print("  No notes detected within the chosen target scale after filtering.")

            print(f"\n--- Out-of-Target-Scale Notes (notes not matching the target scale within tolerance, filtered) ---")
            if out_of_scale_data:
                out_of_scale_counts = collections.Counter(item[2] for item in out_of_scale_data)
                print(f"{'Note Name':<15} {'Occurrences':<15}")
                print(f"{'-'*15:<15} {'-'*15:<15}")
                for note_name, count in out_of_scale_counts.most_common():
                    print(f"{note_name:<15} {count:<15}")
            else:
                print("  All detected notes were within the chosen target scale after filtering, or no notes detected.")

    else:
        print("No significant vocal pitches detected in the audio after noise filtering.")

def analyze_singing_voice_from_file(audio_file_path, target_scale_name=None):
    """
    Analyzes a pre-recorded audio file for singing voice scale adherence
    and finds the best matching scale.
    """
    try:
        y, sr = librosa.load(audio_file_path, sr=SAMPLE_RATE)
        print(f"\nAnalyzing '{audio_file_path}' (Sample Rate: {sr} Hz)...")
        
        all_detected_notes, in_scale, out_of_scale = analyze_vocal_segment(y, sr, target_scale_name)
        
        best_scale, best_score = find_best_matching_scale(all_detected_notes)

        print_scale_analysis_results(all_detected_notes, in_scale, out_of_scale, target_scale_name, best_scale, best_score)
    except FileNotFoundError:
        print(f"Error: Audio file not found at '{audio_file_path}'")
    except Exception as e:
        print(f"An unexpected error occurred while loading or processing the audio file: {e}")

def analyze_singing_voice_live(duration_seconds, target_scale_name=None):
    """
    Records live audio and analyzes it for singing voice scale adherence
    and finds the best matching scale.
    """
    print(f"Starting live recording for {duration_seconds} seconds...")
    audio_data = record_audio(duration_seconds)

    print(f"\nAnalyzing live recording (Sample Rate: {SAMPLE_RATE} Hz)...")
    all_detected_notes, in_scale, out_of_scale = analyze_vocal_segment(audio_data, SAMPLE_RATE, target_scale_name)
    
    best_scale, best_score = find_best_matching_scale(all_detected_notes)

    print_scale_analysis_results(all_detected_notes, in_scale, out_of_scale, target_scale_name, best_scale, best_score)


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Available Scales ---")
    available_scale_names = sorted(list(SCALES.keys()))
    for i, scale in enumerate(available_scale_names):
        print(f"{i+1}. {scale}")

    print("\nChoose an option:")
    print("1. Analyze a pre-recorded audio file")
    print("2. Analyze live singing")

    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        file_path = input("Enter the path to your audio file (e.g., my_singing.wav): ")
        
        target_choice = input("Do you want to specify a target scale? (yes/no): ").lower()
        if target_choice == 'yes':
            scale_name = input("Enter the target scale from the list above (e.g., C Major, A Minor): ")
            analyze_singing_voice_from_file(file_path, scale_name)
        else:
            analyze_singing_voice_from_file(file_path) # Analyze without a specific target scale
            
    elif choice == '2':
        try:
            duration = float(input("Enter recording duration in seconds (e.g., 5): "))
            target_choice = input("Do you want to specify a target scale? (yes/no): ").lower()
            if target_choice == 'yes':
                scale_name = input("Enter the target scale from the list above (e.g., C Major, A Minor): ")
                analyze_singing_voice_live(duration, scale_name)
            else:
                analyze_singing_voice_live(duration) # Analyze without a specific target scale
        except ValueError as e:
            print(f"Invalid duration. Please enter a number. {e}")
    else:
        print("Invalid choice.")