import time
import random
import numpy as np
import pygame 

# --- PYGAME AUDIO SETUP ---
SAMPLE_RATE = 44100
STANDARD_FREQ = 880  # A5 for the standard click
DOWNBEAT_FREQ = 1800  # E5 for the first beat (downbeat)
DURATION = 0.05

# 1. Initialize Pygame Mixer (Crucial step)
try:
    # Use a smaller buffer for better timing precision
    pygame.mixer.init(frequency=SAMPLE_RATE, channels=1, buffer=128) 
except pygame.error as e:
    print(f"Error initializing Pygame Mixer: {e}")
    # Handle case where audio hardware is unavailable

def generate_cached_click_sound(frequency):
    """Generates a pygame Sound object ONCE at the specified frequency."""
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)
    
    # Generate sine wave with a quick fade-out
    amplitude_envelope = np.exp(-t * 25) 
    note = np.sin(frequency * t * 2 * np.pi) * amplitude_envelope
    
    # Convert to 16-bit integer format (signed short)
    audio = note * (2**15 - 1) / np.max(np.abs(note))
    audio = audio.astype(np.int16)
    
    # Create the pygame Sound object
    return pygame.mixer.Sound(audio.tobytes())

# Cache two distinct sound objects globally
STANDARD_CLICK = generate_cached_click_sound(STANDARD_FREQ)
DOWNBEAT_CLICK = generate_cached_click_sound(DOWNBEAT_FREQ)


def play_real_click(is_downbeat=False):
    """
    Plays the standard click or the distinct downbeat click.
    Prints the click symbol to the console.
    """
    if is_downbeat:
        DOWNBEAT_CLICK.play()
        # print(" [DOWNBEAT!]", end="") 
    else:
        STANDARD_CLICK.play()
        # print(" [click]", end="") 
# ---------------------------

def generate_single_minor_pentatonic_interval(bpm):
    """
    Generates a random interval on every 4th beat, controlled by BPM.
    The first beat of the metronome has a distinct sound.
    """
    
    MINOR_PENTATONIC_INTERVALS = [
        "Root (R)", "Minor 3rd (m3)", "Perfect 4th (P4)", 
        "Perfect 5th (P5)", "Minor 7th (m7)"
    ]
    
    time_per_beat = 60 / bpm
    
    print("-" * 50)
    print(f"BPM set to {bpm}. Time per beat: {time_per_beat:.3f} seconds.")
    print(f"Scale Intervals: {MINOR_PENTATONIC_INTERVALS}")
    print("-" * 50)

    try:
        # Generate the FIRST interval before the loop starts
        initial_interval = random.choice(MINOR_PENTATONIC_INTERVALS)
        
        # Output the initial interval, treated as the first beat (Beat 1 of the first measure)
        print(f"Beat 1: Interval: {initial_interval}", end="")
        play_real_click(is_downbeat=True) # Play the downbeat click
        print("\n" + "-" * 50) 

        # Loop for the Metronome
        beat_count = 1 # Start count at 1, as the first beat was played above
        print("Continuing Metronome (Ctrl+C to stop)...")
        while True:
            time.sleep(time_per_beat)
            
            beat_count += 1
            is_downbeat = (beat_count % 4 == 1) # Beat 1, 5, 9, etc., are downbeats

            # Print the beat number
            # print(f"Beat {beat_count}:", end="")
            print(".")
            
            # Check if this beat is the one where a new interval should be generated (Beat 4, 8, 12, etc.)
            # The modulo 4 check in the original code means 4, 8, 12... which are *upbeats* in 4/4 time.
            # I'll update it to be Beat 1 for better musical context (1, 5, 9, etc.)
            if is_downbeat:
                random_interval = random.choice(MINOR_PENTATONIC_INTERVALS)
                # Output the interval and the click sound
                print(f"Interval: {random_interval}", end="")
            else:
                # Print a placeholder if no interval is generated on this beat
                print(" " * 19, end="")
                
            # Play the appropriate click sound
            play_real_click(is_downbeat) 
            print() # Start a new line for the next beat

    except KeyboardInterrupt:
        print("\n" + "-" * 50)
        print("Metronome stopped.")
    finally:
        pygame.mixer.quit()


# --- Program Execution ---

# Configuration:
TARGET_BPM = 40

# Run the function
generate_single_minor_pentatonic_interval(TARGET_BPM)