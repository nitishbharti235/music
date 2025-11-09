import collections

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
    # We will generate scales from MIDI note 48 (C3) to keep them within a reasonable
    # range for general musical analysis, but you can adjust the starting octave.
    chromatic_notes = {
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
    # Major Scale: W-W-H-W-W-W-H (Whole-Whole-Half-Whole-Whole-Whole-Half)
    MAJOR_INTERVALS = [2, 2, 1, 2, 2, 2, 1]
    # Natural Minor Scale: W-H-W-W-H-W-W (Whole-Half-Whole-Whole-Half-Whole-Whole)
    NATURAL_MINOR_INTERVALS = [2, 1, 2, 2, 1, 2, 2]

    all_scales = collections.OrderedDict()

    # Generate Major Scales
    for midi_offset in range(12): # Iterate through 12 possible starting notes (C, C#, D, ..., B)
        # Determine the root note name
        root_info = chromatic_notes[midi_offset]
        root_name = root_info["name"]

        # Starting MIDI note for the scale. We'll use an octave around Middle C (C4=60).
        # C4 (MIDI 60) for C Major, C#4 (MIDI 61) for C# Major, etc.
        start_note_midi = 60 + midi_offset

        # Generate the major scale
        major_scale_notes = generate_scale(start_note_midi, MAJOR_INTERVALS)
        all_scales[f"{root_name} Major"] = major_scale_notes

        # Add enharmonic major scales if applicable (e.g., Db Major for C# Major)
        if root_info["enharmonic"]:
            enharmonic_name = root_info["enharmonic"]
            # For enharmonic names, ensure we use a consistent starting octave/MIDI note
            # We're still using the same start_note_midi as it refers to the same pitch.
            all_scales[f"{enharmonic_name} Major"] = major_scale_notes

    # Generate Natural Minor Scales
    for midi_offset in range(12):
        root_info = chromatic_notes[midi_offset]
        root_name = root_info["name"]

        # Starting MIDI note for the scale.
        start_note_midi = 60 + midi_offset # Same as for Major scales

        # Generate the natural minor scale
        minor_scale_notes = generate_scale(start_note_midi, NATURAL_MINOR_INTERVALS)
        all_scales[f"{root_name} Minor"] = minor_scale_notes

        # Add enharmonic minor scales if applicable
        if root_info["enharmonic"]:
            enharmonic_name = root_info["enharmonic"]
            all_scales[f"{enharmonic_name} Minor"] = minor_scale_notes

    return all_scales

def midi_to_note_name(midi_note):
    """Converts a MIDI note number to its common note name (e.g., 60 -> C4)."""
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi_note // 12) - 1 # MIDI note 0 is C-1, so C4 is octave 4
    note_index = midi_note % 12
    return f"{note_names[note_index]}{octave}"

if __name__ == "__main__":
    generated_scales = get_common_scales()

    print("--- Common Major and Natural Minor Scales (MIDI Note Numbers and Note Names) ---")
    for scale_name, notes in generated_scales.items():
        note_names = [midi_to_note_name(n) for n in notes]
        print(f"\n{scale_name}:")
        print(f"  MIDI Notes: {notes}")
        print(f"  Note Names: {note_names}")

    print("\n--- Example: C Major Scale from generated_scales['C Major'] ---")
    c_major = generated_scales['C Major']
    print(f"C Major MIDI: {c_major}")
    print(f"C Major Notes: {[midi_to_note_name(n) for n in c_major]}")

    print("\n--- Example: A Minor Scale from generated_scales['A Minor'] ---")
    a_minor = generated_scales['A Minor']
    print(f"A Minor MIDI: {a_minor}")
    print(f"A Minor Notes: {[midi_to_note_name(n) for n in a_minor]}")

    print("\n--- You can now integrate this 'generated_scales' dictionary into your singing analysis program. ---")
