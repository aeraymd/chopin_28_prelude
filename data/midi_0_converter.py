import mido
import os


format_1_directory_path = '../raw_data/midi_format_1/'
format_0_directory_path = '../raw_data/midi_format_0_converted/'
new_format = 'format0_'

def midi_0_converter():
    for filename in os.listdir(format_1_directory_path):
        f = os.path.join(format_1_directory_path,filename)
        m = mido.MidiFile(f)
        tracks = mido.merge_tracks(m.tracks)
        type0_mid = mido.MidiFile(type=0)
        type0_mid.tracks.append(tracks)
        new_filename = new_format + filename
        type0_mid.save(os.path.join(format_0_directory_path, new_filename))
    return None
