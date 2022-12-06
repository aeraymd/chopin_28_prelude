import mido
import os


format_1_directory_path = './raw_data/midi_format_1/'
format_0_directory_path = './raw_data/midi_format_0_converted/'
new_format = 'format0_'


for filename in os.listdir(format_1_directory_path):
    #get file path
    f = os.path.join(format_1_directory_path,filename)
    #generate a mido file and get all its tracks
    m = mido.MidiFile(f)
    tracks = mido.merge_tracks(m.tracks)
    #create a new mido file (format 0) and append all tracks
    type0_mid = mido.MidiFile(type=0)
    type0_mid.tracks.append(tracks)
    #save new midi file in format_0_directory
    new_filename = new_format + filename
    type0_mid.save(os.path.join(format_0_directory_path, new_filename))

