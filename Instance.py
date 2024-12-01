import os
from pydub import AudioSegment


def split_audio(input_folder, output_folder, segment_length_ms=10):
    """
    Splits all WAV files in the input_folder into x-second segments
    and saves them in the output_folder.

    :param input_folder: Path to the folder containing the original WAV files.
    :param output_folder: Path to the folder where the split files will be saved.
    :param segment_length_ms: Length of each segment in milliseconds.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            file_path = os.path.join(input_folder, filename)
            audio = AudioSegment.from_wav(file_path)

            # Calculate the number of segments
            num_segments = len(audio) // segment_length_ms

            # Get the base name
            base_name = os.path.splitext(filename)[0]

            # Loop through each segment and export
            for i in range(num_segments):
                start_time = i * segment_length_ms
                end_time = start_time + segment_length_ms
                segment = audio[start_time:end_time]

                # Save the segment
                output_filename = f"{base_name}#{i + 1}.wav"
                output_path = os.path.join(output_folder, output_filename)
                segment.export(output_path, format="wav", parameters=["-ar", str(44100)])
                print(f"Exported: {output_filename}")



input_folder = 'Cricket_Seg/Cricket05'
output_folder = 'Instance/Cricket/'

split_audio(input_folder, output_folder)
