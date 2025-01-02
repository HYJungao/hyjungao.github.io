---
title: MP3 transcription on Colab
date: 2025-01-01 20:17:00 +0800
categories: [Tools]
tags: [stt, ai, ml, tools]
---

### Repository

[MP3 Transcription](https://github.com/HYJungao/tools/blob/main/mp3%20transcription.ipynb)

<pre><code class="language-python">
import os
import whisper
from pydub import AudioSegment

# load Whisper-large
model = whisper.load_model("large")

def save_as_srt(transcription_result, srt_filename):
    with open(srt_filename, 'w') as f:
        for i, segment in enumerate(transcription_result['segments']):
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']

            # convert to srt timestamp (h: m: s: ms)
            start_time_srt = format_srt_time(start_time)
            end_time_srt = format_srt_time(end_time)

            f.write(f"{i + 1}\n")
            f.write(f"{start_time_srt} --> {end_time_srt}\n")
            f.write(f"{text}\n\n")

# convert second timestamp to SRT timestamp
def format_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)

    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

# process all mp3 in the directory
def process_audio_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.mp3'):
            mp3_file = os.path.join(input_folder, file_name)
            wav_file = os.path.join(output_folder, file_name.replace('.mp3', '.wav'))
            srt_file = os.path.join(output_folder, file_name.replace('.mp3', '.srt'))

            print(f"processing: {mp3_file}")

            try:
                # convert mp3 to wav
                audio = AudioSegment.from_mp3(mp3_file)
                audio.export(wav_file, format="wav")

                # transcribe
                result = model.transcribe(wav_file)

                save_as_srt(result, srt_file)

                print(f"complete: {mp3_file} -> {srt_file}")
            except Exception as e:
                print(f"failed: {mp3_file}, error: {e}")

input_folder = "./"  # change this variable when use
output_folder = "./"  # change this variable when use

process_audio_files(input_folder, output_folder)
</code></pre>
