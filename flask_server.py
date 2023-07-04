

from flask import Flask, request
from flask_cors import CORS
from deepspeech import Model
import numpy as np


app = Flask(__name__)
CORS(app)
model_path = './output_graph_imams_tusers_v2.pb'
scorer_path = './quran.scorer'
model = Model(model_path)
model.enableExternalScorer(scorer_path)

import numpy as np

def downsample_audio(data, sample_rate, target_sample_rate):
    # Convert bytes to NumPy array
    audio_data = np.frombuffer(data, dtype=np.int16)

    # Calculate the conversion ratio
    ratio = sample_rate / target_sample_rate

    # Determine the new length of the downsampled audio
    new_length = int(len(audio_data) / ratio)

    # Perform downsampling using NumPy's resample function
    downsampled_data = np.interp(
        np.linspace(0, len(audio_data) - 1, new_length),
        np.arange(len(audio_data)),
        audio_data
    )

    # Convert the data type to int16
    downsampled_data = downsampled_data.astype(np.int16)

    return downsampled_data



@app.route('/api/audio', methods=['POST'])
def process_audio():
    audio_file = request.files['audio']
    if audio_file:
        audio_data = audio_file.read()
        sample_rate = 48000
        target_sample_rate = 16000

        # Perform downsampling
        audio_data = downsample_audio(audio_data, sample_rate, target_sample_rate)

        # Perform speech-to-text using DeepSpeech model
        transcription = model.stt(audio_data)
        return transcription
        
    else:
        return 'No audio file received'

if __name__ == '__main__':
    app.run(host='localhost', port=3300)

