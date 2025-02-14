import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import no_grad, LongTensor
import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence, _clean_text
from mel_processing import spectrogram_torch
from flask import Flask, request, send_file
from flask_cors import CORS
import io
import soundfile as sf
import gc

limitation = os.getenv("SYSTEM") == "spaces"  # limit text and audio length in huggingface spaces

def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(text, speaker, speed, is_symbol):
        if limitation:
            text_len = len(re.sub("\[([A-Z]{2})\]", "", text))
            max_len = 150
            if is_symbol:
                max_len *= 3
            if text_len > max_len:
                return "Error: Text is too long", None

        speaker_id = speaker_ids[speaker]
        stn_tst = get_text(text, hps, is_symbol)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        gc.collect()
        return "Success", (hps.data.sampling_rate, audio)

    return tts_fn

def create_vc_fn(model, hps, speaker_ids):
    def vc_fn(original_speaker, target_speaker, input_audio):
        if input_audio is None:
            return "You need to upload an audio", None
        sampling_rate, audio = input_audio
        duration = audio.shape[0] / sampling_rate
        if limitation and duration > 30:
            return "Error: Audio is too long", None
        original_speaker_id = speaker_ids[original_speaker]
        target_speaker_id = speaker_ids[target_speaker]

        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != hps.data.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=hps.data.sampling_rate)
        with no_grad():
            y = torch.FloatTensor(audio)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(y, hps.data.filter_length,
                                     hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                     center=False).to(device)
            spec_lengths = LongTensor([spec.size(-1)]).to(device)
            sid_src = LongTensor([original_speaker_id]).to(device)
            sid_tgt = LongTensor([target_speaker_id]).to(device)
            audio = model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][
                0, 0].data.cpu().float().numpy()
        del y, spec, spec_lengths, sid_src, sid_tgt
        gc.collect()
        return "Success", (hps.data.sampling_rate, audio)

    return vc_fn

def create_soft_vc_fn(model, hps, speaker_ids):
    def soft_vc_fn(target_speaker, input_audio):
        if input_audio is None:
            return "You need to upload an audio", None
        sampling_rate, audio = input_audio
        duration = audio.shape[0] / sampling_rate
        if limitation and duration > 30:
            return "Error: Audio is too long", None
        target_speaker_id = speaker_ids[target_speaker]

        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        if sampling_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        with torch.inference_mode():
            units = hubert.units(torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0).to(device))
        with no_grad():
            unit_lengths = LongTensor([units.size(1)]).to(device)
            sid = LongTensor([target_speaker_id]).to(device)
            audio = model.infer(units, unit_lengths, sid=sid, noise_scale=.667,
                                noise_scale_w=0.8)[0][0, 0].data.cpu().float().numpy()
        del units, unit_lengths, sid
        gc.collect()
        return "Success", (hps.data.sampling_rate, audio)

    return soft_vc_fn

def create_to_symbol_fn(hps):
    def to_symbol_fn(is_symbol_input, input_text, temp_text):
        return (_clean_text(input_text, hps.data.text_cleaners), input_text) if is_symbol_input \
            else (temp_text, temp_text)

    return to_symbol_fn

def load_model(model_path, config_path, device):
    hps = utils.get_hparams_from_file(config_path)
    model = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    utils.load_checkpoint(model_path, model, None)
    model.eval().to(device)
    if isinstance(hps.speakers, utils.HParams):
        speakers, speaker_ids = zip(*hps.speakers.items())
    else:
        speaker_ids = [sid for sid, name in enumerate(hps.speakers) if name != "None"]
        speakers = [name for sid, name in enumerate(hps.speakers) if name != "None"]
    return model, hps, speakers, speaker_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft", trust_repo=True).to(device)

    flask_app = Flask(__name__)
    CORS(flask_app)  # Enable CORS for all routes

    @flask_app.route('/tts', methods=['POST', 'OPTIONS'])
    def tts():
        if request.method == 'OPTIONS':
            return '', 204

        data = request.json
        text = data['text']
        speaker = data['speaker']
        speed = data['speed']
        is_symbol = data.get('is_symbol', False)
        model_id = data.get('model_id', '0')  # Assume model_id is provided in the request as a number

        # Load model on demand
        config_path = f"saved_model/{model_id}/config.json"
        model_path = f"saved_model/{model_id}/model.pth"
        model, hps, speakers, speaker_ids = load_model(model_path, config_path, device)

        tts_fn = create_tts_fn(model, hps, speaker_ids)
        message, (sampling_rate, audio) = tts_fn(text, speaker, speed, is_symbol)

        if message == "Success":
            # Save audio to a bytes buffer
            buffer = io.BytesIO()
            sf.write(buffer, audio, sampling_rate, format='WAV')
            buffer.seek(0)
            return send_file(buffer, mimetype='audio/wav')
        else:
            return message, 400

    flask_app.run(debug=True, port=5000)
