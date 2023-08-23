

import os
import sys

import joblib
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F


class GSLM(nn.Module):
    def __init__(self, YOUR_FAIRSEQ_PATH, base_path, hubert_checkpoint_path, kmeans_model_path, tts_model_path, glow_model_path,
                 glow_directory, code_dict_path, device='cuda', decode_only=False,
                 max_decoder_steps=86 * 5, use_denoiser=False):  # melspec sr = 22050/256~86
        super(GSLM, self).__init__()

        sys.path.append(YOUR_FAIRSEQ_PATH)

        from examples.textless_nlp.gslm.unit2speech.tts_data import (
            TacotronInputDataset,
        )
        from examples.textless_nlp.gslm.unit2speech.utils import (
            load_tacotron,
            load_waveglow,
        )

        # append the path to waveglow directory and model binaries
        sys.path.append(glow_directory)
        hubert_checkpoint_path = os.path.join(base_path, hubert_checkpoint_path)
        kmeans_model_path = os.path.join(base_path, kmeans_model_path)
        tts_model_path = os.path.join(base_path, tts_model_path)
        code_dict_path = os.path.join(base_path, code_dict_path)
        glow_model_path = os.path.join(base_path, glow_model_path)

        # define parameters
        self.device = device
        self.num_embeddings = int(code_dict_path.split("_")[-1])

        # load hubert encoder model
        import fairseq
        (hubert_model, cfg, self.task,) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [hubert_checkpoint_path])
        if not decode_only:
            self.hubert_model = hubert_model[0].eval().to(self.device)
        self.kmeans_model = joblib.load(open(kmeans_model_path, "rb"))
        self.kmeans_model.verbose = False

        # load tacotron decoder model
        self.tacotron_model, self.sample_rate, self.hparams = load_tacotron(
            tacotron_model_path=tts_model_path,
            max_decoder_steps=max_decoder_steps,
            device=device
        )
        self.tacotron_model = self.tacotron_model.eval().to(self.device)
        self.hparams.code_dict = code_dict_path
        self.tts_dataset = TacotronInputDataset(self.hparams)
        print("Tacotron sampling rate: ", self.sample_rate)

        # load waveglow decoder model
        self.waveglow, denoiser = load_waveglow(waveglow_path=glow_model_path, device=device, use_denoiser=use_denoiser)
        self.waveglow = self.waveglow.remove_weightnorm(self.waveglow).eval().to(device)

    def encode(self, speech):
        """ Encode speech waveform to features and features to discrete units"""
        with torch.no_grad():
            if self.task.cfg.normalize:
                speech = F.layer_norm(speech, speech.shape)
            feat_chunk, _ = self.hubert_model.extract_features(source=speech.squeeze(1),  # BxT
                                                               padding_mask=None,
                                                               mask=False,
                                                               output_layer=-1,
                                                               )
            unit = torch.from_numpy(self.kmeans_model.predict(
                feat_chunk.reshape([-1, feat_chunk.shape[-1]]).detach().cpu().numpy()).reshape(
                feat_chunk[..., 0].shape)).to(device=speech.device, dtype=torch.int64)
            encoding_onehot = F.one_hot(unit, self.num_embeddings)
        return feat_chunk, encoding_onehot, unit

    def encode_feature(self, feat_chunk):
        """ Encode features to discrete units"""
        with torch.no_grad():
            unit = torch.from_numpy(self.kmeans_model.predict(
                feat_chunk.reshape([-1, feat_chunk.shape[-1]]).detach().cpu().numpy()).reshape(
                feat_chunk[..., 0].shape)).to(device=feat_chunk.device, dtype=torch.int64)
            encoding_onehot = F.one_hot(unit, self.num_embeddings)
        return feat_chunk, encoding_onehot, unit

    def decode(self, unit, return_wave=False):
        """ Decode mel features from discrete units and then speech from mel features"""
        with torch.no_grad():
            mel_all = []
            aud_all = []
            unit = unit.detach().cpu().numpy()
            for b in range(unit.shape[0]):
                quantized_units_str = " ".join(map(str, unit[b]))
                tts_input = self.tts_dataset.get_tensor(quantized_units_str).to(self.device)
                # tts_input = self.tts_dataset.get_tensor(quantized_units_str).to('cuda:1')
                _, mel, _, ali, has_eos = self.tacotron_model.inference(tts_input.unsqueeze(0), None, ret_has_eos=True)
                if return_wave:
                    mel = mel.to(self.device)
                    # mel = mel.half().to(self.device)
                    aud = self.waveglow.infer(mel, sigma=0.666)
                    # aud = FA.resample(aud,self.sample_rate,16000)
                    aud_all.append(aud)
                mel_all.append(mel)
        if return_wave:
            return mel_all, aud_all
        else:
            return [mel_all]

    def decode_mel(self, unit):
        """ Decode mel features from discrete units"""
        with torch.no_grad():
            mel_all = []
            unit = unit.detach().cpu().numpy()
            for b in range(unit.shape[0]):
                quantized_units_str = " ".join(map(str, unit[b]))
                tts_input = self.tts_dataset.get_tensor(quantized_units_str).to(self.device)
                _, mel, _, ali, has_eos = self.tacotron_model.inference(tts_input.unsqueeze(0), None, ret_has_eos=True)
                mel_all.append(mel)
        return mel_all

    def decode_wav(self, mel, griffinlim=True):
        """ Decode speech from mel features"""
        with torch.no_grad():
            aud_all = []
            for b in range(mel.shape[0]):
                if griffinlim:
                    x = torch.exp(mel[b])
                    stft = librosa.feature.inverse.mel_to_stft(x.float().detach().cpu().numpy(), n_fft=1024, fmin=0,
                                                               fmax=8000)
                    aud = librosa.griffinlim(stft, win_length=1024, hop_length=256)
                else:
                    mel = mel.to(self.device)
                    aud = self.waveglow.infer(mel, sigma=0.666)
                aud_all.append(aud)
        return aud_all
