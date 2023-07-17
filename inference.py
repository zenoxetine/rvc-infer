import sys, os
from datetime import datetime
print("[*] importing vc_infer_pipeline ...")
from vc_infer_pipeline import VC
print("[*] done import VC.")
from fairseq import checkpoint_utils
import torch
#import librosa
from scipy.io import wavfile
from config import Config
from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
#from my_utils import load_audio

class Inference:
    def __init__(self):
        self.pth_file = ""
        self.initialized = False
        self.config = None
        self.hubert_model = None
        self.model = None
        self.net_g = None
        self.sid = 0
        self.vc = None
        self.ready = False
    
    def initialize(self, device):
        try:
            self.config = Config(device)
            print("[*] device: %s" % self.config.device)
            print("[*] is_half: %s" % self.config.is_half)

            print("[*] loading hubert ...")
            models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(["hubert_base.pt"],suffix="",)
            self.hubert_model = models[0].to(self.config.device)
            if(self.config.is_half):
                self.hubert_model = self.hubert_model.half()
            else:
                self.hubert_model = self.hubert_model.float()
            self.hubert_model.eval()

            self.initialized = True
        except Exception as e:
            print(e)
    
    def load_rvc_model(self, pth_file, index_file=None):
        self.ready = False
        # clean last model if exist
        if (self.model != None):
            del self.model
            del self.vc
            del self.net_g
            self.pth_file = ""

        print("[*] loading pth %s"%pth_file)
        self.model = torch.load(pth_file, map_location="cpu")
        self.model["config"][-3] = self.model["weight"]["emb_g.weight"].shape[0] # n_spk
        if_f0 = self.model.get("f0",1)
        version = self.model.get("version", "v1")
        if version == "v1":
            if if_f0 == 1:
                self.net_g = SynthesizerTrnMs256NSFsid(*self.model["config"], is_half=self.config.is_half)
            else:
                self.net_g = SynthesizerTrnMs256NSFsid_nono(*self.model["config"])
        elif version == "v2":
            if if_f0 == 1:#
                self.net_g = SynthesizerTrnMs768NSFsid(*self.model["config"], is_half=self.config.is_half)
            else:
                self.net_g = SynthesizerTrnMs768NSFsid_nono(*self.model["config"])
        del self.net_g.enc_q
        print(self.net_g.load_state_dict(self.model["weight"], strict=False))  # 不加这一行清不干净，真奇葩
        self.net_g.eval().to(self.config.device)
        if (self.config.is_half):
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()
        tgt_sr = self.model["config"][-1]
        self.vc = VC(tgt_sr, self.config)
        self.ready = True
        self.pth_file = pth_file
        print(f"[*] model {pth_file} is ready!")
        # n_spk=cpt["config"][-3]
        # return {"visible": True,"maximum": n_spk, "__type__": "update"}

    def do_infer(self,audio,audio_path,f0_up_key,f0_method,file_index,index_rate,filter_radius=3,resample_sr=0,rms_mix_rate=1, protect=0.3):
        # if not os.path.isfile(audio_path):
        #     print("[!] Wrong audio path")
        #     return None, None

        #if (file_index == None):
        #    file_index = ""

        # pass model component
        tgt_sr = self.model["config"][-1]
        version = self.model.get("version", "v1")
        if_f0 = self.model.get("f0",1)
        f0_up_key = int(f0_up_key)

        try:
            #audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            #audio = load_audio(audio_path, 16000)
            times = [0, 0, 0]
            print("[*] (calling pipeline)")
            audio_opt = self.vc.pipeline(
                self.hubert_model,
                self.net_g,
                self.sid,
                audio,
                audio_path,
                times,
                f0_up_key,
                f0_method, # harvest, pm, or crepe
                file_index,
                index_rate,
                if_f0,
                filter_radius,
                tgt_sr,
                resample_sr,
                rms_mix_rate,
                version,
                protect,
                f0_file=None
            )
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}]: npy: {times[0]}, f0: {times[1]}s, infer: {times[2]}s")
            return audio_opt, tgt_sr
        except Exception as e:
            print(e)
            return None, None