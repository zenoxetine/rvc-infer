import string, random, sys, subprocess
import numpy as np
import ffmpeg

def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()

def gen_random_string(N=8):
    res = ''.join(random.choices(string.ascii_lowercase + string.digits, k=N))
    return res

def remove_last_if_its_a_space(text):
    if (text[-1] == " "):
        text = text[:-1]
        return remove_last_if_its_a_space(text)
    else:
        return text

def valid_dir_or_filename(name):
    notAllowedCharacter = ['\\', '/', ':', '*', '?', '"', '<', '>', '|', '.']
    for charTest in notAllowedCharacter:
        if (charTest in name):
            if (charTest == '/'):
                name = name.replace('/', " (or) ")
            else:
                name = name.replace(charTest, "")
    name = remove_last_if_its_a_space(name)
    return name

def execute(command):
    output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return f"{output.stdout.decode()}\n{output.stderr.decode()}"