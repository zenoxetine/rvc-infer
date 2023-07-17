from os.path import isdir, isfile
from os import system, remove, mkdir
from yt_dlp import YoutubeDL
import ffmpeg
from my_utils import gen_random_string, valid_dir_or_filename

def yt2wav(url, audio_name=""):
    if (not url.startswith("http")):
        url = f"https://{url}"
        
    if (audio_name == ""):
        with YoutubeDL() as ydl: 
            info_dict = ydl.extract_info(url, download=False)
            video_title = info_dict.get('title', gen_random_string())
            audio_name = valid_dir_or_filename(video_title)
            
    if not isdir("audio"):
        mkdir("audio")

    ydl_opts = {
        'format': 'bestaudio/best',
        # 'outtmpl': 'output.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        "outtmpl": f'./audio/{audio_name}',
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        return f"./audio/{audio_name}.wav"
    
def separate_from_youtube_video(url, audio_name=""):
    if (not url.startswith("http")):
        url = f"https://{url}"

    if (audio_name == ""):
        with YoutubeDL() as ydl: 
            info_dict = ydl.extract_info(url, download=False)
            video_title = info_dict.get('title', gen_random_string())
            audio_name = valid_dir_or_filename(video_title)
    audio_name = f"demucs_{audio_name}"

    if not isdir("audio"):
        mkdir("audio")
    
    yt2wav(url, audio_name)
    
    print("[*] running demucs ...")
    system(f"demucs --two-stems=vocals \"./audio/{audio_name}.wav\" -o \"./audio/\"")
    print("[*] demucs done.")

    remove(f"audio/{audio_name}.wav")

    vocals_path = f"audio/htdemucs/{audio_name}/vocals.wav"
    no_vocals_path = f"audio/htdemucs/{audio_name}/no_vocals.wav"

    if (isfile(vocals_path) and isfile(no_vocals_path)):
        return vocals_path, no_vocals_path
    else:
        return "", ""
