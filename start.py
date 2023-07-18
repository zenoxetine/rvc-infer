import os, argparse
print("[*] importing gradio ...")
import gradio as gr
from inference import Inference
from scipy.io import wavfile
import numpy as np
import librosa
from downloader import yt2wav, separate_from_youtube_video
from my_utils import gen_random_string, valid_dir_or_filename, execute

def search_model():
    folder_path = "models"
    if not os.path.isdir(folder_path): os.mkdir(folder_path)

    pth_path = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pth"):
                pth_path.append(os.path.join(root, file))
    print(f"[*] found {len(pth_path)} model(s) on {folder_path} dir")
    return pth_path

def search_index(model_name=""):
    folder_path = "models"
    if not os.path.isdir(folder_path): os.mkdir(folder_path)

    index_path = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".index") and model_name in file:
                index_path.append(os.path.join(root, file))
    return index_path

def do_infer(pth_path, input_audio, transpose, method, index_rate):
    if (pth_path == ""):
        raise gr.Error("SELECT MODEL FIRST!")
    if input_audio is None:
        raise gr.Error("INPUT AUDIO REQUIRED!")
    if (isinstance(input_audio, str)):
        input_audio = os.path.normpath(input_audio)
        if not os.path.isfile(input_audio):
            raise gr.Error("WRONG AUDIO FILE!")

    if os.path.isfile(pth_path):
        if pth_path != infer.pth_file:
            infer.load_rvc_model(pth_path)
    else:
        raise gr.Error("PTH FILE NOT FOUND!")

    if (infer.ready):
        audio = None
        transpose = int(transpose)
        if (isinstance(input_audio, str)):
            audio, sr = librosa.load(input_audio, sr=16000, mono=True)
            print(f"[*] got input audio {input_audio}")
        else:
            sampling_rate, audio = input_audio
            duration = audio.shape[0] / sampling_rate
            #if duration > 20 and limitation:
            #    return "Please upload an audio file that is less than 20 seconds.", None
            audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio.transpose(1, 0))
            if sampling_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)

        _, model_file = os.path.split(pth_path)
        model_name = ''.join(model_file.split('.')[:-1])
        index_file = None
        index_files = search_index(model_name=model_name)
        if (len(index_files) > 0):
            index_file = index_files[0]
            print(f"[*] using index found: {index_file}")

        audio_opt, tgt_sr = infer.do_infer(audio,input_audio,transpose,method,index_file,index_rate)
        if (tgt_sr != None):
            # saving audio output to file
            if not os.path.isdir("outputs"):
                os.mkdir("outputs")
            _, input_file = os.path.split(input_audio)
            input_file_title = ''.join(input_file.split('.')[:-1])
            out_file_path = f"./outputs/{input_file_title}_{model_name}_{method}_{transpose}.wav"
            wavfile.write(out_file_path, tgt_sr, audio_opt)

            return (tgt_sr, audio_opt), os.path.normpath(f"{os.getcwd()}/{out_file_path}")
        else:
            raise gr.Error("ERROR OCCURED!")
    else:
        raise gr.Error("MODEL IS NOT READY!")

def download_audio_from_yt(separate_mode, url):
    if (url == ''):
        raise gr.Error("INPUT YOUTUBE LINK!")

    current_dir = os. getcwd()
    if separate_mode:
        vocals, no_vocals = separate_from_youtube_video(url)
        
        if (vocals == ""):
            raise gr.Error("Failed outputing file!")
        else:
            abs_path_vocals = os.path.normpath(f"{current_dir}/{vocals}")
            abs_path_no_vocals = os.path.normpath(f"{current_dir}/{no_vocals}")
            return vocals, no_vocals, abs_path_vocals, abs_path_no_vocals
    else:
        audio_path = yt2wav(url)
        return audio_path, os.path.normpath(f"{current_dir}/{audio_path}")

def download_model_from_url(url="", unzip=True):
    if (url == ''):
        raise gr.Error("INPUT A VALID URL!")

    printed_output = ""
    if (unzip):
        if (url.startswith("http")):
            temp_file = gen_random_string()
            printed_output += execute(f"wget -nv -O {temp_file}.zip {url}")
            printed_output += execute(f"unzip {temp_file}.zip -d ./models/")
            os.remove(f"{temp_file}.zip")
        else:
            printed_output += execute(f"unzip {url} -d ./models/")
    else:
        if (url.startswith("http")):
            printed_output += execute(f"wget -nv -P ./models {url}")
        else:
            raise gr.Error("INVALID INPUT!")

    return printed_output

def change_sep_mode(separate_mode):
    if (separate_mode):
        return gr.Button.update(visible=True), gr.Button.update(visible=False), \
        gr.Audio.update(visible=False), gr.Textbox.update(visible=False), \
        gr.Audio.update(visible=True), gr.Textbox.update(visible=True), \
        gr.Audio.update(visible=True), gr.Textbox.update(visible=True),
    else:
        return gr.Button.update(visible=False), gr.Button.update(visible=True), \
        gr.Audio.update(visible=True), gr.Textbox.update(visible=True), \
        gr.Audio.update(visible=False), gr.Textbox.update(visible=False), \
        gr.Audio.update(visible=False), gr.Textbox.update(visible=False),

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true')
    parser.set_defaults(share=False)
    #parser.add_argument("--files", default=True, help="load audio from path")
    args = parser.parse_args()

    # initialize
    infer = Inference()
    infer.initialize("cuda:0")

    with gr.Blocks() as app:
        gr.Markdown("rvc model inference and vocals separator from youtube.")
        with gr.Tab("inference"):
            with gr.Row():
                with gr.Column():
                    pth_models = search_model()
                    f0_methods = ["pm", "harvest"]
                    if (infer.config.device.startswith("cuda")):
                        f0_methods.append("crepe")
                    model_list = None
                    with gr.Row():
                        model_list = gr.Dropdown(choices=pth_models, label="Select models", type="value", scale=3)
                        refresh_button = gr.Button("refresh", scale=1, variant="secondary")
                        refresh_button.click(lambda: gr.Dropdown.update(choices=search_model()) , outputs=model_list)
                    
                    audio_input_path = gr.Textbox(label="Audio path (.wav)") #if args.files else gr.Audio(label="Input audio")
                    transpose = gr.Number(label="Transpose (from -12 to 12)", value=0) #, minimum=-12, maximum=12)
                    pitch_algorithm = gr.Radio(
                            label="Pitch extraction algorithm",
                            choices=f0_methods,
                            value="pm",
                            interactive=True,
                        )
                    ret_ratio = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="Retrieval feature ratio",
                            value=0.7,
                            interactive=True,
                        )
                    
                    infer_process_btn = gr.Button("process", variant="primary")
                with gr.Column():
                    infer_audio_output = gr.Audio(label="Audio Output")
                    infer_path_output = gr.Textbox(label="Output path", show_copy_button=True)

                infer_process_btn.click(fn=do_infer, inputs=[
                    model_list, audio_input_path, transpose, pitch_algorithm, ret_ratio
                ], outputs=[
                    infer_audio_output, infer_path_output
                ])
        with gr.Tab("download"):
            with gr.Row():
                with gr.Column():
                    youtube_url = gr.Textbox(label="Download audio from youtube", placeholder="https://www.youtube.com/watch?v=NAId7IP2bR4" ,interactive=True)
                    separate_mode = gr.Checkbox(value=True, label="separate vocals")
                    separate_process_btn = gr.Button("separate", variant="primary")
                    yt_download_btn = gr.Button("convert", visible=False, variant="primary")
                with gr.Column():
                    yt_download_output = gr.Audio(label="audio output", visible=False)
                    yt_download_path_output = gr.Textbox(label="", visible=False, show_copy_button=True)
                    demucs_vocals_output = gr.Audio(label="vocals output")
                    vocals_path_output = gr.Textbox(label="", show_copy_button=True)
                    demucs_no_vocals_output = gr.Audio(label="no vocals output")
                    no_vocals_path_output = gr.Textbox(label="", show_copy_button=True)

                separate_mode.change(change_sep_mode, [separate_mode], [
                    separate_process_btn, yt_download_btn,
                    yt_download_output, yt_download_path_output,
                    demucs_vocals_output, vocals_path_output,
                    demucs_no_vocals_output, no_vocals_path_output
                ])
                
                yt_download_btn.click(
                    fn=download_audio_from_yt,
                    inputs=[separate_mode, youtube_url],
                    outputs=[yt_download_output, yt_download_path_output]
                )
                separate_process_btn.click(
                    fn=download_audio_from_yt,
                    inputs=[separate_mode, youtube_url],
                    outputs=[demucs_vocals_output, demucs_no_vocals_output, vocals_path_output, no_vocals_path_output]
                )
            with gr.Row():
                with gr.Column():
                    model_url = gr.Textbox(label="Save/extract model from url", placeholder="https://huggxample.com/RVC/cute_femboy_voice_model.zip", interactive=True)
                    unzip_mode = gr.Checkbox(value=True, label="unzip")
                    download_model_btn = gr.Button("download", variant="primary")
                downloaded_model_output = gr.TextArea(label="", interactive=False)
                download_model_btn.click(fn=download_model_from_url, inputs=[model_url, unzip_mode], outputs=[downloaded_model_output])

    print("[*] starting gradio ...")
    app.launch(share=args.share)