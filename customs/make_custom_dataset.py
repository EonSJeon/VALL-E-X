import h5py
import glob
import torch
import numpy as np
import os
import torchaudio
import soundfile as sf
from utils.g2p.symbols import symbols
from utils.g2p import PhonemeBpeTokenizer
from utils.prompt_making import make_prompt, make_transcript
from data.collation import get_text_token_collater
from data.dataset import create_dataloader
import pandas as pd

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
from data.tokenizer import (
    AudioTokenizer,
    tokenize_audio,
)

tokenizer_path = "./utils/g2p/bpe_69.json"
tokenizer = PhonemeBpeTokenizer(tokenizer_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_prompts(name, audio_prompt_path, transcript=None):
    text_tokenizer = PhonemeBpeTokenizer(tokenizer_path="./utils/g2p/bpe_69.json")
    text_collater = get_text_token_collater()
    codec = AudioTokenizer(device)
    wav_pr, sr = torchaudio.load(audio_prompt_path)
    # check length
    if wav_pr.size(-1) / sr > 15:
        raise ValueError(f"Prompt too long, expect length below 15 seconds, got {wav_pr / sr} seconds.")
    if wav_pr.size(0) == 2:
        wav_pr = wav_pr.mean(0, keepdim=True)
    text_pr, lang_pr = make_transcript(name, wav_pr, sr, transcript)

    # tokenize audio
    encoded_frames = tokenize_audio(codec, (wav_pr, sr))
    audio_tokens = encoded_frames[0][0].transpose(2, 1).cpu().numpy()

    # tokenize text
    phonemes, langs = text_tokenizer.tokenize(text=f"{text_pr}".strip())
    text_tokens, enroll_x_lens = text_collater(
        [
            phonemes
        ]
    )

    return audio_tokens, text_tokens, langs, text_pr
    
def create_dataset(data_dir, dataloader_process_only):
    if dataloader_process_only:
        h5_output_path=f"{data_dir}/audio_sum.hdf5"
        ann_output_path=f"{data_dir}/audio_ann_sum.txt"
        #audio_folder = os.path.join(data_dir, 'audio')

        audio_paths = glob.glob(f"{data_dir}/*.wav")  # Change this to match your audio file extension
        transcript_df = get_transcript_csv(data_dir)

        # Create or open an HDF5 file
        with h5py.File(h5_output_path, 'w') as h5_file:
            # Loop through each audio and text file, assuming they have the same stem
            for audio_path in audio_paths:
                stem = os.path.splitext(os.path.basename(audio_path))[0]
                print(stem)
                transcript = transcript_df[transcript_df['audio_path']==stem]['sentence'].values[0]
                print(transcript)
                audio_tokens, text_tokens, langs, text = make_prompts(name=stem, audio_prompt_path=audio_path, transcript=transcript)
                
                text_tokens = text_tokens.squeeze(0)
                # Create a group for each stem
                grp = h5_file.create_group(stem)
                # Add audio and text tokens as datasets to the group
                grp.create_dataset('audio', data=audio_tokens)
                #grp.create_dataset('text', data=text_tokens)
                
                with open(ann_output_path, 'a', encoding='utf-8') as ann_file:
                    try:
                        audio, sample_rate = sf.read(audio_path)
                        duration = len(audio) / sample_rate
                        ann_file.write(f'{stem}|{duration}|{langs[0]}|{text}\n')  # 改行を追加
                        print(f"Successfully wrote to {ann_output_path}")
                    except Exception as e:
                        print(f"An error occurred: {e}")
    else:
        dataloader = create_dataloader(data_dir=data_dir)
        return dataloader

def get_transcript_csv(data_dir):
    transcript_paths = glob.glob(f"{data_dir}/*.csv")
    if len(transcript_paths) != 1:
        raise ValueError(f"Only one transcript file is allowed in {data_dir}")
    transcript_path = transcript_paths[0]
    transcript_df = pd.read_csv(transcript_path)
    transcript_df = transcript_df[transcript_df['status'] == 'processed']
    # audio_path 컬럼에서 경로를 분리하고, 확장자를 제거하여 stem만 남김
    transcript_df['audio_path'] = transcript_df['audio_path'].apply(
        lambda x: os.path.splitext(x.split('/')[-1])[0]
    )
    transcript_df = transcript_df[['sentence', 'audio_path']]
    return transcript_df
