import librosa
import os
import audio_utils as au
import numpy as np
import json

import warnings

DURATION = 29 # sec
DATA_PATH = "Data/genres_original"
JSON_PATH = "predata.json"

# DATA_PATH = "Data/test_files"
# JSON_PATH = "predata_small.json"

# x = np.random.uniform(low=-0.2, high=0.2, size=(32000,)).astype(np.float32)
def get_mfcc_size(sr=22050, n_mfcc=13, num_segments=5):
  '''
  Get mfcc coefficient matrix dimensions (m-by-n). One dimension is frame count
  and the other is n_mfcc.
  '''
  duration_pre_segment = DURATION / num_segments
  num_samples_per_segment = int(duration_pre_segment * sr)
  x = np.random.uniform(low=-0.2, high=0.2, size=(num_samples_per_segment,)).astype(np.float32)
  mfcc = librosa.feature.mfcc(x,
                  sr=sr,
                  n_mfcc=n_mfcc
                )
  mfcc = mfcc.T

  return mfcc.shape

def load_mfcc_jason(json_path=JSON_PATH):
  with open(json_path, mode="r") as f:
    data = json.load(f)
    return data

def save_mfcc_json(data_path=DATA_PATH, json_path=JSON_PATH, n_mfcc=13, num_segments=5):
  '''
  Navigate the dataset and transfer the sound clips into mfcc matrices.
  '''
  # data dictionary
  data = {
    "mapping":[], # genre in str
    "mfcc":[], # mfcc coeffs in array
    "labels":[] # label in int
  }

  duration_pre_segment = DURATION / num_segments

  for i, (dirpath, dirnames, filenames) in enumerate(os.walk(data_path)):
    # ensure we are not at root dir
    if dirpath is data_path:
      continue

    # save the semantic label
    dirpath_components = os.path.split(dirpath) # genre/blues -> ["genre","blues"]
    semantic_label = dirpath_components[-1]
    data["mapping"].append(semantic_label)

    # for each genre, process all files
    for fn in filenames:
      file_path = os.path.join(dirpath, fn)
      signal, sr = au.load_sound_file(file_path, sample_rate=None)
      data["sr"] = sr

      num_samples_per_segment = int(duration_pre_segment * sr)
      # process segments and extract mfcc
      for s in range(num_segments):
        start_sample = num_samples_per_segment * s
        end_sample = start_sample + num_samples_per_segment
        mfcc = librosa.feature.mfcc(signal[start_sample:end_sample],
                                    sr=sr,
                                    n_mfcc=n_mfcc
                                    )
        mfcc = mfcc.T
        data["mfcc"].append(mfcc.tolist())
        data["labels"].append(i-1) # i=0 is root dir
        # print("{}, segment: {}, mfcc: {}".format(fn, s, mfcc.shape))
        # if mfcc.shape[0] < 242:
        #   warnings.warn("mfcc time frames is under expectation! ", mfcc.shape[0])
        

  with open(json_path, 'w') as fp:
    json.dump(data, fp, indent=2)

  return data, sr

if __name__ == "__main__":
  save_mfcc_json()