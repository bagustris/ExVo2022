import os
import audinterface
import audonnx
import torchaudio.transforms as T


wav_path = '/data/22_ICML-ExVo22_TeamAtmaja_ITSN/wav/'
files = os.listdir(wav_path)

model_root = "/data/models/w2v2-L-robust/"
model = audonnx.load(model_root)
save_dir = "/data/22_ICML-ExVo22_TeamAtmaja_ITSN/feats/w2v2-R-emo/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

hidden_states = audinterface.Feature(
    model.outputs['hidden_states'].labels,
    process_func=model,
    process_func_args={
        'output_names': 'hidden_states',
    },
    sampling_rate=16000,    
    resample=True,
    num_workers=5,
    verbose=True,
)

for file in files:
    print(f"Processing {file}")
    feat = hidden_states.process_file(
        os.path.join(wav_path, file)
    )
    feat.index = feat.index.get_level_values(0)
    feat.to_csv(f"{save_dir}/{str(file[:-3])}csv")
_