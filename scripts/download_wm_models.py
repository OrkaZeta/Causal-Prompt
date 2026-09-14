"""Download inference weights to explicit cluster paths, without loading models."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['WM-LongLive','WM-AlayaWorld'], required=True)
    p.add_argument('--root', type=Path, default=Path('/projects/hi-paris/ZiyiData/Models'))
    args=p.parse_args()
    repos = [('Efficient-Large-Model/LongLive-2.0-5B','LongLive-2.0-5B',None)] if args.model=='WM-LongLive' else [
        ('AlayaLab/AlayaWorld-v1.1-stage3','AlayaWorld-v1.1-stage3',None),
        ('AlayaLab/AlayaWorld-v1.1-stage2b','AlayaWorld-v1.1-stage2b',None),
        ('Lightricks/LTX-2.3','LTX-2.3',['ltx-2.3-22b-dev.safetensors','README.md','LICENSE']),
        ('google/gemma-3-12b-it-qat-q4_0-unquantized','gemma-3-12b-it-qat-q4_0-unquantized',None),
        ('pkqbajng/ViGeo1.1','ViGeo1.1',None)]
    api=HfApi()
    for repo,directory,patterns in repos:
        dest=args.root/directory
        revision=api.model_info(repo).sha
        print(f'Downloading {repo}@{revision} -> {dest}',flush=True)
        snapshot_download(repo_id=repo,revision=revision,local_dir=str(dest),allow_patterns=patterns,max_workers=4)
        (dest/'download_manifest.json').write_text(json.dumps({'repo_id':repo,'revision':revision,'allow_patterns':patterns},indent=2)+'\n')

if __name__=='__main__':main()
