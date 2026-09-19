"""Publish a fixed body-only knowledge bundle without starting the backend."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def download_model(directory):
    """Explicit public download; never used by the service or recovery path."""
    from huggingface_hub import HfApi, snapshot_download
    from backend.app.knowledge_retrieval import MODEL_ID, MODEL_REVISION, sha256
    info = HfApi(token=False).model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True, timeout=20)
    if info.sha != MODEL_REVISION:
        raise ValueError('model revision mismatch')
    names = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
             'special_tokens_map.json', 'vocab.txt']
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=directory,
                      allow_patterns=names, token=False, max_workers=2)
    hashes = {name: sha256((directory/name).read_bytes()) for name in names}
    weights = next(f for f in info.siblings if f.rfilename == 'model.safetensors')
    if weights.lfs is None or hashes['model.safetensors'] != weights.lfs.sha256:
        raise ValueError('official weight digest mismatch')
    (directory/'model_integrity.json').write_text(json.dumps(dict(revision=MODEL_REVISION, files=hashes),
                                                            sort_keys=True), encoding='utf-8')


def main():
    from backend.app.knowledge_retrieval import BodyEncoder, CardLibrary, current_bundle, publish
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cards', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download-model', action='store_true', help='Explicitly fetch fixed public weights first')
    args = parser.parse_args()
    library = CardLibrary.model_validate_json(args.cards.read_bytes())
    if args.download_model:
        download_model(args.model)
    encoder = BodyEncoder(args.model)
    previous = current_bundle(args.output) if (args.output / 'current.json').exists() else None
    bundle = publish(library, args.output, encoder, previous=previous)
    print(f'Published {len(bundle.cards.entries)} cards; manifest={bundle.manifest_digest}')


if __name__ == '__main__':
    main()
