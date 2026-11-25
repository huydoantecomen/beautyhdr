"""Chạy một bản mã (PYTHONPATH) trên bộ ảnh set/, ghi ảnh ra + JSON (preset, cài đặt, model, cỡ xuất, thiết bị, chế độ)."""
import json, sys, time, os, platform
from pathlib import Path
import torch
torch.set_num_threads(4)
from net_anh import __version__
from net_anh.pipeline import Enhancer, load_image, save_image
from net_anh.analysis import recommend
from net_anh.settings import Settings, apply_preset, PRESETS, MODES

src, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
only = sys.argv[3].split(',') if len(sys.argv) > 3 and sys.argv[3] else None
def presets_for(n):
    if n.startswith(('cu_', 'phai_', 'amvang_', 'amam_')): return ['restore', 'auto']
    if n.startswith(('xuoc_', 'sach_')): return ['restore']
    return ['auto', 'max']
eng = Enhancer('cpu')
for f in sorted(src.glob('*.png')):
    if f.stem.endswith('_gt') or (only and f.stem not in only): continue
    img = load_image(f)
    for p in presets_for(f.stem):
        dst = out / f'{f.stem}__{p}.png'
        if dst.exists(): continue
        s = Settings()
        why = ''
        if p == 'auto':
            an = eng.analyze(img, with_faces=True)
            chosen, s, w = recommend(an, s); why = f'Tự động → {PRESETS[chosen][0]}: ' + '; '.join(w)
        else:
            s = apply_preset(s, p)
        t = time.time()
        res, rep = eng.process(img, s)
        save_image(res, dst)
        meta = dict(version=__version__, image=f.name, preset=p, why=why, mode=MODES.get(s.mode, s.mode),
                    in_size=list(img.shape[1::-1]), out_size=list(res.shape[1::-1]), device=rep.device,
                    host=f'{platform.processor() or platform.machine()} {torch.get_num_threads()} luồng, torch {torch.__version__}',
                    models=rep.models(), faces_found=rep.faces_found, faces_restored=rep.faces_restored,
                    subjects=rep.subjects, notes=rep.notes, seconds=round(time.time() - t, 1),
                    settings={k: v for k, v in s.to_dict().items() if k != 'face_styles'})
        (out / f'{f.stem}__{p}.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        print(f'{f.stem} {p} {meta["seconds"]}s {rep.models()} faces {rep.faces_restored}/{rep.faces_found}', flush=True)
