#!/usr/bin/env python3
"""Generate synthetic binary crack masks from a manifest or custom widths."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import json
import cv2
import synthetic_geometry as geometry

DEFAULT_MANIFEST=Path(__file__).resolve().parent/'configs'/'balanced160.csv'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST,
                        help='CSV with image_id, level, target_width_px, seed')
    parser.add_argument('--widths',type=int,nargs='+',default=None,
                        help='Custom nominal widths; overrides the preset manifest')
    parser.add_argument('--samples-per-width',type=int,default=10)
    parser.add_argument('--seed',type=int,default=42,help='Base seed for custom widths')
    parser.add_argument('--height',type=int,default=600)
    parser.add_argument('--width',type=int,default=800)
    parser.add_argument('--mm-per-pixel',type=float,default=.01278)
    parser.add_argument('--save-overlays',action='store_true')
    args=parser.parse_args()
    if args.height<=0 or args.width<=0 or args.mm_per_pixel<=0:parser.error('Image dimensions and scale must be positive.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):parser.error('Output directory must be empty. Choose a new directory.')
    if args.widths is None:
        with args.manifest.open(encoding='utf-8-sig',newline='') as f:settings=list(csv.DictReader(f))
    else:
        if args.samples_per_width<1 or any(w<1 for w in args.widths):parser.error('Custom widths and sample count must be positive.')
        settings=[]
        for w0 in args.widths:
            for i in range(args.samples_per_width):
                settings.append({'image_id':f'custom_w{w0:02d}_{i:03d}.png','level':'custom','target_width_px':w0,'seed':args.seed+137*len(settings)+7})
    if not settings:parser.error('The manifest is empty.')
    ids=[str(r['image_id']) for r in settings]
    if len(ids)!=len(set(ids)) or any(Path(i).name!=i or Path(i).suffix.lower()!='.png' for i in ids):parser.error('Image identifiers must be unique PNG basenames.')
    image_dir=args.output_dir/'images';image_dir.mkdir(parents=True,exist_ok=True)
    if args.save_overlays:(args.output_dir/'overlays').mkdir()
    geometry.MM_PER_PIXEL=args.mm_per_pixel
    rows=[]
    for setting in settings:
        w0=int(setting['target_width_px']);seed=int(setting['seed'])
        if w0<=0:parser.error('Nominal widths must be positive.')
        mask,ref,cx,cy,path,_=geometry.generate_single_crack(args.height,args.width,w0,seed=seed)
        image_id=setting['image_id']
        if not cv2.imwrite(str(image_dir/image_id),mask):raise OSError(f'Cannot write {image_id}')
        if args.save_overlays:
            overlay=geometry.create_overlay(mask,path,cx,cy,ref)
            cv2.imwrite(str(args.output_dir/'overlays'/image_id),overlay)
        rows.append({'image_id':image_id,'level':setting['level'],'target_width_px':w0,
                     'gt_max_width_px':round(ref,4),'gt_max_width_mm':round(ref*args.mm_per_pixel,6),
                     'gt_center_x':round(cx,2),'gt_center_y':round(cy,2)})
    with (args.output_dir/'groundtruth.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    run={'samples':len(rows),'nominal_group_counts':dict(Counter(r['level'] for r in rows)),
         'image_height':args.height,'image_width':args.width,'mm_per_pixel':args.mm_per_pixel,
         'reference_definition':'Maximum pre-perturbation design width over samples 100 through 399.',
         'reference_min_mm':min(r['gt_max_width_mm'] for r in rows),'reference_max_mm':max(r['gt_max_width_mm'] for r in rows)}
    with (args.output_dir/'run_metadata.json').open('w',encoding='utf-8') as f:json.dump(run,f,indent=2)
    with (args.output_dir/'sample_settings.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['image_id','level','target_width_px','seed']);writer.writeheader();writer.writerows(settings)
    print(json.dumps(run,indent=2))

if __name__=='__main__':main()
