"""
Download, extract, and verify the SoccerNet tracking dataset.

Usage:
    python download_dataset.py --output_dir data/train
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='Download SoccerNet tracking dataset')
    p.add_argument('--output_dir', default='data/train',
                   help='Destination for SNMOT-xxx folders (default: data/train)')
    p.add_argument('--task', default='tracking-2023',
                   help='SoccerNet task name (default: tracking-2023)')
    p.add_argument('--split', default='train',
                   help='Dataset split to download (default: train)')
    p.add_argument('--download_dir', default='data/soccernet_raw',
                   help='Temporary directory for raw downloads (default: data/soccernet_raw)')
    p.add_argument('--skip_download', action='store_true',
                   help='Skip download step (use if zips already exist)')
    p.add_argument('--cleanup', action='store_true',
                   help='Delete download_dir after successful extraction')
    return p.parse_args()


def download(download_dir, task, split):
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        print("ERROR: SoccerNet package not installed.")
        print("Install with:  pip install SoccerNet")
        sys.exit(1)

    Path(download_dir).mkdir(parents=True, exist_ok=True)
    downloader = SoccerNetDownloader(LocalDirectory=download_dir)
    try:
        downloader.downloadDataTask(task=task, split=[split])
    except Exception as e:
        print(f"\nERROR: Download failed: {e}")
        print("\nYou may need to:")
        print("  1. Sign the SoccerNet NDA at https://www.soccer-net.org/data")
        print("  2. Check your email for the password")
        print("  3. Re-run this script and enter the password when prompted")
        sys.exit(1)


def find_zips(download_dir):
    zips = list(Path(download_dir).rglob('*.zip'))
    if not zips:
        print(f"WARNING: No zip files found under '{download_dir}'.")
        print("Check the directory manually or re-run without --skip_download.")
    return zips


def extract_zips(zips, extract_dir):
    Path(extract_dir).mkdir(parents=True, exist_ok=True)
    for zip_path in zips:
        print(f"  Extracting {zip_path.name}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            print(f"  WARNING: {zip_path.name} is not a valid zip, skipping.")


def find_snmot_folders(search_root):
    """Recursively find SNMOT-* folders that contain an img1/ subdirectory."""
    found = []
    for p in Path(search_root).rglob('img1'):
        parent = p.parent
        if parent.name.startswith('SNMOT-'):
            found.append(parent)
    return sorted(set(found))


def move_sequences(snmot_folders, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    moved = 0
    for src in snmot_folders:
        dst = output_path / src.name
        if dst.exists():
            print(f"  SKIP: {src.name} already exists in {output_dir}")
        else:
            shutil.move(str(src), str(dst))
            print(f"  Moved {src.name} -> {dst}")
            moved += 1
    return moved


def verify(output_dir):
    output_path = Path(output_dir)
    sequences = sorted(output_path.glob('SNMOT-*'))

    if not sequences:
        print(f"ERROR: No SNMOT-* folders found in '{output_dir}'.")
        print("Check that the task/split arguments match the downloaded content.")
        return

    required = {
        'img1 (frames)': lambda s: (s / 'img1').is_dir(),
        'gt/gt.txt':     lambda s: (s / 'gt' / 'gt.txt').is_file(),
        'seqinfo.ini':   lambda s: (s / 'seqinfo.ini').is_file(),
        'gameinfo.ini':  lambda s: (s / 'gameinfo.ini').is_file(),
    }

    col_w = max(len(s.name) for s in sequences) + 2
    header = f"{'Sequence':<{col_w}} {'Frames':>7}  " + '  '.join(required.keys())
    print('\n' + header)
    print('-' * len(header))

    total_frames = 0
    missing_any = False

    for seq in sequences:
        n_frames = len(list((seq / 'img1').glob('*.jpg'))) if (seq / 'img1').is_dir() else 0
        total_frames += n_frames
        checks = ['✓' if fn(seq) else '✗' for fn in required.values()]
        if '✗' in checks:
            missing_any = True
        print(f"{seq.name:<{col_w}} {n_frames:>7}  " + '       '.join(checks))

    print()
    print(f"Found {len(sequences)} sequence(s), {total_frames} total frames.")
    if missing_any:
        print("WARNING: Some sequences have missing files (marked ✗ above).")


def main():
    args = parse_args()
    extract_dir = str(Path(args.download_dir) / '_extracted')

    # Step 1: Download
    if args.skip_download:
        print("Skipping download (--skip_download set).")
    else:
        print(f"Downloading task='{args.task}', split='{args.split}' -> {args.download_dir}")
        download(args.download_dir, args.task, args.split)

    # Step 2: Find and extract zips
    print(f"\nSearching for zip files in '{args.download_dir}'...")
    zips = find_zips(args.download_dir)
    if zips:
        print(f"Found {len(zips)} zip file(s). Extracting to '{extract_dir}'...")
        extract_zips(zips, extract_dir)
    else:
        # Downloader may have already extracted — search download_dir directly
        extract_dir = args.download_dir

    # Step 3: Locate SNMOT folders
    print(f"\nLocating SNMOT-* folders in '{extract_dir}'...")
    snmot_folders = find_snmot_folders(extract_dir)
    if not snmot_folders:
        print("ERROR: No SNMOT-* folders found after extraction.")
        print(f"  task='{args.task}', split='{args.split}' may not match downloaded content.")
        print(f"  Browse '{extract_dir}' manually to check the structure.")
        sys.exit(1)
    print(f"Found {len(snmot_folders)} SNMOT folder(s).")

    # Step 4: Move to output_dir
    print(f"\nMoving sequences to '{args.output_dir}'...")
    move_sequences(snmot_folders, args.output_dir)

    # Step 5: Verify
    print(f"\nVerifying '{args.output_dir}'...")
    verify(args.output_dir)

    # Step 6: Optional cleanup
    if args.cleanup:
        print(f"\nCleaning up '{args.download_dir}'...")
        shutil.rmtree(args.download_dir, ignore_errors=True)
        print("Done.")
    else:
        answer = input(f"\nDelete raw download directory '{args.download_dir}' to save space? [y/N] ")
        if answer.strip().lower() == 'y':
            shutil.rmtree(args.download_dir, ignore_errors=True)
            print("Deleted.")


if __name__ == '__main__':
    main()
