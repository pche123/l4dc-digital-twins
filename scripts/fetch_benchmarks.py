"""Optional downloader for publisher-linked archives. Never assumes version equivalence.
Requires pip install -r requirements-download-optional.txt.
"""
from pathlib import Path
import argparse,zipfile,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1]
SOURCES={
 'silverbox':('17iS-6oBUUgrmiAcrZoG9S5sOaljZnDSy',['SNLS80mV.csv']),
 'emps':('1zwoXYa9-3f8NQ0ohzmjpF7UxbNgRTHkS',['DATA_EMPS.mat','DATA_EMPS_PULSES.mat']),
 'wiener-hammerstein':('16ipySVfKfxkwqWmbO9Z19-VjDoC2S6hx',['WienerHammerBenchmark.csv'])}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=list(SOURCES)+['all'],default='all');p.add_argument('--download',action='store_true',help='Actually contact the public Google Drive downloads; otherwise only print source links')
 a=p.parse_args(); target=ROOT/'external_data';target.mkdir(exist_ok=True)
 for name in (SOURCES if a.dataset=='all' else [a.dataset]):
  did,need=SOURCES[name]; print(name,'https://drive.google.com/file/d/'+did+'/view','expected:',need)
  if not a.download:continue
  import gdown
  z=target/(name+'.zip');gdown.download(id=did,output=str(z),quiet=False)
  if not z.exists() or not zipfile.is_zipfile(z):raise RuntimeError(f'Not a valid ZIP: {z}; visit the official source page')
  extraction=target/('_'+name);extraction.mkdir(exist_ok=True)
  with zipfile.ZipFile(z) as f:
   for item in f.infolist():
    if item.is_dir():continue
    if Path(item.filename).is_absolute() or '..' in Path(item.filename).parts:raise ValueError('Unsafe archive path')
   f.extractall(extraction)
  for basename in need:
   candidates=list(extraction.rglob(basename))
   if len(candidates)!=1: print('MANUAL CHECK REQUIRED: expected 1 match for',basename,'found',len(candidates));continue
   dst=target/basename;shutil.copy2(candidates[0],dst)
   print('copied',dst.name,'sha256',hashlib.sha256(dst.read_bytes()).hexdigest(),'VERIFY CONTENT/SPLIT before interpreting results')
 print('Cascaded tanks: manual retrieval required at https://doi.org/10.4121/12960104')
if __name__=='__main__':main()
