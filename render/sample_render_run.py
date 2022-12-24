import subprocess
import sys
import os

sys.path.append(r'.')
sys.path.append(r'..')

from tools.config import cfg

process = subprocess.Popen([cfg.blender_proc_path, 'run', os.path.join(cfg.ROOT_DIR, '../render', 'sample_render.py')],
                           shell=False)
process.wait()
