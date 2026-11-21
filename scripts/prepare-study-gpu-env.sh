#!/usr/bin/env bash
# Prepare dependencies on a networked host; never query or initialize a GPU.
set -euo pipefail
ftf_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ftf_repo_dir=$(cd -- "$ftf_script_dir/.." && pwd)
ftf_python=/usr/bin/python3.11
ftf_offline=0
ftf_environment=
ftf_uv=
ftf_cache=
ftf_receipt=
ftf_checkpoint=
while (($#)); do
  case "$1" in
    --environment) ftf_environment=$2; shift 2 ;;
    --uv) ftf_uv=$2; shift 2 ;;
    --python) ftf_python=$2; shift 2 ;;
    --cache) ftf_cache=$2; shift 2 ;;
    --receipt) ftf_receipt=$2; shift 2 ;;
    --checkpoint) ftf_checkpoint=$2; shift 2 ;;
    --offline) ftf_offline=1; shift ;;
    *) printf 'Unknown or incomplete option: %s\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ -z "$ftf_environment" || -z "$ftf_uv" || -z "$ftf_cache" || -z "$ftf_receipt" ]]; then
  printf '%s\n' 'Required: --environment DIR --uv EXECUTABLE --cache DIR --receipt FILE [--python EXECUTABLE] [--checkpoint FILE] [--offline]' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export UV_CACHE_DIR="$ftf_cache"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
ftf_lock="$ftf_repo_dir/environments/study-training-gpu.lock"
ftf_marker="$ftf_environment/.finetuneforge-gpu-environment.json"
"$ftf_python" - "$ftf_environment" "$ftf_marker" <<'PY'
import json,sys
from pathlib import Path
if sys.version_info[:2] != (3,11): raise SystemExit('Python3.11 required')
root=Path(sys.argv[1]); marker=Path(sys.argv[2])
if root.is_symlink(): raise SystemExit('Environment symlinks are not accepted')
if root.exists():
    try: value=json.loads(marker.read_text())
    except (OSError,ValueError): raise SystemExit('Refusing to mutate an existing environment without the FineTuneForge GPU ownership marker')
    if value != {'schema':'finetuneforge.gpu_environment_owner.v1','scope':'cuda_qualification_candidate'}:
        raise SystemExit('GPU environment ownership marker mismatch')
PY
ftf_uv_flags=()
if ((ftf_offline)); then ftf_uv_flags+=(--offline); fi
if [[ ! -e "$ftf_environment" ]]; then
  "$ftf_uv" venv --python "$ftf_python" "${ftf_uv_flags[@]}" "$ftf_environment"
  "$ftf_python" - "$ftf_marker" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'schema':'finetuneforge.gpu_environment_owner.v1','scope':'cuda_qualification_candidate'},sort_keys=True)+'\n')
PY
fi
"$ftf_uv" pip sync --python "$ftf_environment/bin/python" --require-hashes "${ftf_uv_flags[@]}" "$ftf_lock"
"$ftf_uv" pip check --python "$ftf_environment/bin/python"
"$ftf_environment/bin/python" - "$ftf_repo_dir" "$ftf_receipt" "$ftf_checkpoint" "$ftf_offline" <<'PY'
from importlib.metadata import distributions,version
from pathlib import Path
import hashlib,json,platform,re,sys
repo,receipt,checkpoint,offline=map(str,sys.argv[1:])
repo=Path(repo);receipt=Path(receipt)
# Capture installed distributions before imports expose setuptools' vendored metadata.
installed=dict(sorted((d.metadata['Name'],d.version) for d in distributions()))
locked=dict(re.findall(r'^([A-Za-z0-9_.-]+)==([^\s;]+)',(repo/'environments/study-training-gpu.lock').read_text(),re.M))
canonical=lambda name:re.sub(r'[-_.]+','-',name).lower()
if {canonical(k):v for k,v in installed.items()}!={canonical(k):v for k,v in locked.items()}:
    raise RuntimeError('Installed distribution inventory does not match the complete lock')
import torch,tabpfn,peft,transformers
if sys.version_info[:2]!=(3,11): raise RuntimeError('Python3.11 required')
if torch.__version__!='2.6.0+cu124' or torch.version.cuda!='12.4': raise RuntimeError('Pinned Torch2.6.0 CUDA12.4 build required')
if torch.cuda.is_initialized(): raise RuntimeError('Preparation must not initialize CUDA')
expected={'tabpfn':'2.0.9','peft':'0.21.3','transformers':'4.57.6','numpy':'1.26.4','scipy':'1.12.0','sbi':'0.23.0'}
if any(version(k)!=v for k,v in expected.items()): raise RuntimeError('Direct dependency ancestry mismatch')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checkpoint_info=None
if checkpoint:
    checksum=sha(checkpoint)
    if checksum!='2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736': raise RuntimeError('Pinned checkpoint checksum mismatch')
    checkpoint_info={'path':str(Path(checkpoint).resolve()),'sha256':checksum}
value={'schema':'finetuneforge.gpu_environment_preparation.v1','environment':str(Path(sys.executable).parent.parent),'python':sys.version,'platform':platform.platform(),'torch_build':torch.__version__,'cuda_runtime':torch.version.cuda,'cuda_initialized':torch.cuda.is_initialized(),'gpu_hardware_checked':False,'gpu_jobs_submitted':0,'scientifically_qualified':False,'offline_preparation':offline=='1','imports':['torch','tabpfn','peft','transformers'],'checkpoint':checkpoint_info,'source_files':{name:sha(repo/name) for name in ['environments/study-training-gpu.in','environments/study-training-gpu.lock','scripts/prepare-study-gpu-env.sh']},'packages':installed,'package_inventory_policy':'top_level_distributions_before_setuptools_vendor_imports'}
receipt.parent.mkdir(parents=True,exist_ok=True)
receipt.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
print(json.dumps({'receipt':str(receipt),'lock_sha256':value['source_files']['environments/study-training-gpu.lock'],'torch_build':torch.__version__,'cuda_initialized':False,'gpu_hardware_checked':False,'packages':len(value['packages'])},sort_keys=True))
PY
