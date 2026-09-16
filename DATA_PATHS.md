# Local data configuration

The release is dataset-excluded. Before running `scripts/evaluate_v2.py` or a
training script, provide local equivalents of the following roots in the
script configuration (or adapt the path variables at the top of the script):

| Root | Required contents |
|---|---|
| source package | prepared source-system inputs, metadata, and topology artifacts |
| relation/control root | validated relation caches, control index, and normalization |
| Test root | frozen Test manifest and prepared Test inputs |
| WECC root | frozen WECC manifest and prepared Generalization inputs |
| run root | model checkpoints and run configurations |

The historical scripts retain path defaults from the internal archive so that
the original audit can be traced. Those defaults are not portable and must be
replaced before execution on another machine. No dataset path or test result
is inferred automatically.
