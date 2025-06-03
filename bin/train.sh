# =================================================
# This setup assumes 1 A100 GPU with 80GB of memory
# and we want to achieve 128 effective batch size
# also assumes a machine with 8 CPUs
# =================================================


# Debugging flags (optional)
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"


run() {
    uv run primer/train.py --config-path ../experiments --config-name exp_default "$@"
}


# ==== Minipile with 57M-tied and all tokenizers
# Need to run separately for bpe128000minipile since we need to change batch size to fit
run -m model=me57M-tied \
    tok_name=bpe8064minipile,bpe32000minipile,bpe2wp32000minipile \

run model=me57M-tied \
    tok_name=bpe128000minipile \
    optim.grad_acc_schedule='{0: 16}' \
    data.batch_size=8 \
    data.eval_batch_size=16 -m

# for some reasons, I changed the batch size for this tokeniser, even though it should have fit!
run model=me57M-tied \
    tok_name=wordpiece32000minipile \
    data.batch_size=16 \
    data.eval_batch_size=64


# ==== Minipile with bpe32000minipile on all other model sizes
run model=me340M-tied \
    tok_name=bpe32000minipile \
    optim.grad_acc_schedule='{0: 2}' \
    data.batch_size=16 \
    data.eval_batch_size=128

run model=me850M-tied \
    tok_name=bpe32000minipile \
    optim.grad_acc_schedule='{0: 2}' \
    data.batch_size=16 \
    data.eval_batch_size=64


# ==== Fineweb-Edu with bpe32000minipile on me100M-tied and me100M
# For some reason, I changed the optimisation hparams (though, this make more sense to me)
run -m dataset=finewebedu-20B \
    model=me100M-tied,me100M \
    tok_name=bpe32000minipile \
    optim.weight_decay=0.01 \
    optim.scheduler_kwargs.num_decay_steps: 4000 \
    optim.scheduler_kwargs.num_stable_steps: 44000