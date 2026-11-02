CPU: python3 cpu_check.py
DCU: python3 gpu_quickstart.py --steps 12
Resume: python3 gpu_quickstart.py --steps 16 --resume checkpoint.pt
Inference: python3 gpu_quickstart.py --infer --resume checkpoint.pt
Live config: python3 gpu_quickstart.py --steps 60 --delay 1
Requires a certified exclusive DCU instance with preinstalled PyTorch.
Do not treat CPU execution as DCU validation.
