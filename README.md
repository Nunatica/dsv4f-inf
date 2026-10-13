# dsv4f-inf
Almost fully vibe-coded, highly experimental serving engine for DeepSeek V4 Flash.

## Target system
9900x (or any zen5 cpus such as 9800x3d) + 192GiB ddr5 5600mhz + 7900xtx (or nvidia gpus such 3090, 4090 which has at least 24gb vram, not tested)

## Target model
- main [https://huggingface.co/unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF/tree/main/UD-Q4_K_XL]
- mtp [https://huggingface.co/ddh0/DeepSeek-V4-Flash-GGUF/tree/a79b345f46fad371767f6a57cc3be31f28e41205]

## How to run
Put ggufs into models and ensure model path is correct.

```bash
./build-engine.sh
./run-server.sh
```

## Performance
At target system, it reached to 230 tok/s for prefill and 18.93 tok/s for decode.
For comparison, llama cpp gives you 197 tok/s for prefill and 12.43 tok/s for decode with moe expert cache + mtp.
Performance test is done with same prompt from PROMPT_TEST.md

## Credits
- llama.cpp [https://github.com/ggml-org/llama.cpp]
- Strata [https://github.com/Niko1221/Strata]
- project-maya [https://github.com/mw00/project-maya]

## License
Project itself is under MIT License. included llama.cpp source follows its own license.

## Disclaimer
This project is not intended to use for serious tasks. Use it at your own risk.
