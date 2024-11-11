

### Fine-Tuning ###
```
cd LLMs
python3 finetune.py
```


### Inference ###

Only LLM inference:
```
cd LLMs
python3 finetuned_llama.py
```

Note you can change ```configs/inference.yml``` for different settings.

Download the [pretrained](https://drive.google.com/drive/folders/1SftVU4kSOVy7OP2FLdV7Eg5gN0gFg9y5?usp=sharing) model and place it at ```pretrained/``` as it is.

### Data Preparation for Fine-Tuning ###

```
cd utils
python3 save_oneliner.py
```

The code will convert the normal prompts into ones for LLaMa, saved in ```data/dump``` as a backup (not for a real usage). It also push the data to your own huggingface repository, which our code actually uses.

### End-to-End Demo ###

```
python3 demo_efficient.py
```
