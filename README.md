### Fine-Tuning ###
```
cd LLMs
python3 qlora.py
```


### Inference ###

Only LLM inference:
```
cd LLMs
python3 quantized_llama.py
```

Note ```finetuned_llama.py```is not available yet since fine-tuning keeps failing.

### Data Preparation for Fine-Tuning ###

```
cd LLMs
python3 save.py
```

The code will convert the normal prompts into ones for LLaMa, saved in ```./~.json```.

### End-to-End Demo ###

```
python3 demo_efficient.py
```

### Problems ###
1) Although ```data.json``` for prompt - json is well created, preprocessing(especially tokenizing) before feeding it to the model is tricky.

Getting accustomed to fine-tuning LLaMa might take some time.