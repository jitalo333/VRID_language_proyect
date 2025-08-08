# :symbols: TeCla-NLP: Text Classification with Natural Languaje processing

## :question: **Installation**
 - Clone the repository:
```
git clone https://github.com/jitalo333/VRID_language_proyect.git
cd VRID_language_proyect/Tecla-NLP/
```

- Set up your working envirioment:
To run the LLMs experiments you need to have [ollama](https://github.com/ollama/ollama/tree/main) installed in your host machine and the desired LLM available. We strongly suggest the use of docker to perform your experiments, a docker image is contained in Dockerfile. For a containerized setup, use the provided scripts:
```
bash core-build-container.sh
```
Please have in mind that you need to expose an available port when using a remote host. The container must have the link to the running ollama service to use the LLMs.

---
## :ballot_box_with_check: **How to run the Jupyter Notebooks**

First (asumming you have your desired LLM pulled in your host machine) initialize ollama:
```
ollama serve 
```

The run the docker container, you can use the suggested instrucction provided in the repository as:
```
bash core-run-container.sh
```

The file will share the host network. After running the container from the terminal use the following instruction:
```
jupyter notebook --ip 0.0.0.0 --port 8883 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
```

And have fun playing with the tutorial notebooks :sunglasses:.
