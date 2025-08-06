# :symbols: TeCla-NLP: Text Classification with Natural Languaje processing

## :question: **Installation**
 - Clone the repository:
```
git clone https://github.com/jitalo333/VRID_language_proyect.git
cd VRID_language_proyect/Tecla-NLP/
```

- Set up your working envirioment:
To run the LLMs experiments you need to have [ollama](https://github.com/ollama/ollama/tree/main) installed in your host machine. We strongly suggest the use of docker to perform your experiments, a docker image is contained in Dockerfile. For a containerized setup, use the provided scripts:
```
bash core-build-container.sh
bash core-run-container.sh
```
Please have in mind that you need to expose an available port when using a remote host. The container must have the link to the running ollama service to use the LLMs.
