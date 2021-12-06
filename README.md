# emergent-language
An implementation of Emergence of Grounded Compositional Language in Multi-Agent Populations by Igor Mordatch and Pieter Abbeel

To run, invoke `python3 train.py` in environment with PyTorch installed. To experiment with parameters, invoke `python3 train.py --help` to get a list of command line arguments that modify parameters. Currently training just prints out the loss of each game episode run, without any further analysis, and the model weights are not saved at the end. These features are coming soon.

* `game.py` provides a non-tensor based implementation of the game mechanics (used for game behavior exploration and random game generation during training
* `model.py` provides the full computational model including agent and game dynamics through an entire episode
* `train.py` provides the training harness that runs many games and trains the agents
* `configs.py` provides the data structures that are passed as configuration to various modules in the computational graph as well as the default values used in training now
* `constants.py` provides constant factors that shouldn't need modification during regular running of the model
* `visualize.py` provides a computational graph visualization tool taken from [here](https://github.com/szagoruyko/functional-zoo/blob/master/visualize.py)
* `simple_model.py` provides a simple model that doesn't communicate and only moves based on its own goal (used for testing other components)
* `comp-graph.pdf` is a pdf visualization of the computational graph of the game-agent mechanics

## Todos
- [ ] Separate NLP code from article's implementation
- [ ] What is tensorboard?
- [ ] review derivative implementation
- [ ] clean, separate front and backends
- [ ] generate plots and gifs