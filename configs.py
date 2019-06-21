from typing import NamedTuple, Any

import os
from pathlib import Path
import time
import constants
from modules import data

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_EPOCHS = 1000
DEFAULT_LR = 5e-4
SAVE_MODEL = True
DEFAULT_MODEL_FILE = 'modules_weights.pt'

DEFAULT_HIDDEN_SIZE = 256
DEFAULT_DROPOUT = 0.1
DEFAULT_FEAT_VEC_SIZE = 256
DEFAULT_TIME_HORIZON = 16

USE_UTTERANCES = False
PENALIZE_WORDS = True
DEFAULT_VOCAB_SIZE = 10
DEFAULT_OOV_PROB = 1
DEFAULT_DF_UTTERANCE_COL_NAME = ['agent_color', 'agent_shape', 'lm_color', 'lm_shape', 'sentence']

DEFAULT_WORLD_DIM = 16
MAX_AGENTS = 2 #TODO: add to readme
MAX_LANDMARKS = 3
MIN_AGENTS = 2
MIN_LANDMARKS = 3
NUM_COLORS = 3
NUM_SHAPES = 2

DEFAULT_UPLOAD_TRAINED_MODEL = False
DEFAULT_DIR_UPLOAD_MODEL = ""
DEFAULT_SAVE_TO_A_NEW_DIR = False
DEFAULT_CREATING_DATASET_MODE = False
DEFAULT_FOLDER_DIR = str(Path(os.getcwd())) + os.sep + 'debag' + os.sep
DEFAULT_CORPUS = None
DEFAULT_USE_OLD_UTTERANCE_CODE = False


DEFAULT_INIT_RANGE = 0.1
DEFAULT_NHID_LANG = 256
DEFAULT_NEMBED_WORDS = 256
DEFAULT_NHID_CTX = 256
DEFAULT_DROPOUT = 0
DEFAULT_MOMENTUM = 0.1
DEFAULT_LR = 0.0001
DEFAULT_NESTEROV = False
DEFAULT_CLIP = 0.5
DEFAULT_TEMPERATURE = 0.5

UtteranceConfig = NamedTuple('UtteranceConfig', [
    ('init_range', float),
    ('nhid_lang', int),
    ('nembed_word', int),
    ('nhid_ctx', int),
    ('dropout', float),
    ('momentum', float),
    ('lr', int),
    ('nesterov', bool),
    ('clip', float),
    ('batch_size', int),
    ('temperature', float),
])


TrainingConfig = NamedTuple('TrainingConfig', [
    ('num_epochs', int),
    ('learning_rate', float),
    ('load_model', bool),
    ('load_model_file', str),
    ('save_model', bool),
    ('save_model_file', str),
    ('use_cuda', bool),
    ('no_utterances', bool),
])

GameConfig = NamedTuple('GameConfig', [
    ('batch_size', int),
    ('world_dim', Any),
    ('max_agents', int),
    ('max_landmarks', int),
    ('min_agents', int),
    ('min_landmarks', int),
    ('num_shapes', int),
    ('num_colors', int),
    ('use_utterances', bool),
    ('vocab_size', int),
    ('memory_size', int),
    ('use_cuda', bool),
    ('time_horizon', int),
    ('num_epochs', int)
])

ProcessingModuleConfig = NamedTuple('ProcessingModuleConfig', [
    ('input_size', int),
    ('hidden_size', int),
    ('dropout', float)
    ])

WordCountingModuleConfig = NamedTuple('WordCountingModuleConfig', [
    ('vocab_size', int),
    ('oov_prob', float),
    ('use_cuda', bool)
    ])

GoalPredictingProcessingModuleConfig = NamedTuple("GoalPredictingProcessingModuleConfig", [
    ('processor', ProcessingModuleConfig),
    ('hidden_size', int),
    ('dropout', float),
    ('goal_size', int)
    ])

ActionModuleConfig = NamedTuple("ActionModuleConfig", [
    ('goal_processor', ProcessingModuleConfig),
    ('action_processor', ProcessingModuleConfig),
    ('hidden_size', int),
    ('dropout', float),
    ('movement_dim_size', int),
    ('movement_step_size', int),
    ('vocab_size', int),
    ('use_utterances', bool),
    ('use_cuda', bool)
    ])

AgentModuleConfig = NamedTuple("AgentModuleConfig", [
    ('time_horizon', int),
    ('feat_vec_size', int),
    ('movement_dim_size', int),
    ('goal_size', int),
    ('vocab_size', int),
    ('utterance_processor', GoalPredictingProcessingModuleConfig),
    ('physical_processor', ProcessingModuleConfig),
    ('action_processor', ActionModuleConfig),
    ('word_counter', WordCountingModuleConfig),
    ('use_utterances', bool),
    ('penalize_words', bool),
    ('use_cuda', bool),
    ('df_utterance_col_name', list)
    ])

RunModuleConfig = NamedTuple("RunModuleConfig", [
    ('save_to_a_new_dir', bool),
    ('creating_data_set_mode', bool),
    ('upload_trained_model', bool),
    ('dir_upload_model', str),
    ('folder_dir', str),
    ('corpus', str),
    ('create_utterance_using_old_code',bool),
    ])

default_training_config = TrainingConfig(
        num_epochs=DEFAULT_NUM_EPOCHS,
        learning_rate=DEFAULT_LR,
        load_model=False,
        load_model_file="",
        save_model=SAVE_MODEL,
        save_model_file=DEFAULT_MODEL_FILE,
        use_cuda=False,
        no_utterances=False)

default_word_counter_config = WordCountingModuleConfig(
        vocab_size=DEFAULT_VOCAB_SIZE,
        oov_prob=DEFAULT_OOV_PROB,
        use_cuda=False)

default_game_config = GameConfig(
    DEFAULT_BATCH_SIZE,
    DEFAULT_WORLD_DIM,
    MAX_AGENTS,
    MAX_LANDMARKS,
    MIN_AGENTS,
    MIN_LANDMARKS,
    NUM_SHAPES,
    NUM_COLORS,
    USE_UTTERANCES,
    DEFAULT_VOCAB_SIZE,
    DEFAULT_HIDDEN_SIZE,
    False,
    DEFAULT_TIME_HORIZON,
    DEFAULT_NUM_EPOCHS,)

default_run_config = RunModuleConfig(
    save_to_a_new_dir=DEFAULT_SAVE_TO_A_NEW_DIR,
    creating_data_set_mode=DEFAULT_CREATING_DATASET_MODE,
    upload_trained_model=DEFAULT_UPLOAD_TRAINED_MODEL,
    dir_upload_model=DEFAULT_DIR_UPLOAD_MODEL,
    folder_dir=DEFAULT_FOLDER_DIR,
    corpus=DEFAULT_CORPUS,
    create_utterance_using_old_code=DEFAULT_USE_OLD_UTTERANCE_CODE,)


if USE_UTTERANCES:
    feat_size = DEFAULT_FEAT_VEC_SIZE*3
else:
    feat_size = DEFAULT_FEAT_VEC_SIZE*2


def get_processor_config_with_input_size(input_size):
    return ProcessingModuleConfig(
        input_size=input_size,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        dropout=DEFAULT_DROPOUT)


default_action_module_config = ActionModuleConfig(
        goal_processor=get_processor_config_with_input_size(constants.GOAL_SIZE),
        action_processor=get_processor_config_with_input_size(feat_size),
        hidden_size=DEFAULT_HIDDEN_SIZE,
        dropout=DEFAULT_DROPOUT,
        movement_dim_size=constants.MOVEMENT_DIM_SIZE,
        movement_step_size=constants.MOVEMENT_STEP_SIZE,
        vocab_size=DEFAULT_VOCAB_SIZE,
        use_utterances=USE_UTTERANCES,
        use_cuda=False)

default_goal_predicting_module_config = GoalPredictingProcessingModuleConfig(
    processor=get_processor_config_with_input_size(DEFAULT_VOCAB_SIZE),
    hidden_size=DEFAULT_HIDDEN_SIZE,
    dropout=DEFAULT_DROPOUT,
    goal_size=constants.GOAL_SIZE)

default_agent_config = AgentModuleConfig(
        time_horizon=DEFAULT_TIME_HORIZON,
        feat_vec_size=DEFAULT_FEAT_VEC_SIZE,
        movement_dim_size=constants.MOVEMENT_DIM_SIZE,
        utterance_processor=default_goal_predicting_module_config,
        physical_processor=get_processor_config_with_input_size(constants.MOVEMENT_DIM_SIZE + constants.PHYSICAL_EMBED_SIZE),
        action_processor=default_action_module_config,
        word_counter=default_word_counter_config,
        goal_size=constants.GOAL_SIZE,
        vocab_size=DEFAULT_VOCAB_SIZE,
        use_utterances=USE_UTTERANCES,
        penalize_words=PENALIZE_WORDS,
        use_cuda=False,
        df_utterance_col_name = DEFAULT_DF_UTTERANCE_COL_NAME)

default_utterance_config = UtteranceConfig(
        init_range=DEFAULT_INIT_RANGE,
        nhid_lang=DEFAULT_NHID_LANG,
        nembed_word=DEFAULT_NEMBED_WORDS,
        nhid_ctx=DEFAULT_NHID_CTX,
        dropout=DEFAULT_DROPOUT,
        momentum=DEFAULT_MOMENTUM,
        lr=DEFAULT_LR,
        nesterov=DEFAULT_NESTEROV,
        clip=DEFAULT_CLIP,
        batch_size=default_game_config.batch_size,
        temperature=DEFAULT_TEMPERATURE)


def get_utterance_config():
    return UtteranceConfig(
        init_range=default_utterance_config.init_range,
        nhid_lang=default_utterance_config.nhid_lang,
        nembed_word=default_utterance_config.nembed_word,
        nhid_ctx=default_utterance_config.nhid_ctx,
        dropout=default_utterance_config.dropout,
        momentum=default_utterance_config.momentum,
        lr=default_utterance_config.lr,
        nesterov=default_utterance_config.nesterov,
        clip=default_utterance_config.clip,
        batch_size=default_game_config.batch_size,
        temperature=default_utterance_config.temperature)
        # init_range=kwargs['init_range'] or default_utterance_config.init_range,
        # nhid_lang=kwargs['nhid_lang'] or default_utterance_config.nhid_lang,
        # nembed_word=kwargs['nembed_word'] or default_utterance_config.nembed_word,
        # nhid_ctx=kwargs['nhid_ctx'] or default_utterance_config.nhid_ctx,
        # dropout=kwargs['dropout'] or default_utterance_config.dropout,
        # momentum=kwargs['momentum'] or default_utterance_config.momentum,
        # lr=kwargs['lr'] or default_utterance_config.lr,
        # nesterov=kwargs['nesterov'] or default_utterance_config.nesterov,
        # clip=kwargs['clip'] or default_utterance_config.clip,
        # batch_size = kwargs['batch_size'] or default_game_config.batch_size,
        # temperature=kwargs['temperature'] or default_utterance_config.temperature)


def get_training_config(kwargs,folder_dir):
    return TrainingConfig(
            num_epochs=kwargs['n_epochs'] or default_training_config.num_epochs,
            learning_rate=kwargs['learning_rate'] or default_training_config.learning_rate,
            load_model=bool(kwargs['load_model_weights']),
            load_model_file=kwargs['load_model_weights'] or default_training_config.load_model_file,
            save_model=default_training_config.save_model,
            save_model_file= folder_dir + (kwargs['save_model_weights'] or default_training_config.save_model_file),
            use_cuda=kwargs['use_cuda'],
            no_utterances=kwargs['no_utterances'],)


def get_game_config(kwargs):
    return GameConfig(
        batch_size=kwargs['batch_size'] or default_game_config.batch_size,
        world_dim=kwargs['world_dim'] or default_game_config.world_dim,
        max_agents=kwargs['max_agents'] or default_game_config.max_agents,
        min_agents=kwargs['min_agents'] or default_game_config.min_agents,
        max_landmarks=kwargs['max_landmarks'] or default_game_config.max_landmarks,
        min_landmarks=kwargs['min_landmarks'] or default_game_config.min_landmarks,
        num_shapes=kwargs['num_shapes'] or default_game_config.num_shapes,
        num_colors=kwargs['num_colors'] or default_game_config.num_colors,
        use_utterances=not kwargs['no_utterances'],
        vocab_size=kwargs['vocab_size'] or default_game_config.vocab_size,
        memory_size=default_game_config.memory_size,
        use_cuda=kwargs['use_cuda'],
        time_horizon=kwargs['n_timesteps'] or default_game_config.time_horizon,
        num_epochs=kwargs['n_epochs'] or default_game_config.num_epochs,
    )


def get_agent_config(kwargs):
    vocab_size = kwargs['vocab_size'] or DEFAULT_VOCAB_SIZE
    use_utterances = (not kwargs['no_utterances'])
    use_cuda = kwargs['use_cuda']
    penalize_words = kwargs['penalize_words']
    oov_prob = kwargs['oov_prob'] or DEFAULT_OOV_PROB
    if use_utterances:
        feat_vec_size = DEFAULT_FEAT_VEC_SIZE*3
    else:
        feat_vec_size = DEFAULT_FEAT_VEC_SIZE*2
    utterance_processor = GoalPredictingProcessingModuleConfig(
            processor=get_processor_config_with_input_size(vocab_size),
            hidden_size=DEFAULT_HIDDEN_SIZE,
            dropout=DEFAULT_DROPOUT,
            goal_size=constants.GOAL_SIZE)
    action_processor = ActionModuleConfig(
            goal_processor=get_processor_config_with_input_size(constants.GOAL_SIZE),
            action_processor=get_processor_config_with_input_size(feat_vec_size),
            hidden_size=DEFAULT_HIDDEN_SIZE,
            dropout=DEFAULT_DROPOUT,
            movement_dim_size=constants.MOVEMENT_DIM_SIZE,
            movement_step_size=constants.MOVEMENT_STEP_SIZE,
            vocab_size=vocab_size,
            use_utterances=use_utterances,
            use_cuda=use_cuda)
    word_counter = WordCountingModuleConfig(
            vocab_size=vocab_size,
            oov_prob=oov_prob,
            use_cuda=use_cuda)

    return AgentModuleConfig(
            time_horizon=kwargs['n_timesteps'] or default_agent_config.time_horizon,
            feat_vec_size=default_agent_config.feat_vec_size,
            movement_dim_size=default_agent_config.movement_dim_size,
            utterance_processor=utterance_processor,
            physical_processor=default_agent_config.physical_processor,
            action_processor=action_processor,
            word_counter=word_counter,
            goal_size=default_agent_config.goal_size,
            vocab_size=vocab_size,
            use_utterances=use_utterances,
            penalize_words=penalize_words,
            use_cuda=use_cuda,
            df_utterance_col_name=default_agent_config.df_utterance_col_name
            )


def get_run_config(kwargs):
    save_to_a_new_dir = kwargs['save_to_a_new_dir'] or default_run_config.save_to_a_new_dir
    creating_data_set_mode = kwargs['creating_data_set_mode'] or default_run_config.creating_data_set_mode
    upload_trained_model = kwargs['upload_trained_model'] or default_run_config.upload_trained_model
    if save_to_a_new_dir:
        folder_dir = create_new_dir()
    else:
        folder_dir = default_run_config.folder_dir
    if creating_data_set_mode:
        # corpus = None
        corpus = data.WordCorpus('data' + os.sep, freq_cutoff=20, verbose=True)

    else:
        corpus = data.WordCorpus('data' + os.sep, freq_cutoff=20, verbose=True)
    if upload_trained_model:
        dir_upload_model = kwargs['dir_upload_model'] or DEFAULT_DIR_UPLOAD_MODEL
        dir_upload_model = dir_upload_model + os.sep + DEFAULT_MODEL_FILE
    else:
        dir_upload_model = DEFAULT_DIR_UPLOAD_MODEL
    return RunModuleConfig(
        save_to_a_new_dir=save_to_a_new_dir,
        creating_data_set_mode=creating_data_set_mode,
        upload_trained_model=upload_trained_model,
        dir_upload_model= dir_upload_model,
        folder_dir=folder_dir,
        corpus=corpus,
        create_utterance_using_old_code=kwargs['create_utterance_using_old_code'] or default_run_config.create_utterance_using_old_code
          )


def create_new_dir():
    """
       Create the new folder for the experiment and in it designated folders for the plots, movies and the tensorboard files.
       return: Full path of the new folder (str), and only the folder name
    """
    folder_dir = str(Path(os.getcwd())) + os.sep + str(time.strftime("%H%M-%d%m%Y")) + os.sep
    folder_name_suffix = 0
    while True:
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir + os.sep) #open the folder
            break
        else:
            folder_name_suffix += 1 # add an index to the folder so we will not overwrite the previous folder
            folder_dir = folder_dir + "_" + str(folder_name_suffix)
    plots_dir = folder_dir + 'plots' + os.sep
    movies_dir = folder_dir + 'movies' + os.sep
    tensorboard_dir = folder_dir + 'tensorboard' + os.sep
    os.makedirs(movies_dir)
    os.makedirs(tensorboard_dir)
    os.makedirs(plots_dir)

    return folder_dir