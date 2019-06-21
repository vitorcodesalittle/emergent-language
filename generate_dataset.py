import random

vocab = ['go', 'to', 'agent', 'red', 'green', 'blue', 'landmark', 'circle',
         'triangle', 'continue', 'next', 'ahead', 'done', 'good', 'stay',
         'goal']

colors = ['red', 'green', 'blue']
shapes = ['circle', 'triangle']


goto_sentences= [
    "<agent_color> agent go to <lm_color> landmark",
    "<agent_color> <agent_shape> agent go to <lm_color> <lm_shape> landmark",
    "<agent_shape> agent go to <lm_shape> landmark",
    "<agent_color> agent go to <lm_shape> landmark",
    "<agent_shape> agent go to <lm_color> landmark"]
continue_sentences = [
    "<agent_color> agent continue",
    "<agent_color> <agent_shape> agent continue",
    "<agent_shape> agent continue",
    "you are on the right track"]
stay_sentences=[
    "<agent_color> agent stay",
    "<agent_color> <agent_shape> agent stay",
    "<agent_shape> agent stay"]
done_sentences=[
    "<agent_color> agent is done",
    "<agent_color> <agent_shape> agent is done",
    "<agent_shape> agent is done"
    "<agent_color> good job",
    "<agent_color> <agent_shape> good job",
    "<agent_shape> good job",
    "you go girl"
]

sentence_form = goto_sentences + continue_sentences+stay_sentences+done_sentences


@staticmethod
def generate_sentence(agent_color, agent_shape, lm_color, lm_shape):
    sentence = random.choice(sentence_form)
    sentence.replace('<agent_color>', agent_color)
    sentence.replace('<agent_shape>', agent_shape)
    sentence.replace('<lm_color>', lm_color)
    sentence.replace('<lm_shape>', lm_shape)
    return sentence

# agent_num = 2
# lm_num = random.randint(2,3)
# locations = [(random.uniform(0, 16), random.uniform(0, 16)) for i in agent_num + lm_num]
# colors = [random.choice(colors) for i in agent_num + lm_num]
# shapes = [random.choice(shapes) for i in agent_num + lm_num]
# input = "<input> "
# for color, shape, location in zip(colors, shapes, locations):
#     input += color + shape + location
# input+="</input>"
# first_turn = random.randint(1, 2)
# second_turn = 3 - first_turn
# dialogue = "<dialogue> AGENT_"+first_turn+": "
# dialogue+=

