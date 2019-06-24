#creat data set
vocab = ['go', 'to', 'agent', 'red', 'green', 'blue', 'landmark',
             'circle', 'triangle', 'continue', 'next', 'ahead', 'done',
             'good', 'stay', 'goal']

colors = ['red', 'green', 'blue']
shapes = ['circle', 'triangle']

sentence_form = ["<color1> agent go to <color2> landmark",
                 "<color1> <shape1> agent go to <color2> <shape2> landmark",
                 "<shape1> agent go to <shape2> landmark",
                 "<color1> agent go to <shape1> landmark",
                 "<shape1> agent go to <color1> landmark",
                 "<color1> agent stay",
                 "<color1> <shape1> agent stay",
                 "<shape1> agent stay",
                 "<color1> agent continue",
                 "<color1> <shape1> agent continue",
                 "<shape1> agent continue",
                 "<color1> agent is done",
                 "<color1> <shape1> agent is done",
                 "<shape1> agent is done"
                 "<color1> good job",
                 "<color1> <shape1> good job",
                 "<shape1> good job",
                 "you go girl"]



def create_dataset(res):
    for shape1 in shapes:
        for color1 in colors:
            for shape2 in shapes:
                for color2 in colors:
                    tmp = sentence_form.copy()
                    tmp = [t.replace("<color1>", color1).replace("<shape1>", shape1).replace("<color2>", color2).replace("<shape2>", shape2) for t in tmp]
                    res+=tmp
    print(list(set(res)))


create_dataset([])