import torch

def encode_to_fixed_length(game_log, sequence_length=64):
    '''
    converts a game log string into an int sequence
    the ints correspond to the tokens
    model has 66 tokens:
        64 for the positions, 
        1 (XX) for "end of game" 
        1 (PP) for padding
    '''
    moves=game_log.split(" ")
    while len(moves)<sequence_length:
        moves.append("PP")
    if len(moves)>sequence_length:
        moves=moves[:sequence_length]
    tokens=tokens_list()
    moves_as_ints=[tokens.index(move) for move in moves]
    return moves_as_ints

def encode(game_log):
    ints_to_tokens, tokens_to_ints=tokens_dicts_bidirectional()
    moves=game_log.split(" ")
    int_sequence=torch.tensor([tokens_to_ints[move] for move in moves])
    return int_sequence

def tokens_list():
    tokens=[f"{letter}{number}" for letter in "ABCDEFGH" for number in range(1,9)]
    tokens.append("XX") #end-of-game token
    tokens.append("PP") #pad token
    return tokens

def tokens_dicts_bidirectional():
    tokens=tokens_list()
    ints_to_tokens={n:token for n, token in enumerate(tokens)}
    tokens_to_ints={token:n for n, token in enumerate(tokens)}
    return ints_to_tokens, tokens_to_ints


def decode(int_sequence):
    ints_to_tokens, tokens_to_ints=tokens_dicts_bidirectional()
    game_log=" ".join([ints_to_tokens[int(x)] for x in int_sequence])
    return game_log


def load_data(data_location, train_split=.9):
    with open(data_location) as f:
        input_as_text=f.read()
    input_as_text=input_as_text.replace("\n", " ")
    all_data=encode(input_as_text)
    training_cutoff=int(len(all_data)*train_split)
    train_data=all_data[:training_cutoff]
    test_data=all_data[training_cutoff:]
    return train_data, test_data


