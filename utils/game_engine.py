import numpy as np
from random import choice, seed 
import matplotlib.pyplot as plt
import os
import torch
from functools import cache

class OthelloGame:

    def __init__(self, board_size=8):
        self.board_size=board_size
        self.board=np.zeros((board_size,board_size))
        self.board[3,3]=-1
        self.board[3,4]=1
        self.board[4,3]=1
        self.board[4,4]=-1
        self.turns_history=[]
        self.just_flipped=[]
        self.active_player=1
        self.directions=self.get_directions()

    def make_move(self, coordinates):
        '''
        adjusts the board state as if a move was made at coordinates, returning True if successful
        if the move is illegal, returns False
        inputs:
            coordinates: a 2-ple of positions, from {0,1,...,7}^2
        '''
        pieces_to_flip=self.pieces_to_flip_with_move_at(coordinates)
        if pieces_to_flip:
            for location in pieces_to_flip:
                self.board[tuple(location)]*=-1
            self.just_flipped=pieces_to_flip
            self.board[coordinates]=self.active_player
            self.turns_history.append(coordinates)
            self.active_player*=-1
            return True
        else:
            return False

    @cache 
    def coordinates_on_board(self, coordinates):
        '''
        checks if the listed coordinates are on the board
        '''
        return coordinates[0]>=0 and coordinates[1]>=0 and coordinates[0]<self.board_size and coordinates[1]<self.board_size
                             

    def get_directions(self):
        '''
        returns a list of the eight directions
        '''
        all_directions=[np.array([x,y]) for x in [-1,0,1] for y in [-1,0,1]]
        all_directions.pop(4) # this is the (0,0) direction so should be removed
        return all_directions

    def pieces_to_flip_with_move_at(self, coordinates, early_stop=False):
        '''
        returns a list of the pieces that would be flipped if the active player placed a piece at coordinates
        returns the empty list if the move is invalid
        '''
        if self.board[coordinates]!=0:
            return []
        pieces_to_flip=[]
        # directions=self.get_directions()
        for direction in self.directions:
            enemy_pieces_in_this_direction=[]
            next_coordinates=coordinates
            for _ in range(self.board_size):
                next_coordinates=tuple(next_coordinates+direction)
                if not self.coordinates_on_board(tuple(next_coordinates)):
                    break
                next_move_state=self.board[tuple(next_coordinates)]*self.active_player
                if next_move_state==0:
                    break
                elif next_move_state==-1:
                    #enemy piece there
                    enemy_pieces_in_this_direction.append(next_coordinates)
                elif next_move_state==1:
                    pieces_to_flip.extend(enemy_pieces_in_this_direction)
                    if early_stop and enemy_pieces_in_this_direction:
                        return True
                    break
        return pieces_to_flip

    # def is_legal_move(self, coordinates):
    #     return self.pieces_to_flip_with_move_at(coordinates, early_stop=True)

    def list_legal_moves(self):
        return [(x,y) for x in range(self.board_size) for y in range(self.board_size) if self.pieces_to_flip_with_move_at((x,y), early_stop=True)]

    def print_board(self):
        piece_markers={0:".", 1:"O", -1:"*"}
        top_row=" ABCDEFGH"
        to_print_lines=[top_row]
        for i, row in enumerate(self.board.transpose()):
            this_row_as_text=f"{i+1}{''.join([piece_markers[piece] for piece in row])}"
            to_print_lines.append(this_row_as_text)
        to_print="\n".join(to_print_lines)
        print(to_print)

    def plot_board(self, directory="game_histories/example"):
        black_pieces_locations=[]
        white_pieces_locations=[]
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[(x,y)]==1:
                    black_pieces_locations.append((x,y))
                elif self.board[(x,y)]==-1:
                    white_pieces_locations.append((x,y))
        
        fig, ax = plt.subplots()
        ax.grid()
        ax.set_axisbelow(True)
        ax.set_aspect(1)
        ax.set_axisbelow(True)

        marker_size=1000

        plt.scatter(x=np.array(black_pieces_locations)[:,0],y=np.array(black_pieces_locations)[:,1], c="black", s=marker_size, linewidths=1.5, edgecolors="black")
        plt.scatter(x=np.array(white_pieces_locations)[:,0],y=np.array(white_pieces_locations)[:,1], c="white", s=marker_size, linewidths=1.5, edgecolors="black")
        
        plt.scatter(x=np.array(self.turns_history)[-1,0],y=np.array(self.turns_history)[-1,1], marker="s", c="red", s=100)
        plt.scatter(x=np.array(self.just_flipped)[:,0],y=np.array(self.just_flipped)[:,1], c="orange", s=100)


        plt.gca().invert_yaxis()

        plt.yticks(range(self.board_size),labels=range(1, self.board_size+1))
        plt.xticks(range(self.board_size), labels=["A", "B", "C", "D", "E", "F", "G", "H"])
        plt.title(f"Turn {len(self.turns_history)}")


    def board_to_string(self):
        representations_by_row=[]
        for i, row in enumerate(self.board.transpose()):
            this_row_representation=f"{' '.join([str(int(piece)) for piece in row])}"
            representations_by_row.append(this_row_representation)
        board_as_string=" ".join(representations_by_row)
        return board_as_string

def generate_random_game():
    '''
    returns a list of game moves chosen at random from the legal moves
    '''
    game=OthelloGame()
    legal_moves=game.list_legal_moves()
    while legal_moves:
        next_move=choice(legal_moves)
        game.make_move(next_move)
        legal_moves=game.list_legal_moves()
    return game.turns_history

def play_back_move_log(move_log):
    game=OthelloGame()
    for move in move_log:
        game.make_move(move)
        game.print_board()

def plot_back_move_log(move_log, directory="game_histories/example"):
    game=OthelloGame()
    if not os.path.exists(directory):
        os.mkdir(directory)
    for n, move in enumerate(move_log):
        game.make_move(move)
        game.plot_board()
        plt.savefig(f"{directory}/turn{len(game.turns_history)}")
        plt.close()

def coordinates_to_string(coord_as_tuple):
    first_coord="ABCDEFGH"[coord_as_tuple[0]]
    second_coord=coord_as_tuple[1]+1
    return f"{first_coord}{second_coord}"

def move_log_to_string(move_log_list,insert_terminal_XX=False):
    moves=[coordinates_to_string(move) for move in move_log_list]
    if insert_terminal_XX:
        moves.append("XX")
    return " ".join(moves)

def string_to_coordinates(coord_as_string):
    first_coord={"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6,"H":7}[coord_as_string[0]]
    second_coord=int(coord_as_string[1])-1
    return (first_coord, second_coord)

def string_to_move_log(move_log_string):
    move_log_split=move_log_string.split(" ")
    if move_log_split[-1]=="XX":
        move_log_split.pop()
    return [string_to_coordinates(move) for move in move_log_split]

def int_to_coordinates(coords_as_int, game_board_size=8):
    return (coords_as_int%game_board_size, coords_as_int//game_board_size)

def coordinates_to_int(coords_as_tuple,game_board_size=8):
    return coords_as_tuple[0]+game_board_size*coords_as_tuple[1]

def tokens_list():
    tokens=[f"{letter}{number}" for number in range(1,9) for letter in "ABCDEFGH" ]
    tokens.append("XX") #end-of-game token
    tokens.append("PP") #pad token
    # tokens.append("SS") #start-of-game token
    return tokens



def test_conversion():
    random_move_log=generate_random_game()
    y=move_log_to_string(random_move_log)
    z=string_to_move_log(y)
    if np.all(np.array(random_move_log)==np.array(z)):
        print("Yay!")
    else:
        print("NOOOOOOOOO!")

# for _ in range(100):
#     test_conversion()

def history_to_legal_moves(move_log_tensor, trim_to_length_64=True):
    '''
    takes in a tensor of ints, the move log, and for each game computes the legal next moves at each stage, returned as a 1-hot tensor of ints
    input: shape (B, W) where B is the batch size, W is the window size
    output: shape (B, W, V) where V is the vocabulary size (64+N where N is the number of additional tokens)
    '''
    valid_moves_by_game=[]
    vocab_size=len(tokens_list())
    for game_log in move_log_tensor:
        valid_moves_by_turn=[]
        game=OthelloGame()
        for move in game_log:
            if move in [64,65]:
                # game was terminated or is in the padding after termination
                valid_moves_as_int_list=[65]
            else:
                coords_as_tuple=int_to_coordinates(move, game_board_size=game.board_size)
                game.make_move(coords_as_tuple)
                valid_moves_as_tuple_list=game.list_legal_moves()
                valid_moves_as_int_list=[coordinates_to_int(x, game.board_size) for x in valid_moves_as_tuple_list]
                if not valid_moves_as_int_list:
                    #game has ended early, put in a pad token
                    valid_moves_as_int_list=[65]
            valid_moves_one_hot=[int(x in valid_moves_as_int_list) for x in range(vocab_size)]
            valid_moves_by_turn.append(valid_moves_one_hot)
        valid_moves_by_game.append(valid_moves_by_turn)
    to_return= torch.tensor(valid_moves_by_game)
    if trim_to_length_64:
        to_return=to_return[:, :, :64]
    return to_return

def history_to_board_states(game_log):
    board_states_by_turn=[]
    game=OthelloGame()
    for move in game_log:
        board_states_by_turn.append(game.board_to_string())
        game.make_move(move)
    board_states=";".join(board_states_by_turn)
    return board_states



# if __name__=="__main__":
#     seed(0)
#     random_move_log=generate_random_game()
#     print(random_move_log)
#     print(move_log_to_string(random_move_log))
#     plot_back_move_log(random_move_log)