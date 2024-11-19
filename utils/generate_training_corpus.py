from game_engine import generate_random_game, move_log_to_string, history_to_board_states
import time
import argparse
import os

def generate_move_log_corpus(num_games=100, to_write_location="datasets/test_file.txt"):
    write_initial_newline=os.path.exists(to_write_location)
    games_as_list=[move_log_to_string(generate_random_game(), insert_terminal_XX=True) for _ in range(num_games)]
    games_as_string="\n".join(games_as_list)
    with open(to_write_location, 'a') as f:
        if write_initial_newline:
            f.write("\n")
        f.write(games_as_string)
    return

def generate_move_log_and_board_state_corpus(num_games=100, to_write_location="datasets/test_file.txt"):
    write_initial_newline=os.path.exists(to_write_location)
    games=[generate_random_game() for _ in range(num_games)]
    games_as_list=[move_log_to_string(game, insert_terminal_XX=True)+"/"+history_to_board_states(game) for game in games]
    games_as_string="\n".join(games_as_list)
    with open(to_write_location, 'a') as f:
        if write_initial_newline:
            f.write("\n")
        f.write(games_as_string)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate move log and board state corpus.")
    parser.add_argument('--num_batches', type=int, help='Number of batches to generate', default=10)
    parser.add_argument('--batch_size', type=int, help='Size of each batch', default=100)
    parser.add_argument('--write_location', type=str, help='File location to write the data', default="datasets/probe_test_corpus.txt")
    parser.add_argument('--save_board_state', action='store_true', help='Flag to generate board state along with move log')

    args = parser.parse_args()
    
    num_batches = args.num_batches
    batch_size = args.batch_size
    write_location = args.write_location
    save_board_state = args.save_board_state

    directory = os.path.dirname(write_location)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    for n in range(num_batches):
        start = time.time()
        if save_board_state:
            generate_move_log_and_board_state_corpus(batch_size, write_mode='a', to_write_location=write_location)
        else:
            generate_move_log_corpus(batch_size, to_write_location=write_location)
        end = time.time()
        print(f"Time for batch size {batch_size}: {end-start} seconds")

