
import ConnectFour as cf

from random import randint

def main():
    play(True, 'random')



#class Model(object):
#    def __play__(self, board):


def play(player_first, model):

    player = cf.RED
    computer = cf.YELLOW

    board = cf.create_board()
    print cf.draw(board)

    while True:

        player_move = input('Next move (Enter 0-7): \n')
        cf.play(board, player_move, player)
        print 'PLAYER MOVE: \n'
        print cf.draw(board)
        if cf.is_winner(board, player):
            print "PLAYER WINS"
            return

        computer_move = randint(0, cf.NUM_COLUMNS-1)
        cf.play(board, computer_move, computer)
        print 'COMPUTER MOVE: \n'
        print cf.draw(board)
        if cf.is_winner(board, computer):
            print "COMPUTER WINS"
            return


if __name__ == '__main__':
    main()
