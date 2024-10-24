import numpy as np


# contains information about state, children nodes and states
class Node:
    def __init__(self, board, turn, parent=None):
        # state contains info about the turn too
        self.board: np.ndarray = board
        # string, contains whose turn it is to play NEXT
        self.turn = turn
        # parent node
        self.parent = parent
        # number of visits to this node
        self.visits = 0
        # 1 for player 1 win, -1 for player 2 win, 0 for draws
        self.stats = {1: 0, -1: 0, 0: 0}
        # all the children nodes in the game tree
        self.children = []
        self.unexplored = TTT.get_legal(self.board)

    @property
    def Q(self):
        wins = self.stats[self.turn]
        losses = self.stats[-1 * self.turn]
        # NOTE: we ignore draws here, but value function need to account for BOTH wins and losses!!!
        return wins - losses

    @property
    def N(self):
        return self.visits


class TTT:
    @staticmethod
    def is_terminal(board):
        return TTT.get_winner(board) is not None

    @staticmethod
    def get_winner(board):
        # return 1 if player 1 wins, -1 if player 2 wins, and None otherwise

        # check rows
        for row in range(3):
            if abs(np.sum(board[row, :])) == 3:
                return np.sum(board[row, :]) // 3
        # check col
        for col in range(3):
            if abs(np.sum(board[:, col])) == 3:
                return np.sum(board[:, col]) // 3

        # check left to right diag
        diag_sum = np.trace(board)
        # check if either player won
        if abs(diag_sum) == 3:
            # do not absolute this
            return diag_sum // 3

        # check other diag
        other_diag_sum = np.trace(np.fliplr(board))
        if abs(other_diag_sum) == 3:
            # do not absolute this
            return other_diag_sum // 3

        # check for draw. this result will be sent up, but zero will be updated
        if not np.any(board == 0):
            return 0

        # didnt find winner
        return None

    @staticmethod
    # gets all the possible moves for a board state
    def get_legal(board):
        # returns a list of legal (row,col) positions
        positions = np.where(board == 0)
        row_pos, col_pos = positions[0], positions[1]
        return list(zip(row_pos, col_pos))


class MCTS:
    # Upper confidence bound for trees
    def UCT(self, node, c):
        all_v = [
            (child.Q / child.N + c * np.sqrt(np.log(node.visits) / child.N))
            for child in node.children
        ]
        # use argmax here, not max!
        return node.children[int(np.argmax(all_v))]

    def select(self, root: Node):
        node = root
        # keep going until terminal node
        while not TTT.is_terminal(node.board):
            # still has unexplored actions
            if len(node.unexplored) > 0:
                return node
            # fully expanded node
            # go one level deeper, on its children
            else:
                node = self.UCT(node, np.sqrt(2))
        # terminal node
        return node

    def expand(self, node: Node) -> Node:
        # where returns a tuple of (row indices, corresponding col indices)
        # choose the first child in the list of available positions, and remove from unexplored
        row, col = node.unexplored.pop()
        new_board = node.board.copy()
        # fill in the appropriate value for the action to be played
        new_board[row, col] = node.turn
        # add to game tree
        # flip the turn for the next player
        child = Node(new_board, -1 * node.turn, node)
        node.children.append(child)
        return child

    def simulate(self, node: Node):
        # the next player to move
        next_turn = node.turn
        # need to make a copy of this
        cur_board = node.board.copy()
        while not TTT.is_terminal(cur_board):
            # rollout policy
            cur_board = self.rollout_policy(cur_board, next_turn)
            # flip to next turn
            next_turn = -1 * next_turn

        # terminal state, should return -1,0,or 1
        return TTT.get_winner(cur_board)

    def backpropagate(self, node: Node, reward):
        node.visits += 1
        node.stats[reward] += 1
        # backprop up parents
        if node.parent:
            self.backpropagate(node.parent, reward)

    # to make the game tree better
    def train(self, node):
        leaf = self.select(node)
        # only expand, if the node is not fully expanded
        if len(leaf.unexplored) > 0:
            child = self.expand(leaf)
            reward = self.simulate(child)
            self.backpropagate(child, reward)
        # simulate on a fully expanded node
        else:
            reward = self.simulate(leaf)
            self.backpropagate(leaf, reward)

    # actually choose an action for exploitation
    def choose(self, node: Node):
        # has children in the game tree
        if len(node.children) > 0:
            # pure exploitation
            return self.UCT(node, 0)
        # create a dummy node
        else:
            new_board = self.rollout_policy(node.board, node.turn)
            # flip the turn, no parent
            return Node(new_board, -1 * node.turn, None)

    def rollout_policy(self, board, turn):
        # takes a random action on board using that turn and returns a new board
        # first get all possible positions on this board
        all_possible = TTT.get_legal(board)
        random_action = all_possible[np.random.randint(len(all_possible))]
        row, col = random_action
        new_board = board.copy()
        # edit the position of the cur_board
        new_board[row, col] = turn
        return new_board


# extra debug function
def debug(node: Node, layer):
    print("\n\n")
    print("=====================")
    print(node.board)
    print("child in layer ", layer)
    print("stats of", node.stats)
    for child in node.children:
        debug(child, layer + 1)


# MAIN function
mcts = MCTS()
init_board = np.zeros(shape=(3, 3))
# player 1 to start first
root = Node(init_board, 1, None)

# train
for i in range(100000):
    if i % 10000 == 0:
        print("training iteration ", i)
    mcts.train(root)

# debug(root, 0)

# autoplay
print("===GAME START===")
node = root
print(node.board)
while not TTT.is_terminal(node.board):
    print("\n")
    print("Board:")
    node = mcts.choose(node)
    print(node.board)

print("===GAME END===")
# mapping of values to outcome
V2O = {1: "P1 wins", -1: "P2 wins", 0: "Draw"}
print("Outcome of the game is ", V2O[TTT.get_winner(node.board)])
print("root node stats are", root.stats)
