class Tiktaktoe:
    game = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    player_turn = 1
    turn_nr = 0
    auto_output = False

    def move(self, line, row):
        if line > len(self.game) or row > len(self.game[line]):
            return "Out of Index"
        elif self.game[line][row] != 0:
            return "Field already set"
        else:
            self.game[line][row] = player_turn
            if check_won():
                if self.auto_output:
                    output()
                return f"Player {self.player_turn} has won!"
            else:
                self.next_move()
                return f"It's your turn, {self.player_turn}!"

    def next_move(self):
        if self.player_turn == 1:
            self.player_turn = 2
        else:
            self.player_turn = 1
        self.turn_nr += 1
        if self.auto_output:
                    output()

    def check_won(self):
        pass

    def output(self):
        print("Player1: X, Player2: O\n-" + ("-"*len(self.game[0])*2))
        for line in self.game:
            print(" " + str(line).replace("0", "-").replace("1", "X").replace("2", "O").replace(",", "").replace("[", "").replace("]", ""))
        print("-" + "-"*len(self.game[0])*2)