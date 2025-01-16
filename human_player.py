import os
import platform


class Human_Player:
    def _clear_terminal(self):
        if platform.system() == "Windows":
            os.system("cls")  # Windows-specific command
        else:
            os.system("clear")  # Unix/Linux/Mac-specific command

    def _print_actions(self,legal_actions):
        action_num_to_str = {0: "Fold: 0",
                             1: "Call 1",
                             2: "Raise half pot: 2",
                             3: "All in: 3",
                             4: "Raise full pot: 4"
                             }
        for action in legal_actions:
            print(action_num_to_str[action], '\n')

    def take_action(self, state):
        self._clear_terminal()
        current_player = state.current_player()
        print(state.information_state_string(current_player))
        print("Choose action:")
        self._print_actions(state.legal_actions(current_player))
        action = int(input())

        while action not in state.legal_actions(current_player):
            print("Error, not a legal action, try again:")
            action = int(input())

        return action
        
        

