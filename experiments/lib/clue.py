from itertools import chain, cycle
import numpy as np
from ortools.sat.python import cp_model
import pandas as pd
import random
from typing import Optional


class Clue:
    suspects = [
        "Miss Scarlet",
        "Mr. Green",
        "Mrs. White",
        "Mrs. Peacock",
        "Colonel Mustard",
        "Professor Plum",
        # Additional Master Detective Suspects
        "Miss Peach",
        "Sgt. Gray",
        "Monsieur Brunette",
        "Madame Rose",
    ]

    weapons = [
        "Candlestick",
        "Knife",
        "Lead Pipe",
        "Revolver",
        "Rope",
        "Wrench",
        # Additional Master Detective Weapons
        "Horseshoe",
        "Poison",
    ]

    rooms = [
        "Hall",
        "Lounge",
        "Dining Room",
        "Kitchen",
        "Ballroom",
        "Conservatory",
        "Billiard Room",
        "Library",
        "Study",
        # Additional Master Detective Rooms
        "Carriage House",
        "Cloak Room",
        "Trophy Room",
        "Drawing Room",
        "Gazebo",
        "Courtyard",
        "Fountain",
        "Studio",
    ]

    motives = [
        "Revenge",
        "Jealousy",
        "Greed",
        "Blackmail",
        "Power",
        "Cover-up",
        "Betrayal",
        "Obsession",
        "Inheritance",
        "Self-preservation",
    ]

    @staticmethod
    def get_times(start: str, end: str, freq: str) -> list:
        times = (
            (
                pd.date_range(start=start, end="23:59", freq=freq).time.tolist()
                + pd.date_range(start="00:00", end=end, freq=freq).time.tolist()
            )
            if end < start
            else pd.date_range(start=start, end=end, freq=freq).time.tolist()
        )
        return [time.strftime("%I:%M %p") for time in times]

    def __init__(self) -> None:
        self.num_players = 4
        self.elements = {
            "suspect": Clue.suspects[:6],
            "weapon": Clue.weapons[:6],
            "room": Clue.rooms[:6],
            # "motive": Clue.motives[:6],
            # "time": Clue.get_times("21:00", "03:00", "1h"),
        }

    def play(self, print_solver_summary_statistics: bool = False) -> None:
        self.solution = {
            element: random.choice(values) for element, values in self.elements.items()
        }
        deck = list(chain(*self.elements.values()))
        filtered_deck = deck.copy()
        for cards in self.solution.values():
            filtered_deck.remove(cards)
        random.shuffle(filtered_deck)
        self.hands = [
            set(filtered_deck[i :: self.num_players]) for i in range(self.num_players)
        ]
        self.hands.reverse()  # Reverse hands so players with fewer cards go first
        for player, hand in enumerate(self.hands):
            print(f"Player {player + 1}'s Hand: {hand}")
        print(f"Solution: {self.solution}")
        self.history: list[tuple[list[str], dict[int, Optional[str]]]] = []
        self.index = {card: i for i, card in enumerate(deck)}
        ground_truth = np.zeros((len(deck), self.num_players))
        for player, hand in enumerate(self.hands):
            for card in hand:
                ground_truth[self.index[card], player] = 1
        self.print_grid(ground_truth)
        manual_solver = ManualSolver()
        cp_solver = CpSolver(self, max_solve_time_per_turn=0.5)
        turn = 0
        for player in cycle(range(self.num_players)):
            manual_grid = manual_solver.grid(self, player)
            print(f"Player {player + 1}'s Manual Solver Grid:")
            self.print_grid(manual_grid)
            np.testing.assert_array_equal(
                manual_grid[~np.isnan(manual_grid)],
                ground_truth[~np.isnan(manual_grid)],
                err_msg="Non-NaN values in grid do not match ground truth",
            )
            cp_grid = cp_solver.grid(self, player)
            print(f"Player {player + 1}'s CP-SAT Solver Grid:")
            self.print_grid(cp_grid)
            np.testing.assert_array_equal(
                cp_grid[~np.isnan(cp_grid)],
                ground_truth[~np.isnan(cp_grid)],
                err_msg="Non-NaN values in grid do not match ground truth",
            )
            # assert np.array_equal(manual_grid, cp_grid, equal_nan=True), (
            #     f"Manual and CP solvers' grids do not match for player {player + 1} "
            #     f"on turn {turn}"
            # )

            grid = cp_grid

            accusation = len(np.where(grid.sum(axis=1) == 0)[0]) == len(self.elements)
            suggestion: list[str] = []
            start = 0
            for cards in self.elements.values():
                end = start + len(cards)
                if (
                    not accusation
                    and random.random() < 0.4
                    and np.sum(grid[start:end, player]) > 0
                ):
                    suggestion.append(
                        deck[
                            np.random.choice(
                                np.where(grid[start:end, player] == 1)[0] + start
                            )
                        ]
                    )
                else:
                    suggestion.append(
                        deck[
                            np.random.choice(
                                np.where(
                                    (np.sum if accusation else np.nansum)(grid, axis=1)[
                                        start:end
                                    ]
                                    == 0
                                )[0]
                                + start
                            )
                        ]
                    )
                start += len(cards)

            if accusation:
                print(f"Player {player + 1} has an accusation: {suggestion}")
                print(f"The actual solution is: {self.solution}")
                assert all(
                    suggestion[i] == self.solution[element]
                    for i, element in enumerate(self.elements)
                )
                print(f"Player {player + 1} won on turn {turn}!")
                if print_solver_summary_statistics:
                    manual_solver.print_summary_statistics()
                break

            print(f"Player {player + 1} suggests: {suggestion}")

            responses: dict[int, Optional[str]] = {}
            for j in chain(range(player + 1, self.num_players), range(player)):
                responses[j] = None
                suggestion_copy = suggestion.copy()
                random.shuffle(suggestion_copy)
                for card in suggestion_copy:
                    if card in self.hands[j]:
                        responses[j] = card
                        print(f"Player {j + 1} reveals {card} to Player {player + 1}")
                        break
                if responses[j] is not None:
                    break
                else:
                    print(f"Player {j + 1} has no card to reveal")
            self.history.append((suggestion, responses))
            turn += 1

    def print_grid(self, grid: np.ndarray) -> None:
        df = pd.DataFrame(
            grid,
            columns=[f"{i + 1}" for i in range(self.num_players)],
        ).replace({np.nan: "", 0: "✗", 1: "✓"})
        df.index = pd.MultiIndex.from_tuples(
            [
                (element.capitalize(), card)
                for element in self.elements
                for card in self.elements[element]
            ],
            names=["Element", "Card"],
        )
        df.columns.name = "Player"
        print(df)


class ManualSolver:
    def __init__(self) -> None:
        self.all_but = 0
        self.suggestion_elimination = 0
        self.element_deduction = 0
        self.element_inference = 0

    def grid(self, game: Clue, player: int) -> np.ndarray:
        card_count = len(game.index)
        grid = np.full((card_count, game.num_players), np.nan)
        last_grid = grid.copy()

        for card in game.hands[player]:
            grid[game.index[card], player] = 1.0
        for i, (suggestion, responses) in enumerate(game.history):
            i %= game.num_players
            for j, card in responses.items():
                # if player j did not reveal a card
                if card is None:
                    # then all suggested cards must not be in player j's hand
                    for suggested_card in suggestion:
                        grid[game.index[suggested_card], j] = 0.0
                # alternatively if player j revealed a card and we are the player who made the suggestion
                elif player == i:
                    # then we know that player j has the revealed card
                    grid[game.index[card], j] = 1.0

        while not np.array_equal(grid, last_grid, equal_nan=True):
            last_grid = grid.copy()
            for i in range(card_count):
                # if we have deduced that someone has a card
                if np.nansum(grid[i, :]) == 1:
                    # then all other players must not have that card
                    grid[i, np.isnan(grid[i, :])] = 0.0
            for i in range(game.num_players):
                # if we have deduced all the cards in player j's hand
                if np.nansum(grid[:, i]) == len(game.hands[i]):
                    # then all other cards must not be in player j's hand
                    grid[np.isnan(grid[:, i]), i] = 0.0
                # alternatively if we have deduced that all but len(hands[j]) cards are not in player j's hand
                elif (grid[:, i] == 0.0).sum() == card_count - len(game.hands[i]):
                    # then the remaining cards must be in player j's hand
                    grid[np.isnan(grid[:, i]), i] = 1.0
                    self.all_but += 1
            for suggestion, responses in game.history:
                for i, card in responses.items():
                    if (
                        # if player i revealed a card
                        card is not None
                        # and we don't know that any of the suggested cards are in player i's hand
                        and np.nansum(grid[[game.index[c] for c in suggestion], i]) < 1
                        # but we do know that all but one of the suggested cards are not in player i's hand
                        and (grid[[game.index[c] for c in suggestion], i] == 0.0).sum()
                        == len(suggestion) - 1
                    ):
                        # then the one card that we are unsure of must be in player i's hand
                        grid[
                            game.index[
                                next(
                                    c
                                    for c in suggestion
                                    if np.isnan(grid[game.index[c], i])
                                )
                            ],
                            i,
                        ] = 1.0
                        self.suggestion_elimination += 1
            start = 0
            for cards in game.elements.values():
                end = start + len(cards)
                element_grid = grid[start:end]
                if (
                    # If we are unsure of any cells in the grid for a given game element
                    np.isnan(element_grid).sum() > 0
                    # but we know where all but one of the cards are
                    and np.nansum(element_grid) == len(cards) - 1
                ):
                    # then we can fill the remaining unknown cells with 0
                    element_grid[np.isnan(element_grid)] = 0
                    self.element_deduction += 1
                if (
                    # If we know which card is part of the solution
                    np.where(element_grid.sum(axis=0) == 0)[0].size == 1
                    # and there are any cards with one unknown cell
                    and np.where(np.isnan(element_grid).sum(axis=0) == 1)[0].size > 0
                ):
                    # then we can fill the unknown cells with 1
                    solvable_rows = element_grid[
                        np.where(np.isnan(element_grid).sum(axis=0) == 1)
                    ]
                    solvable_rows[np.isnan(solvable_rows)] = 1
                    self.element_inference += 1
                start += len(cards)

        return grid

    def print_summary_statistics(self) -> None:
        print(f"# of all but deductions: {self.all_but}")
        print(f"# of suggestion eliminations: {self.suggestion_elimination}")
        print(f"# of element deductions: {self.element_deduction}")
        print(f"# of element inferences: {self.element_inference}")


class CpSolver:
    def __init__(self, game: Clue, max_solve_time_per_turn: float) -> None:
        self.model = cp_model.CpModel()

        self.vars = [
            [
                self.model.new_bool_var(f"Player {player + 1} has {card}")
                for player in range(game.num_players)
            ]
            for card in game.index
        ]

        # Enforce that each card i is assigned to at most one player
        for i in range(len(game.index)):
            self.model.add(sum(self.vars[i]) <= 1)

        # Enforce that each player has exactly len(hand) cards assigned to them
        for player, hand in enumerate(game.hands):
            self.model.add(sum([row[player] for row in self.vars]) == len(hand))

        # Enforce that there are len(cards) - 1 assignments for each element
        start = 0
        for cards in game.elements.values():
            end = start + len(cards)
            assignments = [var for row in self.vars[start:end] for var in row]
            self.model.add(sum(assignments) == len(cards) - 1)
            start = end

        self.solver = cp_model.CpSolver()
        self.solver.parameters.enumerate_all_solutions = True
        self.solver.parameters.max_time_in_seconds = max_solve_time_per_turn

    def grid(self, game: Clue, player: int) -> np.ndarray:
        # Add constraints for last turn
        suggestion, response = game.history[-1] if len(game.history) > 0 else ([], {})
        for other_player, card in response.items():
            if card is not None:
                # Everyone knows that player i has at least one of the suggested cards
                self.model.add_at_least_one(
                    [self.vars[game.index[card]][other_player] for card in suggestion]
                )
            else:
                # Everyone knows that player i does not have any of the suggested cards
                self.model.add(
                    sum(
                        self.vars[game.index[card]][other_player] for card in suggestion
                    )
                    == 0
                )

        # Add assumptions for the cards in the player's hand
        for card in game.hands[player]:
            self.model.add_assumption(self.vars[game.index[card]][player])

        # Add assumptions for the cards that were revealed to the player in previous turns
        for i, (_, responses) in enumerate(game.history):
            i %= game.num_players
            if player == i:
                for j, card in responses.items():
                    if card is not None:
                        self.model.add_assumption(self.vars[game.index[card]][j])

        callback = SolutionCallback(game, self.vars)
        status = self.solver.solve(self.model, callback)
        assert status == cp_model.OPTIMAL or status == cp_model.FEASIBLE
        self.model.clear_assumptions()  # Remove private knowledge from the model
        grid = callback.grid / callback.num_solutions
        # set every cell that does not equal zero or one to NaN
        grid[(grid != 0) & (grid != 1)] = np.nan
        return grid


class SolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, game: Clue, vars: list[list[cp_model.IntVar]]) -> None:
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.grid = np.zeros((len(game.index), game.num_players))
        self.vars = vars
        self.num_solutions = 0

    def on_solution_callback(self):
        self.grid += np.array([[self.value(var) for var in row] for row in self.vars])
        self.num_solutions += 1
