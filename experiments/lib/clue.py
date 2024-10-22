from itertools import chain, cycle
import numpy as np
from ortools.sat.python import cp_model
import pandas as pd
import random
import sympy as sp
from typing import Optional, Union


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

    def __init__(
        self,
        num_players: int = 3,
        elements: Optional[dict[str, list[str]]] = None,
    ) -> None:
        self.num_players = num_players
        self.elements = elements or {
            "suspect": Clue.suspects[:3],
            "weapon": Clue.weapons[:3],
            "room": Clue.rooms[:3],
            # "motive": Clue.motives[:6],
            # "time": Clue.get_times("21:00", "03:00", "1h"),
        }

    def play(
        self,
        deductive_solver: Optional["DeductiveSolver"] = None,
        check_if_deductive_solver_and_cp_solver_grids_match: bool = True,
        suggest_own_card_probability: float = 0.4,
    ) -> None:
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
        deductive_solver = deductive_solver or DeductiveSolver()
        cp_solver = CpSolver(self, max_solve_time_per_turn=0.5)
        self.num_turns = 1
        for player in cycle(range(self.num_players)):
            deductive_grid = deductive_solver.grid(self, player)
            print(f"Player {player + 1}'s Simple Solver Grid:")
            self.print_grid(deductive_grid)
            np.testing.assert_array_equal(
                deductive_grid[~np.isnan(deductive_grid)],
                ground_truth[~np.isnan(deductive_grid)],
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

            if check_if_deductive_solver_and_cp_solver_grids_match:
                assert np.array_equal(
                    deductive_grid, cp_grid, equal_nan=True
                ), "Deductive Solver and CP-SAT Solver grids do not match"

            grid = deductive_grid

            accusation = len(np.where(grid.sum(axis=1) == 0)[0]) == len(self.elements)
            suggestion: list[str] = []
            start = 0
            for cards in self.elements.values():
                end = start + len(cards)
                if (
                    not accusation
                    and random.random() < suggest_own_card_probability
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
                print(f"Player {player + 1} won on turn {self.num_turns}!")
                self.winner = player
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
            self.num_turns += 1

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


Constraint = tuple[
    Union[np.ndarray, tuple[np.ndarray, ...]],
    Union[int, tuple[Optional[int], Optional[int]]],
]


class DeductiveSolver:
    def __init__(
        self,
        note_cards_in_hand: bool = True,
        note_responses_to_suggestions: bool = True,
        note_cards_that_players_do_not_have: bool = True,
        check_unique_card_placement_constraints: bool = True,
        check_player_hand_size_constraints: bool = True,
        check_solution_has_one_and_only_one_card_per_element: bool = True,
        check_one_of_constraints: bool = True,
        check_inverse_one_of_constraints: bool = True,
        merge_and_check_disjoint_inverse_one_of_constraints: bool = True,
        exhaustively_test_possible_assignments: bool = True,
    ) -> None:
        self.note_cards_in_hand = note_cards_in_hand
        self.note_responses_to_suggestions = note_responses_to_suggestions
        self.note_cards_that_players_do_not_have = note_cards_that_players_do_not_have
        self.check_unique_card_placement_constraints = (
            check_unique_card_placement_constraints
        )
        self.check_player_hand_size_constraints = check_player_hand_size_constraints
        self.check_solution_has_one_and_only_one_card_per_element = (
            check_solution_has_one_and_only_one_card_per_element
        )
        self.check_one_of_constraints = check_one_of_constraints
        self.check_inverse_one_of_constraints = check_inverse_one_of_constraints
        self.merge_and_check_disjoint_inverse_one_of_constraints = (
            merge_and_check_disjoint_inverse_one_of_constraints
        )
        self.exhaustively_test_possible_assignments = (
            exhaustively_test_possible_assignments
        )

    def grid(self, game: Clue, player: int) -> np.ndarray:
        grid = np.full((len(game.index), game.num_players + 1), np.nan)
        constraints: list[Constraint] = []

        if self.note_cards_in_hand:
            # Fill in the grid with the known cards
            for card in game.hands[player]:
                grid[game.index[card], player] = 1

        if self.check_unique_card_placement_constraints:
            # Each card may only be in one place
            for i in range(len(game.index)):
                constraints.append((grid[i], 1))

        if self.check_player_hand_size_constraints:
            # Each player has game.hands[i] cards
            for i, hand in enumerate(game.hands):
                constraints.append((grid[:, i], len(hand)))

        if self.check_solution_has_one_and_only_one_card_per_element:
            # The solution must have exactly one card from each game element
            start = 0
            for cards in game.elements.values():
                end = start + len(cards)
                constraints.append((grid[start:end, -1], 1))
                start = end

        one_of: dict[int, list[set[int]]] = {i: [] for i in range(game.num_players)}

        # Fill in the grid with the known cards from previous turns
        for suggesting_player, (suggestion, responses) in enumerate(game.history):
            suggesting_player %= game.num_players
            for responding_player, card in responses.items():
                if card is not None:
                    if player == suggesting_player:
                        if self.note_responses_to_suggestions:
                            grid[game.index[card], responding_player] = 1
                    elif player != responding_player:
                        indices = [game.index[c] for c in suggestion]
                        # At least one of the suggested cards is in the responding player's hand
                        if self.check_one_of_constraints:
                            constraints.append(
                                (
                                    tuple(
                                        grid[i : i + 1, responding_player]
                                        for i in indices
                                    ),
                                    (1, len(suggestion)),
                                )
                            )
                        if self.check_inverse_one_of_constraints:
                            # And no more than len(game.hands[responding_player]) - 1 of the
                            # unsuggested cards are in the responding player's hand
                            constraints.append(
                                (
                                    tuple(
                                        grid[i + 1 : j, responding_player]
                                        for i, j in zip(
                                            [-1] + indices, indices + [len(game.index)]
                                        )
                                    ),
                                    (0, len(game.hands[responding_player]) - 1),
                                )
                            )
                        for previous_indices in one_of[responding_player]:
                            if (
                                previous_indices.isdisjoint(indices)
                                and self.merge_and_check_disjoint_inverse_one_of_constraints
                            ):
                                union = sorted(previous_indices.union(indices))
                                constraints.append(
                                    (
                                        tuple(
                                            grid[i + 1 : j, responding_player]
                                            for i, j in zip(
                                                [-1] + union, union + [len(game.index)]
                                            )
                                        ),
                                        (
                                            0,
                                            len(game.hands[responding_player])
                                            - len(union) // len(indices),
                                        ),
                                    )
                                )
                                one_of[responding_player].append(set(union))
                        one_of[responding_player].append(set(indices))
                elif self.note_cards_that_players_do_not_have:
                    for card in suggestion:
                        grid[game.index[card], responding_player] = 0

        self.fill_grid(grid, constraints)

        if self.exhaustively_test_possible_assignments:
            last_grid = np.full_like(grid, np.nan)
            while not np.array_equal(grid, last_grid, equal_nan=True):
                last_grid = grid.copy()
                for indices in np.argwhere(np.isnan(grid)):
                    for assignment in (0, 1):
                        original_grid = grid.copy()
                        grid[indices[0], indices[1]] = assignment
                        try:
                            self.fill_grid(grid, constraints)
                            grid[:] = original_grid[:]
                        except ValueError:
                            grid[:] = original_grid[:]
                            grid[indices[0], indices[1]] = 1 - assignment
                            self.fill_grid(grid, constraints)

        return grid[:, :-1]

    def fill_grid(self, grid: np.ndarray, constraints: list[Constraint]) -> None:
        last_grid = np.full_like(grid, np.nan)
        while not np.array_equal(grid, last_grid, equal_nan=True):
            last_grid = grid.copy()
            for views, bounds in constraints:
                if not isinstance(views, tuple):
                    views = (views,)
                if isinstance(bounds, int):
                    lower_bound = upper_bound = bounds
                else:
                    lower_bound, upper_bound = bounds
                values = np.concatenate([view.flatten() for view in views])
                if np.sum(np.nan_to_num(values, nan=1)) < lower_bound:
                    raise ValueError(
                        f"Impossible to have at least {lower_bound} cards in constraint"
                    )
                elif np.sum(np.nan_to_num(values, nan=1)) == lower_bound:
                    for view in views:
                        view[np.isnan(view)] = 1
                if np.nansum(values) == upper_bound:
                    for view in views:
                        view[np.isnan(view)] = 0
                elif np.nansum(values) > upper_bound:
                    raise ValueError(
                        f"Impossible to have at most {upper_bound} cards in constraint"
                    )


class CpSolver:
    def __init__(self, game: Clue, max_solve_time_per_turn: float) -> None:
        self.model = cp_model.CpModel()

        self.vars = np.array(
            [
                [
                    self.model.new_bool_var(f"Player {player + 1} has '{card}'")
                    for player in range(game.num_players)
                ]
                for card in game.index
            ]
        )

        # Enforce that each card i is assigned to at most one player
        for i in range(len(game.index)):
            self.model.add(sum(self.vars[i]) <= 1)

        # Enforce that each player has exactly len(hand) cards assigned to them
        for player, hand in enumerate(game.hands):
            self.model.add(sum(self.vars[:, player]) == len(hand))

        # Enforce that there are len(cards) - 1 assignments for each element
        for cards in game.elements.values():
            self.model.add(
                sum(self.vars[[game.index[card] for card in cards]].flatten())
                == len(cards) - 1
            )

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
                    self.vars[[game.index[c] for c in suggestion], other_player]
                )
            else:
                # Everyone knows that player i does not have any of the suggested cards
                self.model.add(
                    sum(self.vars[[game.index[c] for c in suggestion], other_player])
                    == 0
                )

        # Add assumptions for the cards in the player's hand
        for card in game.hands[player]:
            self.model.add_assumption(self.vars[game.index[card], player])

        # Add assumptions for the cards that were revealed to the player in previous turns
        for i, (_, responses) in enumerate(game.history):
            if player == i % game.num_players:
                for j, card in responses.items():
                    if card is not None:
                        self.model.add_assumption(self.vars[game.index[card], j])

        callback = SolutionCallback(game, self.vars)
        status = self.solver.solve(self.model, callback)
        assert status == cp_model.OPTIMAL or status == cp_model.FEASIBLE
        self.model.clear_assumptions()  # Remove private knowledge from the model
        grid = callback.grid / callback.num_solutions
        # set every cell that does not equal zero or one to NaN
        grid[(grid != 0) & (grid != 1)] = np.nan
        return grid


class SolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, game: Clue, vars: np.ndarray) -> None:
        super().__init__()
        self.grid = np.zeros((len(game.index), game.num_players))
        self.vars = vars
        self.num_solutions = 0

    def on_solution_callback(self) -> None:
        self.grid += np.vectorize(self.value)(self.vars)
        self.num_solutions += 1
