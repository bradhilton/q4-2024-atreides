from itertools import chain, cycle
import numpy as np
from ortools.sat.python import cp_model
import pandas as pd
import random
from typing import Iterable, Optional, Union

from .data import player_names


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

    prefixes = dict(weapon="the ", room="the ")

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
        cp_solver_max_solve_time_per_turn: float = 0.5,
        check_cp_solver_grid: bool = True,
        check_if_deductive_solver_and_cp_solver_grids_match: bool = True,
        return_first_solver_as_winner: bool = False,
        print_playthrough: bool = True,
        max_turns: int = 1_000,
    ) -> "Clue":
        self.return_first_solver_as_winner = return_first_solver_as_winner
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
        if print_playthrough:
            for player, hand in enumerate(self.hands):
                print(f"Player {player + 1}'s Hand: {hand}")
            print(f"Solution: {self.solution}")
        self.history: list[tuple[list[str], dict[int, Optional[str]]]] = []
        self.index = {card: i for i, card in enumerate(deck)}
        ground_truth = np.zeros((len(deck), self.num_players))
        for player, hand in enumerate(self.hands):
            for card in hand:
                ground_truth[self.index[card], player] = 1
        if print_playthrough:
            self.print_grid(ground_truth)
        deductive_solver = deductive_solver or DeductiveSolver()
        cp_solver = CpSolver(
            self, max_solve_time_per_turn=cp_solver_max_solve_time_per_turn
        )
        self.num_turns = 1
        for player in cycle(range(self.num_players)):
            if self.num_turns > max_turns:
                raise ValueError("Game took too many turns")
            deductive_grid = deductive_solver.grid(self, player)
            if print_playthrough:
                print(f"Player {player + 1}'s Simple Solver Grid:")
                self.print_grid(deductive_grid)
            np.testing.assert_array_equal(
                deductive_grid[~np.isnan(deductive_grid)],
                ground_truth[~np.isnan(deductive_grid)],
                err_msg="Non-NaN values in grid do not match ground truth",
            )
            if (
                print_playthrough
                or check_cp_solver_grid
                or check_if_deductive_solver_and_cp_solver_grids_match
            ):
                cp_grid = cp_solver.grid(self, player)
            if print_playthrough:
                print(f"Player {player + 1}'s CP-SAT Solver Grid:")
                self.print_grid(cp_grid)
            if check_cp_solver_grid:
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

            if self._successful_accusation(deck, grid, print_playthrough, player):
                break

            suggestion: list[str] = []
            start = 0
            for cards in self.elements.values():
                end = start + len(cards)
                subgrid = grid[start:end]
                found_solution = np.nansum(subgrid.sum(axis=1) == 0) > 0
                x = (
                    # solution card
                    (3, subgrid.sum(axis=1) == 0),
                    # possible solution cards
                    (
                        12,
                        np.logical_and(
                            np.nansum(subgrid, axis=1) == 0, not found_solution
                        ),
                    ),
                    # other unknown cards
                    (
                        9,
                        np.logical_and(np.nansum(subgrid, axis=1) == 0, found_solution),
                    ),
                    # player cards
                    (6, grid[start:end, player] == 1),
                    # other player known cards
                    (
                        1,
                        np.logical_and(
                            subgrid.sum(axis=1) == 1, grid[start:end, player] != 1
                        ),
                    ),
                )
                x = ((x[0], x[1].nonzero()[0]) for x in x)
                x = (x for x in x if len(x[1]) > 0)
                p, card_index_sets = zip(*x)
                p = np.array(p) / np.sum(p)
                card_index_set = card_index_sets[
                    np.random.choice(len(card_index_sets), p=p)
                ]
                card = deck[np.random.choice(card_index_set) + start]
                suggestion.append(card)
                start += len(cards)

            if print_playthrough:
                print(f"Player {player + 1} suggests: {suggestion}")

            responses: dict[int, Optional[str]] = {}
            for other_player in chain(
                range(player + 1, self.num_players), range(player)
            ):
                responses[other_player] = None
                suggestion_copy = suggestion.copy()
                random.shuffle(suggestion_copy)
                for card in suggestion_copy:
                    if card in self.hands[other_player]:
                        responses[other_player] = card
                        if print_playthrough:
                            print(
                                f"Player {other_player + 1} reveals {card} to Player {player + 1}"
                            )
                        break
                if responses[other_player] is not None:
                    break
                elif print_playthrough:
                    print(f"Player {other_player + 1} has no card to reveal")
            self.history.append((suggestion, responses))
            if return_first_solver_as_winner:
                found_solution = False
                for player in chain(range(player, self.num_players), range(player)):
                    grid = deductive_solver.grid(self, player)
                    if self._successful_accusation(
                        deck, grid, print_playthrough, player
                    ):
                        found_solution = True
                        break
                if found_solution:
                    break
            elif all(card is None for card in responses.values()):
                grid = deductive_solver.grid(self, player)
                if self._successful_accusation(deck, grid, print_playthrough, player):
                    break
            self.num_turns += 1

        return self

    def _successful_accusation(
        self,
        deck: list[str],
        grid: np.ndarray,
        print_playthrough: bool,
        player: int,
    ) -> bool:
        accusation = [deck[i] for i in np.where(grid.sum(axis=1) == 0)[0]]

        if len(accusation) == len(self.elements):
            if print_playthrough:
                print(f"Player {player + 1} has an accusation: {accusation}")
                print(f"The actual solution is: {self.solution}")
            assert all(
                accusation[i] == self.solution[element]
                for i, element in enumerate(self.elements)
            )
            if print_playthrough:
                print(f"Player {player + 1} won on turn {self.num_turns}!")
            self.winner = player
            return True

        return False

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

    def get_prompt_and_follow_up_and_solution(self) -> tuple[str, str, dict[str, str]]:
        def comma_and_join(items: Iterable[str]) -> str:
            items = list(items)
            return f"{', '.join(items[:-1])}{"," if len(items) > 2 else ""} and {items[-1]}"

        players = player_names.copy()
        random.shuffle(players)

        def turn(
            player: int, suggestion: list[str], responses: dict[int, Optional[str]]
        ) -> str:
            prefixed = {
                card: Clue.prefixes.get(element, "") + card
                for element, card in zip(self.elements, suggestion)
            }
            return f"{players[player]} asked if anyone had {' or '.join(prefixed[card] for card in suggestion)}:\n{'\n'.join(f'- {players[responding_player]} showed {players[player]} {prefixed[card] if player == self.winner or responding_player == self.winner else 'a card'}' if card is not None else f'- {players[responding_player]} did not have any of the cards' for responding_player, card in responses.items())}"

        prompt = f"""
        On a {random.choice(("cool", "warm"))} {random.choice(("spring", "autumn"))} {random.choice(("morning", "afternoon", "evening", "day"))} {comma_and_join(players[:self.num_players])} sat down to play a {random.choice(("friendly", "competitive", "casual"))} {random.choice(("mystery", "deduction", "sleuthing"))} game.

        They {random.choice(("assembled", "gathered"))} {len(self.elements)} {random.choice(("decks", "groups", "stacks"))} of cards, each for a {random.choice(("different", "separate"))} {random.choice(("category", "type"))} of {random.choice(("information", "data"))} composed of the following:

        {"\n\n".join(f"{category.capitalize()}:\n{'\n'.join(f'- ' + card for card in cards)}" for category, cards in self.elements.items())}

        After randomly (and blindly) choosing one card from each {random.choice(("deck", "group", "stack"))} and placing them in the {random.choice(("center", "middle"))} of the table facedown, they shuffled the remaining cards and dealt out the following to each player:

        {"\n".join(f'- {players[player]}: {len(self.hands[player])} cards' + (f' ({comma_and_join(self.hands[player])})' if player == self.winner else "") for player in range(self.num_players))}

        The game proceeded as follows:

        1. On their turn, a player asked about a set of exactly {len(self.elements)} cards, one from each of the game's categories. (Note: Players could ask about any cards, including those in their own hand.)
        2. The player directed this question to the other players in clockwise order, starting with the player to their left.
        3. If a player had one or more of the asked-about cards, they had to show one of those cards (of their choice) to the asking player privately. The turn then ended, and play passed to the next player.
        4. If a player did not have any of the asked-about cards, they said so, and the question passed to the next player in clockwise order.
        5. This continued until either:
            a) A player showed a card to the asking player, or
            b) All the queried players had stated they didn't have any of the asked-about cards.
        6. After a player's turn ended (either by being shown a card or having all queried players pass), play moved to the next player in clockwise order.

        Here is how the game played out:

        {"\n\n".join(turn(player % self.num_players, suggestion, responses) for player, (suggestion, responses) in enumerate(self.history))}

        At this point {players[self.winner]} was able to correctly {random.choice(("deduce", "infer"))} the {random.choice(("solution", "answer"))}{"" if self.return_first_solver_as_winner else " and win the game"}.

        What were the facedown cards in the {random.choice(("center", "middle"))} of the table?{" And where were the other cards?" if self.return_first_solver_as_winner else ""}
        """.strip().replace(
            "    ", ""
        )

        if self.return_first_solver_as_winner:
            follow_up = (
                "Fill out your answer like this:\n"
                + "\n".join(
                    f"{card}: <#LOCATION#>"
                    for cards in self.elements.values()
                    for card in cards
                )
                + f"\nWhere valid locations are {', '.join(players[:self.num_players])}, or Solution."
            )
        else:
            follow_up = "Fill out your answer like this:\n" + "\n".join(
                f"{element.capitalize()}: <#{element.upper()}#>"
                for element in self.elements
            )

        if self.return_first_solver_as_winner:
            solution = {
                card: location
                for card, location in chain(
                    zip(self.solution.values(), ["Solution"] * len(self.solution)),
                    (
                        (card, players[i])
                        for i, hand in enumerate(self.hands)
                        for card in hand
                    ),
                )
            }
            solution = {
                card: solution[card]
                for cards in self.elements.values()
                for card in cards
            }
        else:
            solution = self.solution

        return prompt, follow_up, solution


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
            # Fill in the grid with the player's cards
            for card in game.hands[player]:
                grid[game.index[card], player] = 1

        if self.check_unique_card_placement_constraints:
            # Add a constraint for each card that it may only be in one place
            for i in range(len(game.index)):
                constraints.append((grid[i], 1))

        if self.check_player_hand_size_constraints:
            # Add a constraint for each player that they should have len(hand) cards
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

        # Fill in the grid with the known cards and add constraints from previous turns
        for turn, (suggestion, responses) in enumerate(game.history):
            suggesting_player = turn % game.num_players
            for responding_player, card in responses.items():
                if card is not None:
                    if player == suggesting_player:
                        if self.note_responses_to_suggestions:
                            grid[game.index[card], responding_player] = 1
                    elif player != responding_player:
                        self.add_one_of_constraints(
                            game=game,
                            responding_player=responding_player,
                            grid=grid,
                            constraints=constraints,
                            one_of=one_of,
                            suggestion=suggestion,
                        )
                elif self.note_cards_that_players_do_not_have:
                    for card in suggestion:
                        grid[game.index[card], responding_player] = 0

        self.fill_grid(grid, constraints)

        if self.exhaustively_test_possible_assignments:
            # Iteratively test all possible assignments for unknown cards
            # If an assignment violates a constraint, then we can safely make the opposite assignment
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

    def add_one_of_constraints(
        self,
        game: Clue,
        responding_player: int,
        grid: np.ndarray,
        constraints: list[Constraint],
        one_of: dict[int, list[set[int]]],
        suggestion: list[str],
    ) -> None:
        """
        Add constraints when a player privately reveals a card to another player.

        This method adds two types of constraints:
        1. One-of constraint: At least one of the suggested cards is in the responding player's hand.
        2. Inverse one-of constraint: No more than a certain number of unsuggested cards are in the responding player's hand.

        It also merges disjoint inverse one-of constraints if applicable.
        """
        # Get the indices of the suggested cards
        indices = [game.index[c] for c in suggestion]
        # At least one of the suggested cards is in the responding player's hand
        if self.check_one_of_constraints:
            constraints.append(
                (
                    tuple(grid[i : i + 1, responding_player] for i in indices),
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
                        for i, j in zip([-1] + indices, indices + [len(game.index)])
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
                            for i, j in zip([-1] + union, union + [len(game.index)])
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

    def fill_grid(self, grid: np.ndarray, constraints: list[Constraint]) -> None:
        """
        Iteratively check constraints and fill the grid where possible until convergence.
        """
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
