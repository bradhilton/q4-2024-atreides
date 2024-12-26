import itertools
import black
from dataclasses import dataclass
import math
from ortools.sat.python import cp_model
import random
import re
from typing import Callable, Hashable, Iterable, Literal, NewType, TypeVar, Union
from PIL import Image, ImageDraw, ImageFont  # type: ignore
import os

from .clue import Clue
from .utils import retry

# Type definitions
Suspect = NewType("Suspect", str)
Weapon = NewType("Weapon", str)
Room = NewType("Room", str)
Time = NewType("Time", str)
Motive = NewType("Motive", str)
MrBoddy = Literal["Mr. Boddy"]
Character = Union[Suspect, MrBoddy]
Piece = Union[Character, Weapon]
Element = Union[Piece, Room, Time]
Answer = Union[Suspect, Weapon, Room, Time, Motive]

H = TypeVar("H", bound=Hashable)
T = TypeVar("T")


def shuffled(iterable: Iterable[T]) -> list[T]:
    """Returns a shuffled list of the input iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items


class TemporalClueCpModel(cp_model.CpModel):
    """CP-SAT model with helper methods for Temporal Clue puzzle generation."""

    def __init__(self, scenario: "TemporalClueScenario") -> None:
        super().__init__()
        self._all_vars: dict[frozenset[cp_model.IntVar], cp_model.IntVar] = {}
        self._any_vars: dict[frozenset[cp_model.IntVar], cp_model.IntVar] = {}
        self._sum_vars: dict[
            tuple[frozenset[cp_model.IntVar], int], cp_model.IntVar
        ] = {}
        self._variable_element_time_room_vars: dict[
            tuple[
                dict[Suspect | Weapon, dict[Time, dict[Room, cp_model.IntVar]]],
                dict[Suspect | Weapon, cp_model.IntVar],
                Time,
                Room,
            ],
            cp_model.IntVar,
        ] = {}
        self.scenario = scenario

        # Create variables for suspect locations at each time
        self.suspect_time_room_vars = {
            suspect: {
                time: self.categorical_vars(scenario.rooms, prefix=f"{suspect} {time} ")
                for time in scenario.times
            }
            for suspect in scenario.suspects
        }

        # Create variables for suspect motives
        self.suspect_motive_vars = {
            suspect: self.categorical_vars(
                scenario.motives, prefix=f"{suspect} motive "
            )
            for suspect in scenario.suspects
        }
        self.motive_suspect_vars = {
            motive: {
                suspect: self.suspect_motive_vars[suspect][motive]
                for suspect in scenario.suspects
            }
            for motive in scenario.motives
        }
        if scenario.unique_motives:
            for motive in scenario.motives:
                self.add(
                    sum(
                        self.suspect_motive_vars[suspect][motive]
                        for suspect in scenario.suspects
                    )
                    == 1
                )

        # Create variables for weapon locations at each time
        self.weapon_time_room_vars = {
            weapon: {
                time: self.categorical_vars(scenario.rooms, prefix=f"{weapon} {time} ")
                for time in scenario.times
            }
            for weapon in scenario.weapons
        }

        # Create variables for Mr. Boddy's time rooms
        self.mr_boddy_time_room_vars = {
            time: self.categorical_vars(scenario.rooms, prefix=f"Mr. Boddy {time} ")
            for time in scenario.times
        }

        # Variables for character locations at each time
        self.character_time_room_vars = {
            **self.suspect_time_room_vars,
            "Mr. Boddy": self.mr_boddy_time_room_vars,
        }

        # Variables for piece locations at each time
        self.piece_time_room_vars = {
            **self.suspect_time_room_vars,
            **self.weapon_time_room_vars,
            "Mr. Boddy": self.mr_boddy_time_room_vars,
        }
        self.piece_room_time_vars = {
            piece: {
                room: {
                    time: self.piece_time_room_vars[piece][time][room]
                    for time in scenario.times
                }
                for room in scenario.rooms
            }
            for piece in self.piece_time_room_vars
        }

        # Create variables for murder solution
        self.murderer_vars = self.categorical_vars(
            scenario.suspects, suffix=" murderer"
        )
        self.murder_weapon_vars = self.categorical_vars(
            scenario.weapons, suffix=" murder weapon"
        )
        self.murder_room_vars = self.categorical_vars(
            scenario.rooms, suffix=" murder room"
        )
        self.murder_time_vars = self.categorical_vars(
            scenario.times, suffix=" murder time"
        )

        self._add_constraints()

    def _add_constraints(self) -> None:
        """Adds all constraints to the model."""
        # Murder constraints
        for room, room_var in self.murder_room_vars.items():
            for time, time_var in self.murder_time_vars.items():
                # If this is the murder room and time, then only one suspect can be in this room at this time
                self.add(
                    sum(
                        self.suspect_time_room_vars[s][time][room]
                        for s in self.scenario.suspects
                    )
                    == 1
                ).only_enforce_if(room_var, time_var)
        for suspect, suspect_var in self.murderer_vars.items():
            for weapon, weapon_var in self.murder_weapon_vars.items():
                for room, room_var in self.murder_room_vars.items():
                    for time, time_var in self.murder_time_vars.items():
                        # If this is the murder suspect, weapon, room, and time, then the suspect and weapon must be in this room at this time
                        self.add(
                            self.suspect_time_room_vars[suspect][time][room]
                            + self.weapon_time_room_vars[weapon][time][room]
                            == 2
                        ).only_enforce_if(suspect_var, weapon_var, room_var, time_var)

        # Mr. Boddy must be in the murder room at and following the murder time
        accumulated_murder_time_vars = []
        for time, time_room_vars in self.mr_boddy_time_room_vars.items():
            accumulated_murder_time_vars.append(self.murder_time_vars[time])
            for room, room_var in time_room_vars.items():
                self.add(room_var == 1).only_enforce_if(
                    self.murder_room_vars[room],
                    self.any(*accumulated_murder_time_vars),
                )

        # Characters can only move to adjacent rooms
        for time_room_vars in self.character_time_room_vars.values():
            prev_room_vars = None
            for room_vars in time_room_vars.values():
                if prev_room_vars:
                    for prev_room, prev_room_var in prev_room_vars.items():
                        for room, room_var in room_vars.items():
                            if prev_room != room:
                                if not any(
                                    self.scenario.room_coords[prev_room][0]
                                    == self.scenario.room_coords[room][0] + dx
                                    and self.scenario.room_coords[prev_room][1]
                                    == self.scenario.room_coords[room][1] + dy
                                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                ):
                                    # If prev_room and room are not orthogonally adjacent, then a move cannot be made
                                    self.add_at_most_one(prev_room_var, room_var)
                prev_room_vars = room_vars

        # Weapons can only move with suspects
        for time_room_vars in self.weapon_time_room_vars.values():
            prev_time = None
            prev_room_vars = None
            for time, room_vars in time_room_vars.items():
                if prev_time is not None and prev_room_vars is not None:
                    for prev_room, prev_room_var in prev_room_vars.items():
                        for room, room_var in room_vars.items():
                            if prev_room != room:
                                # For the weapon to move, at least one suspect has to have made the same move at the same time
                                self.add_at_least_one(
                                    self.all(
                                        self.suspect_time_room_vars[s][prev_time][
                                            prev_room
                                        ],
                                        self.suspect_time_room_vars[s][time][room],
                                    )
                                    for s in self.scenario.suspects
                                ).only_enforce_if(prev_room_var, room_var)
                prev_time = time
                prev_room_vars = room_vars

    def all(self, *bool_vars: cp_model.IntVar) -> cp_model.IntVar:
        """Returns a variable that is true iff all input variables are true."""
        var_set = frozenset(bool_vars)
        if var_set in self._all_vars:
            return self._all_vars[var_set]
        var = self.new_bool_var(" and ".join(str(var) for var in var_set))
        self.add(sum(var_set) == len(var_set)).only_enforce_if(var)
        self.add(sum(var_set) < len(var_set)).only_enforce_if(~var)
        self._all_vars[var_set] = var
        return var

    def any(self, *bool_vars: cp_model.IntVar) -> cp_model.IntVar:
        """Returns a variable that is true iff any input variables are true."""
        var_set = frozenset(bool_vars)
        if var_set in self._any_vars:
            return self._any_vars[var_set]
        var = self.new_bool_var(" or ".join(str(var) for var in var_set))
        self.add(sum(var_set) >= 1).only_enforce_if(var)
        self.add(sum(var_set) < 1).only_enforce_if(~var)
        self._any_vars[var_set] = var
        return var

    def sum(self, *bool_vars: cp_model.IntVar, value: int) -> cp_model.IntVar:
        """Returns a variable that is true iff the sum of input variables equals value."""
        var_set = frozenset(bool_vars)
        if (var_set, value) in self._sum_vars:
            return self._sum_vars[(var_set, value)]
        var = self.new_bool_var("sum(" + ", ".join(str(var) for var in var_set) + ")")
        self.add(sum(var_set) == value).only_enforce_if(var)
        self.add(sum(var_set) != value).only_enforce_if(~var)
        self._sum_vars[(var_set, value)] = var
        return var

    def categorical_vars(
        self,
        values: list[H],
        *,
        prefix: str = "",
        suffix: str = "",
    ) -> dict[H, cp_model.IntVar]:
        """Creates variables for mutually exclusive categories.

        Args:
            values: List of possible values
            prefix: Optional prefix for variable names
            suffix: Optional suffix for variable names

        Returns:
            Dictionary mapping values to their corresponding variables
        """
        vars = {
            value: self.new_bool_var(f"{prefix}{value}{suffix}") for value in values
        }
        self.add_exactly_one(vars.values())
        return vars

    def categorical_any_equal(
        self, lhs: dict[H, cp_model.IntVar], rhs: dict[H, cp_model.IntVar]
    ) -> cp_model.IntVar:
        return self.any(*(self.all(lhs[key], rhs[key]) for key in lhs))

    def categorical_exactly_one_equal(
        self,
        lhs: dict[H, cp_model.IntVar],
        rhs: dict[H, cp_model.IntVar],
    ) -> cp_model.IntVar:
        return self.sum(*(self.all(lhs[key], rhs[key]) for key in lhs), value=1)

    def motive_time_room_var(
        self, motive: Motive, time: Time, room: Room
    ) -> cp_model.IntVar:
        return self._variable_element_time_room_var(
            self.suspect_time_room_vars,
            self.motive_suspect_vars[motive],
            f"motivated by {motive}",
            time,
            room,
            unique=self.scenario.unique_motives,
        )

    def motive_room_vars(
        self, motive: Motive, time: Time
    ) -> dict[Room, cp_model.IntVar]:
        return {
            room: self.motive_time_room_var(motive, time, room)
            for room in self.scenario.rooms
        }

    def motive_time_vars(
        self, motive: Motive, room: Room
    ) -> dict[Time, cp_model.IntVar]:
        return {
            time: self.motive_time_room_var(motive, time, room)
            for time in self.scenario.times
        }

    def murderer_time_room_var(self, time: Time, room: Room) -> cp_model.IntVar:
        return self._variable_element_time_room_var(
            self.suspect_time_room_vars, self.murderer_vars, "the murderer", time, room
        )

    def murderer_time_vars(self, room: Room) -> dict[Time, cp_model.IntVar]:
        return {
            time: self.murderer_time_room_var(time, room)
            for time in self.scenario.times
        }

    def murderer_room_vars(self, time: Time) -> dict[Room, cp_model.IntVar]:
        return {
            room: self.murderer_time_room_var(time, room)
            for room in self.scenario.rooms
        }

    def murder_weapon_time_room_var(self, time: Time, room: Room) -> cp_model.IntVar:
        return self._variable_element_time_room_var(
            self.weapon_time_room_vars,
            self.murder_weapon_vars,
            "the murder weapon",
            time,
            room,
        )

    def murder_weapon_time_vars(self, room: Room) -> dict[Time, cp_model.IntVar]:
        return {
            time: self.murder_weapon_time_room_var(time, room)
            for time in self.scenario.times
        }

    def murder_weapon_room_vars(self, time: Time) -> dict[Room, cp_model.IntVar]:
        return {
            room: self.murder_weapon_time_room_var(time, room)
            for room in self.scenario.rooms
        }

    def _variable_element_time_room_var(
        self,
        element_time_room_vars: dict[T, dict[Time, dict[Room, cp_model.IntVar]]],
        variable_element_vars: dict[T, cp_model.IntVar],
        variable: str,
        time: Time,
        room: Room,
        unique: bool = True,
    ) -> cp_model.IntVar:
        key = (
            frozenset(element_time_room_vars.keys()),
            frozenset(variable_element_vars.keys()),
            variable,
            time,
            room,
        )
        if key in self._variable_element_time_room_vars:
            return self._variable_element_time_room_vars[key]
        vars = []
        for (
            element,
            time_room_vars,
        ) in element_time_room_vars.items():
            time_room_var = time_room_vars[time][room]
            variable_var = variable_element_vars[element]
            var = self.new_bool_var(
                f"If {element} is {variable}, then they were in the {room} at {time}"
            )
            self.add(var == 1).only_enforce_if(variable_var, time_room_var)
            self.add(var == 0).only_enforce_if(~self.all(variable_var, time_room_var))
            vars.append(var)
        var = self.sum(*vars, value=1) if unique else self.any(*vars)
        self._variable_element_time_room_vars[key] = var  # type: ignore
        return var


@dataclass
class CharacterMove:
    character: Character
    time: Time
    from_room: Room
    to_room: Room


@dataclass
class WeaponMove:
    weapon: Weapon
    suspect: Suspect
    time: Time
    from_room: Room
    to_room: Room


@dataclass
class TemporalClue:
    description: str
    get_bool_var: Callable[[TemporalClueCpModel], cp_model.IntVar]
    bias: float = 0.0
    question_answer: tuple[str, Answer] | None = None


prompt_template = """
On a dark winter night, wealthy and enigmatic Mr. John Q. Boddy hosted a small, but lavish, dinner party for some of his closest associates. However, the night ended in tragedy when Mr. Boddy was found dead in one of the rooms of Tudor Mansion in the early hours of the morning. The following persons of interest have been identified as suspects:

{suspects}

And the following weapons were found on the premises:

{weapons}

The murder could only have occured in one of the following rooms:

{rooms}

The rooms are laid out as follows:

{board}

The exact time of the murder is a bit uncertain, but it has been narrowed down to one of the following times:

{times}

At every time the suspects and Mr. Boddy either stayed in their current room or moved to an orthogonally adjacent room (north, south, east, or west). Weapons could be moved by suspects between rooms as well.

Each suspect {uniquely}had one of the following possible motives for killing Mr. Boddy:

{motives}

For the murder to occur, the murderer and Mr. Boddy must have been alone in a room with at least one weapon at some point in the night. Any clue about Mr. Boddy's whereabouts should be read as "Mr. Boddy (dead or alive) ..."

The available clues are as follows:

{clues}

Please answer the following question(s):

{questions}

And the following bonus question(s):

{bonus_questions}

Fill out your final answers in the following format:

{format}

Best of luck, detective.
""".strip()


@dataclass
class TemporalCluePuzzle:
    scenario: "TemporalClueScenario"
    clues: list[str]
    questions: dict[str, Answer]
    bonus_questions: dict[str, Answer]

    @property
    def all_questions(self) -> dict[str, Answer]:
        return {**self.questions, **self.bonus_questions}

    def answer_type(self, answer: Answer) -> str:
        if answer in self.scenario.suspects:
            return "SUSPECT"
        elif answer in self.scenario.weapons:
            return "WEAPON"
        elif answer in self.scenario.rooms:
            return "ROOM"
        elif answer in self.scenario.times:
            return "TIME"
        elif answer in self.scenario.motives:
            return "MOTIVE"
        else:
            return "ANSWER"

    def prompt(self) -> str:
        prompt = prompt_template.format(
            suspects="\n".join(f"• {suspect}" for suspect in self.scenario.suspects),
            weapons="\n".join(f"• {weapon}" for weapon in self.scenario.weapons),
            rooms="\n".join(
                f"{i}. {room}" for i, room in enumerate(self.scenario.rooms, start=1)
            ),
            board=self.scenario.formatted_board(),
            times="\n".join(f"• {time}" for time in self.scenario.times),
            uniquely="uniquely " if self.scenario.unique_motives else "",
            motives="\n".join(f"• {motive}" for motive in self.scenario.motives),
            clues="\n".join(f"- {clue}" for clue in self.clues),
            questions="\n".join(
                [
                    f"{chr(65+i)}. {question}"
                    for i, question in enumerate(self.questions.keys())
                ]
            ),
            bonus_questions="\n".join(
                [
                    f"{chr(65+i)}. {question}"
                    for i, question in enumerate(
                        self.bonus_questions.keys(), start=len(self.questions)
                    )
                ]
            ),
            format="\n".join(
                [
                    f"{chr(65+i)}. {self.answer_type(answer)}"
                    for i, answer in enumerate(self.all_questions.values())
                ]
            ),
        )
        if len(self.scenario.times) < 2:
            prompt = re.sub(
                r"\nThe exact time of the murder.*by suspects between rooms as well.\n",
                "",
                prompt,
                flags=re.DOTALL,
            )
            prompt = re.sub(
                r" at some point in the night\. Any clue about Mr\. Boddy's whereabouts should be read as \"Mr\. Boddy \(dead or alive\) \.\.\.\"",
                ".",
                prompt,
                flags=re.DOTALL,
            )
            prompt = re.sub(
                " at least once",
                "",
                prompt,
                flags=re.DOTALL,
            )
            prompt = re.sub(
                " at 12:00 AM",
                "",
                prompt,
                flags=re.DOTALL,
            )
        if len(self.scenario.motives) < 2:
            prompt = re.sub(
                r"Each suspect had one of the following possible motives for killing Mr\. Boddy:.*For the murder to occur",
                "For the murder to occur",
                prompt,
                flags=re.DOTALL,
            )
        return prompt


TemporalCluePuzzle__repr__ = TemporalCluePuzzle.__repr__


def TemporalCluePuzzle__repr__hook(self) -> str:
    return black.format_str(TemporalCluePuzzle__repr__(self), mode=black.Mode())


TemporalCluePuzzle.__repr__ = TemporalCluePuzzle__repr__hook


class NoValidMurderSolutionError(ValueError):
    """Raised when no valid murder solution can be found."""

    def __init__(
        self,
        message: str = "Failed to find a valid murder solution",
        attempts: int = 10,
    ) -> None:
        super().__init__(f"{message} in {attempts} attempts")


class SolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self) -> None:
        super().__init__()
        self.num_solutions = 0

    def on_solution_callback(self) -> None:
        self.num_solutions += 1
        if self.num_solutions > 1:
            self.stop_search()


class TemporalClueScenario:

    @retry(max_attempts=10, delay=0, exceptions=(NoValidMurderSolutionError,))
    def __init__(
        self,
        *,
        min_players: int = 3,
        max_players: int = 6,
        max_suspects: int | None = None,
        max_weapons: int | None = None,
        max_rooms: int | None = None,
        max_times: int | None = None,
        max_motives: int | None = None,
        character_move_probability: float = 0.5,
        suspect_moves_weapon_probability: float = 0.8,
        unique_motives: bool | float = 0.5,
    ):
        """Initialize a new Temporal Clue game.

        Args:
            min_players: Minimum number of players (default: 3)
            max_players: Maximum number of players (default: 6)
            max_suspects: Maximum number of suspects (default: len(Clue.suspects))
            max_weapons: Maximum number of weapons (default: len(Clue.weapons))
            max_rooms: Maximum number of rooms (default: len(Clue.rooms))
            max_times: Maximum number of time periods (default: None)
            character_move_probability: Probability a character moves each time period (default: 0.5)
            suspect_moves_weapon_probability: Probability a suspect moves the weapon they are holding (default: 0.8)
            unique_motives: If True, each suspect has a unique motive. If False, multiple suspects may share the
                same motive. If a float [0, 1], whether motives are unique is randomly determined with this probability.
                Defaults to 0.5, i.e. 50% of the time motives are unique.
        """
        self.num_players = random.randint(min_players, max_players)

        # Validate and set max values
        self.max_suspects = min(
            max_suspects or min(len(Clue.suspects), int(self.num_players * 2.5)),
            len(Clue.suspects),
        )
        self.max_weapons = min(
            max_weapons or min(len(Clue.weapons), int(self.num_players * 2.125)),
            len(Clue.weapons),
        )
        self.max_rooms = min(
            max_rooms or min(len(Clue.rooms), self.num_players * 3),
            len(Clue.rooms),
        )
        self.max_times = min(
            max_times or 2**31,
            int(self.num_players * 2),
        )
        self.max_motives = min(
            max_motives or 2**31,
            self.max_suspects,
        )
        self.unique_motives = (
            unique_motives
            if isinstance(unique_motives, bool)
            else random.random() < unique_motives
        )

        # Validate probabilities
        if not 0 <= character_move_probability <= 1:
            raise ValueError("character_move_probability must be between 0 and 1")
        if not 0 <= suspect_moves_weapon_probability <= 1:
            raise ValueError("suspect_moves_weapon_probability must be between 0 and 1")

        self.character_move_probability = character_move_probability
        self.suspect_moves_weapon_probability = suspect_moves_weapon_probability

        # Initialize random game parameters
        self.num_weapons = max(
            2,
            min(
                self.num_players + random.randint(-1, 5),
                self.max_weapons,
            ),
        )
        self.num_suspects = min(
            self.num_weapons + random.randint(0, self.num_weapons - 1),
            self.max_suspects,
        )
        self.num_rooms = min(
            self.num_suspects + random.randint(0, self.num_suspects - 2),
            self.max_rooms,
        )
        self.num_times = random.randint(
            min(self.num_players, self.max_times), self.max_times
        )
        self.num_motives = (
            self.num_suspects
            if self.unique_motives
            else random.randint(1, min(round(self.num_suspects / 2), self.max_motives))
        )

        # Select random game elements
        self.suspects = [
            Suspect(s) for s in random.sample(Clue.suspects, k=self.num_suspects)
        ]
        self.weapons = [
            Weapon(w) for w in random.sample(Clue.weapons, k=self.num_weapons)
        ]
        self.rooms = [Room(r) for r in random.sample(Clue.rooms, k=self.num_rooms)]

        # Generate time range
        self.frequency = random.choice([0.25, 0.5, 1.0])
        self.start = 24.0 - self.frequency if self.num_times > 1 else 0.0
        self.end = 0.0
        for _ in range(self.num_times - 2):
            if random.randint(0, 1):
                self.end += self.frequency
            else:
                self.start -= self.frequency

        # Generate times
        self.times = [
            Time(t)
            for t in Clue.get_times(
                self._format_time(self.start),
                self._format_time(self.end),
                f"{int(self.frequency * 60)}min",
            )
        ]

        self.motives = [
            Motive(m) for m in random.sample(Clue.motives, k=self.num_motives)
        ]

        # Create indices for lookup
        self.suspect_indices = {
            suspect: index for index, suspect in enumerate(self.suspects)
        }
        self.weapon_indices = {
            weapon: index for index, weapon in enumerate(self.weapons)
        }
        self.room_indices = {room: index for index, room in enumerate(self.rooms)}
        self.time_indices = {time: index for index, time in enumerate(self.times)}

        # Setup game board
        (
            self.board_columns,
            self.board_rows,
            self.room_coords,
            self.coord_rooms,
        ) = self._board_data()

        # Generate movement and room occupancy data
        self.character_moves: list[CharacterMove] = []
        self.suspect_time_rooms = self._generate_suspect_movements(self.character_moves)
        self.time_room_suspects = self._get_room_occupancy(self.suspect_time_rooms)
        self.weapon_moves, self.weapon_time_rooms = self._generate_weapon_movements()
        self.time_room_weapons = self._get_room_occupancy(self.weapon_time_rooms)

        # Select suspect motives
        self.suspect_motives = {
            suspect: motive
            for suspect, motive in zip(
                self.suspects,
                (
                    shuffled(self.motives)
                    if self.unique_motives
                    else random.choices(self.motives, k=self.num_suspects)
                ),
            )
        }

        # Generate murder solution
        self.murderer, self.murder_weapon, self.murder_room, self.murder_time = (
            self._generate_murder()
        )

        # Generate Mr. Boddy's time rooms
        self.mr_boddy_time_rooms = self._generate_mr_boddy_time_rooms()

        # Create a dictionary of all piece time rooms
        self.piece_time_rooms: dict[Piece, dict[Time, Room]] = {
            **self.suspect_time_rooms,
            **self.weapon_time_rooms,
            "Mr. Boddy": self.mr_boddy_time_rooms,
        }
        self.time_room_pieces = self._get_room_occupancy(self.piece_time_rooms)
        self.pieces = list(self.piece_time_rooms.keys())

        self.articles: dict[str, str] = {
            weapon_or_room: "the " for weapon_or_room in self.weapons + self.rooms
        }
        self.prepositions = {"Fountain": "at"}

    def _format_time(self, time: float) -> str:
        return f"{int(time):02d}:{int(60 * (time - int(time))):02d}"

    def _board_data(
        self,
    ) -> tuple[int, int, dict[Room, tuple[int, int]], dict[tuple[int, int], Room]]:
        """Sets up the game board dimensions and room coordinates

        Returns:
            tuple[int, int, dict[Room, tuple[int, int]], dict[tuple[int, int], Room]]:
                - board_columns: Number of columns on the board
                - board_rows: Number of rows on the board
                - room_coords: Dictionary mapping rooms to their coordinates
                - coord_rooms: Dictionary mapping coordinates to rooms
        """
        board_columns = {
            2: random.choice([1, 2]),
            3: 2,
            4: 2,
            5: random.choice([2, 3]),
            6: random.choice([2, 3]),
            7: 3,
            8: 3,
            9: 3,
            10: random.choice([3, 4]),
            11: random.choice([3, 4]),
            12: random.choice([3, 4]),
            13: 4,
            14: 4,
            15: 4,
            16: 4,
            17: 4,
        }[len(self.rooms)]
        board_rows = math.ceil(len(self.rooms) / board_columns)

        room_coords = {
            room: (
                id % board_columns,
                board_rows - id // board_columns - 1,
            )
            for id, room in enumerate(self.rooms)
        }
        coord_rooms = {v: k for k, v in room_coords.items()}

        return board_columns, board_rows, room_coords, coord_rooms

    def _generate_suspect_movements(
        self, character_moves: list[CharacterMove]
    ) -> dict[Suspect, dict[Time, Room]]:
        """Generates random movement patterns for suspects"""
        return {
            suspect: self._random_character_time_rooms(suspect, character_moves)
            for suspect in self.suspects
        }

    def _random_character_time_rooms(
        self,
        character: Character,
        character_moves: list[CharacterMove],
        args: tuple[Room, dict[Time, Room], list[Time]] | None = None,
    ) -> dict[Time, Room]:
        if args is None:
            room = random.choice(self.rooms)
            time_rooms = {self.times[0]: room}
            times = self.times
        else:
            room, time_rooms, times = args
        for time in times[1:]:
            if random.random() < self.character_move_probability:
                coords = self.room_coords[room]
                from_room = room
                room = random.choice(
                    [
                        room
                        for room in (
                            self.coord_rooms.get((coords[0] + x, coords[1] + y))
                            for x, y in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                        )
                        if room is not None
                    ]
                )
                character_moves.append(
                    CharacterMove(
                        character=character,
                        time=time,
                        from_room=from_room if args is None else room,
                        to_room=room if args is None else from_room,
                    )
                )
            time_rooms[time] = room
        return time_rooms

    def _generate_weapon_movements(
        self,
    ) -> tuple[list[WeaponMove], dict[Weapon, dict[Time, Room]]]:
        """Generates random movement patterns for weapons"""
        weapon_moves: list[WeaponMove] = []
        weapon_time_rooms = {}

        for weapon in self.weapons:
            weapon_time_rooms[weapon] = self._random_weapon_time_rooms(
                weapon, weapon_moves
            )

        return weapon_moves, weapon_time_rooms

    def _random_weapon_time_rooms(
        self, weapon: Weapon, weapon_moves: list[WeaponMove]
    ) -> dict[Time, Room]:
        room = random.choice(self.rooms)
        time_rooms = {self.times[0]: room}
        suspect: Suspect | None = None

        for time in self.times[1:]:
            if (
                suspect
                and (suspect_room := self.suspect_time_rooms[suspect][time])
                and suspect_room != room
                and random.random() < self.suspect_moves_weapon_probability
            ):
                # If a suspect is holding the weapon and moves it, add the move to the list
                weapon_moves.append(
                    WeaponMove(
                        weapon=weapon,
                        suspect=suspect,
                        time=time,
                        from_room=room,
                        to_room=suspect_room,
                    )
                )
                # Update the room to the suspect's room
                time_rooms[time] = room = suspect_room
            else:
                # Otherwise, the weapon is still in the original room
                time_rooms[time] = room

            if suspects := self.time_room_suspects[time][room]:
                # If there are any suspects, one of them randomly picks up the weapon
                suspect = random.choice(list(suspects))
            else:
                # Otherwise, no suspect is holding the weapon
                suspect = None

        return time_rooms

    def _get_room_occupancy(
        self, element_time_rooms: dict[T, dict[Time, Room]]
    ) -> dict[Time, dict[Room, set[T]]]:
        time_room_elements = {
            time: {room: set[T]() for room in self.rooms} for time in self.times
        }
        for element, time_rooms in element_time_rooms.items():
            for time, room in time_rooms.items():
                time_room_elements[time][room].add(element)
        return time_room_elements

    def _generate_murder(self) -> tuple[Suspect, Weapon, Room, Time]:
        """Generates the murder solution"""
        for _ in range(10):
            murderer = random.choice(self.suspects)
            murder_time = random.choice(self.times)
            murder_room = self.suspect_time_rooms[murderer][murder_time]

            # Ensure the suspect is alone at the murder time in the murder room
            if len(self.time_room_suspects[murder_time][murder_room]) > 1:
                continue

            # Ensure there is at least one weapon in the murder room at the murder time
            if possible_weapons := self.time_room_weapons[murder_time][murder_room]:
                murder_weapon = random.choice(list(possible_weapons))
                break
        else:
            raise NoValidMurderSolutionError(attempts=10)

        return murderer, murder_weapon, murder_room, murder_time

    def _generate_mr_boddy_time_rooms(self) -> dict[Time, Room]:
        """Generates Mr. Boddy's time rooms"""
        reversed_times = list(reversed(self.times))
        murder_time_index = reversed_times.index(self.murder_time)
        return {
            time: room
            for time, room in reversed(
                self._random_character_time_rooms(
                    "Mr. Boddy",
                    character_moves=[],
                    args=(
                        self.murder_room,
                        {
                            time: self.murder_room
                            for time in reversed_times[: murder_time_index + 1]
                        },
                        reversed_times[murder_time_index:],
                    ),
                ).items()
            )
        }

    def formatted_board(self) -> str:
        """Returns a markdown table representing the layout of the mansion.

        Each cell contains a number corresponding to the room index (1-based)
        or a hyphen for empty cells. The numbers increment from left to right,
        top to bottom based on the room indices. Compass directions (N,S,E,W)
        are added around the board.
        """
        # Create empty board
        board = [
            ["-" for _ in range(self.board_columns)] for _ in range(self.board_rows)
        ]

        # Fill in room numbers (1-based indexing)
        for room, (x, y) in self.room_coords.items():
            board[self.board_rows - y - 1][x] = str(self.room_indices[room] + 1)

        # Add North/South indicators above each column
        north = "  " + " ".join("N" * self.board_columns) + "  \n"
        south = "  " + " ".join("S" * self.board_columns) + "  "

        # Add West/East indicators and format rows
        rows = []
        for row in board:
            rows.append("W " + "|".join(row) + " E")

        # Combine all parts
        return north + "\n".join(rows) + "\n" + south

    def __repr__(self) -> str:

        @dataclass
        class TemporalClue:
            suspects: list[Suspect]
            weapons: list[Weapon]
            rooms: list[Room]
            times: list[Time]
            motives: list[Motive]
            murderer: Suspect
            murder_weapon: Weapon
            murder_room: Room
            murder_time: Time

        return black.format_str(
            str(
                TemporalClue(
                    suspects=self.suspects,
                    weapons=self.weapons,
                    rooms=self.rooms,
                    times=self.times,
                    motives=self.motives,
                    murderer=self.murderer,
                    murder_weapon=self.murder_weapon,
                    murder_room=self.murder_room,
                    murder_time=self.murder_time,
                )
            ).split("<locals>.")[1],
            mode=black.Mode(),
        )

    def create_puzzle(
        self,
    ) -> TemporalCluePuzzle:
        """Creates a puzzle from the ground truth data"""
        solution_clues = list(self.solution_clues())
        piece_time_room_clues = list(self.piece_time_room_clues())
        motive_clues = list(self.motive_clues())
        clues = solution_clues + piece_time_room_clues + motive_clues
        clues += list(self.indirect_clues())
        clues = sorted(clues, key=lambda clue: clue.bias + random.random())
        needed_clues: list[TemporalClue] = []

        for popped_clue in itertools.chain(
            [None], (clues.pop(0) for _ in range(len(clues))), [None]
        ):
            model = TemporalClueCpModel(self)
            for clue in clues + needed_clues:
                model.add_assumption(clue.get_bool_var(model))
            if not self.sufficient_clues(model):
                if popped_clue is None:
                    print(len(clues), len(needed_clues))
                    self.sufficient_clues(model, debug=True)
                    raise ValueError("No valid puzzle found")
                needed_clues.append(popped_clue)

        questions = {
            clue.question_answer[0]: clue.question_answer[1]
            for clue in solution_clues
            if not clue in needed_clues and clue.question_answer is not None
        }
        bonus_questions = {
            clue.question_answer[0]: clue.question_answer[1]
            for clue in shuffled(piece_time_room_clues + motive_clues)[
                : 8 - len(questions)
            ]
            if clue not in needed_clues and clue.question_answer is not None
        }

        # For now, return a basic puzzle with no clues
        return TemporalCluePuzzle(
            scenario=self,
            clues=shuffled([clue.description for clue in needed_clues]),
            questions=questions,
            bonus_questions=bonus_questions,
        )

    def solution_clues(self) -> Iterable[TemporalClue]:
        yield TemporalClue(
            description=f"Mr. Boddy was murdered by {self.murderer}",
            get_bool_var=lambda model: model.murderer_vars[self.murderer],
            bias=-0.2,
            question_answer=("Who murdered Mr. Boddy?", self.murderer),
        )
        yield TemporalClue(
            description=f"Mr. Boddy was killed with the {self.murder_weapon}",
            get_bool_var=lambda model: model.murder_weapon_vars[self.murder_weapon],
            bias=-0.2,
            question_answer=("What weapon did the murderer use?", self.murder_weapon),
        )
        yield TemporalClue(
            description=f"Mr. Boddy was murdered in the {self.murder_room}",
            get_bool_var=lambda model: model.murder_room_vars[self.murder_room],
            bias=-0.2,
            question_answer=("Where was the murder committed?", self.murder_room),
        )
        if len(self.times) > 1:
            yield TemporalClue(
                description=f"Mr. Boddy was murdered at {self.murder_time}",
                get_bool_var=lambda model: model.murder_time_vars[self.murder_time],
                bias=-0.2,
                question_answer=("When did the murder occur?", self.murder_time),
            )
        if len(self.motives) > 1:
            yield TemporalClue(
                description=f"Mr. Boddy was murdered by {'the' if self.unique_motives else 'a'} suspect motivated by {self.suspect_motives[self.murderer]}",
                get_bool_var=lambda model: model.categorical_exactly_one_equal(
                    model.motive_suspect_vars[self.suspect_motives[self.murderer]],
                    model.murderer_vars,
                ),
                bias=-0.2,
                question_answer=(
                    "Why did the murderer do it?",
                    self.suspect_motives[self.murderer],
                ),
            )

    def piece_time_room_clues(self, false: bool = False) -> Iterable[TemporalClue]:
        for piece, time_rooms in shuffled(self.piece_time_rooms.items()):
            for time, room in shuffled(time_rooms.items()):
                if false:
                    room = random.choice([r for r in self.rooms if r != room])
                if piece == self.murderer:
                    yield TemporalClue(
                        get_bool_var=lambda model, t=time, r=room: (
                            model.murderer_time_room_var(t, r)
                        ),
                        description=f"The murderer was {self.prepositions.get(room, 'in')} the {room} at {time}",
                    )
                if piece == self.murder_weapon:
                    yield TemporalClue(
                        get_bool_var=lambda model, t=time, r=room: model.murder_weapon_time_room_var(
                            t, r
                        ),
                        description=f"The murder weapon was {self.prepositions.get(room, 'in')} the {room} at {time}",
                    )
                yield TemporalClue(
                    description=f"{self.articles.get(piece, '').capitalize()}{piece} was {self.prepositions.get(room, 'in')} the {room} at {time}",
                    get_bool_var=lambda model, p=piece, t=time, r=room: (
                        model.piece_time_room_vars[p][t][r]
                    ),
                    question_answer=(
                        (
                            f"Where was {self.articles.get(piece, '')}{piece} at {time}?",
                            room,
                        )
                        if piece != "Mr. Boddy"
                        else None
                    ),
                )
                if (
                    len(self.motives) > 1
                    and piece in self.suspects
                    and (not false or self.unique_motives)
                ):
                    yield TemporalClue(
                        description=f"{'The' if self.unique_motives else 'A'} suspect motivated by {self.suspect_motives[piece]} was {self.prepositions.get(room, 'in')} the {room} at {time}",
                        get_bool_var=lambda model, s=piece, t=time, r=room: model.motive_time_room_var(
                            self.suspect_motives[s], t, r
                        ),
                        question_answer=(
                            (
                                f"Where was the suspect motivated by {self.suspect_motives[piece]} at {time}?",
                                room,
                            )
                            if self.unique_motives
                            else None
                        ),
                    )

    def motive_clues(self) -> Iterable[TemporalClue]:
        if len(self.motives) > 1:
            for suspect, motive in self.suspect_motives.items():
                yield TemporalClue(
                    description=f"{suspect} is motivated by {motive}",
                    get_bool_var=lambda model, s=suspect, m=motive: (
                        model.suspect_motive_vars[s][m]
                    ),
                    question_answer=(
                        (f"What motivates {suspect}?", motive)
                        if random.random() < 0.5 or not self.unique_motives
                        else (f"Who is motivated by {motive}?", suspect)
                    ),
                )

    def indirect_clues(self) -> Iterable[TemporalClue]:
        yield from self.xor_clues()
        yield from self.same_time_or_place_clues()
        yield from self.character_move_clues()
        yield from self.weapon_move_clues()
        yield from self.spatial_relation_clues()

    def xor_clues(self) -> Iterable[TemporalClue]:
        for clue1, clue2 in zip(
            shuffled(
                itertools.chain(
                    self.piece_time_room_clues(),
                    # self.same_time_or_place_clues(),
                )
            ),
            shuffled(
                itertools.chain(
                    self.piece_time_room_clues(false=True),
                    # self.same_time_or_place_clues(false=True),
                )
            ),
        ):
            if random.random() < 0.5:
                clue1, clue2 = clue2, clue1
            yield TemporalClue(
                get_bool_var=lambda model, c1=clue1, c2=clue2: (
                    model.sum(c1.get_bool_var(model), c2.get_bool_var(model), value=1)
                ),
                description=f"{clue1.description} or {clue2.description.replace("The ", "the ")}",
                bias=(clue1.bias + clue2.bias) / 2,
            )

    def same_time_or_place_clues(self, false: bool = False) -> Iterable[TemporalClue]:
        for time, room_pieces in shuffled(self.time_room_pieces.items()):
            for room, pieces in shuffled(room_pieces.items()):
                if not len(pieces) > 1 and not false:
                    continue
                pieces = shuffled(pieces)
                if false:
                    piece1, piece2 = random.sample(self.pieces, k=2)
                    if piece1 in pieces and piece2 in pieces:
                        continue
                else:
                    piece1 = pieces.pop()
                    piece2 = pieces.pop()
                motive = None
                if (
                    random.random() < 0.66
                    and piece1 == self.murderer
                    and not piece2 in self.suspects
                ):
                    piece1 = "The murderer"
                elif (
                    random.random() < 0.5
                    and len(self.motives) > 1
                    and piece1 in self.suspects
                    and not piece2 in self.suspects
                ):
                    motive = self.suspect_motives[piece1]
                    piece1 = f"{'The' if self.unique_motives else 'A'} suspect motivated by {self.suspect_motives[piece1]}"
                if (
                    random.random() < 0.66
                    and piece1 == self.murder_weapon
                    and not piece2 in self.weapons
                ):
                    piece1 = "The murder weapon"
                if random.random() < 0.5:
                    yield TemporalClue(
                        description=f"{self.articles.get(piece1, '').capitalize()}{piece1} was in the same room as {self.articles.get(piece2, '')}{piece2} at {time}",
                        get_bool_var=lambda model, p1=piece1, p2=piece2, t=time, m=motive: model.categorical_exactly_one_equal(
                            (
                                model.motive_room_vars(m, t)
                                if m is not None
                                else (
                                    model.murderer_room_vars(t)
                                    if p1 == "The murderer"
                                    else (
                                        model.murder_weapon_room_vars(t)
                                        if p1 == "The murder weapon"
                                        else model.piece_time_room_vars[p1][t]
                                    )
                                )
                            ),
                            model.piece_time_room_vars[p2][t],
                        ),
                        bias=0.5 + (1.5 if "murder" in piece1 else 0),
                    )
                elif len(self.times) > 1:
                    yield TemporalClue(
                        description=f"{self.articles.get(piece1, '').capitalize()}{piece1} and {self.articles.get(piece2, '')}{piece2} were {self.prepositions.get(room, 'in')} the {room} together at least once",
                        get_bool_var=lambda model, p1=piece1, p2=piece2, r=room, m=motive: model.categorical_any_equal(
                            (
                                model.motive_time_vars(m, r)
                                if m is not None
                                else (
                                    model.murderer_time_vars(r)
                                    if p1 == "The murderer"
                                    else (
                                        model.murder_weapon_time_vars(r)
                                        if p1 == "The murder weapon"
                                        else model.piece_room_time_vars[p1][r]
                                    )
                                )
                            ),
                            model.piece_room_time_vars[p2][r],
                        ),
                        bias=0.5 + (1.5 if "murder" in piece1 else 0),
                    )

    def character_move_clues(self) -> Iterable[TemporalClue]:
        for character_move in shuffled(self.character_moves):
            prev_time = self.times[self.time_indices[character_move.time] - 1]
            character = character_move.character
            if character == self.murderer:
                yield TemporalClue(
                    description=f"The murderer moved from the {character_move.from_room} to the {character_move.to_room} at {character_move.time}",
                    get_bool_var=lambda model, c=character_move, pt=prev_time: model.all(
                        model.murderer_time_room_var(pt, c.from_room),
                        model.murderer_time_room_var(c.time, c.to_room),
                    ),
                    bias=0.2,
                )
            if len(self.motives) > 1 and character in self.suspects:
                yield TemporalClue(
                    description=f"{'The' if self.unique_motives else 'A'} suspect motivated by {self.suspect_motives[character]} moved from the {character_move.from_room} to the {character_move.to_room} at {character_move.time}",
                    get_bool_var=lambda model, c=character, m=character_move, pt=prev_time: model.all(
                        model.motive_time_room_var(
                            self.suspect_motives[c], pt, m.from_room
                        ),
                        model.motive_time_room_var(
                            self.suspect_motives[c], m.time, m.to_room
                        ),
                    ),
                    bias=0.1,
                )
            yield TemporalClue(
                description=f"{character} moved from the {character_move.from_room} to the {character_move.to_room} at {character_move.time}",
                get_bool_var=lambda model, m=character_move, pt=prev_time: model.all(
                    model.character_time_room_vars[m.character][pt][m.from_room],
                    model.character_time_room_vars[m.character][m.time][m.to_room],
                ),
                bias=0.1,
            )

    def weapon_move_clues(self) -> Iterable[TemporalClue]:
        for weapon_move in shuffled(self.weapon_moves):
            prev_time = self.times[self.time_indices[weapon_move.time] - 1]
            if weapon_move.suspect == self.murderer:
                yield TemporalClue(
                    description=f"The murderer moved the {weapon_move.weapon} from the {weapon_move.from_room} to the {weapon_move.to_room} at {weapon_move.time}",
                    get_bool_var=lambda model, w=weapon_move, pt=prev_time: model.all(
                        model.murderer_time_room_var(pt, w.from_room),
                        model.murderer_time_room_var(w.time, w.to_room),
                        model.weapon_time_room_vars[w.weapon][pt][w.from_room],
                        model.weapon_time_room_vars[w.weapon][w.time][w.to_room],
                    ),
                    bias=0.3,
                )
            if weapon_move.weapon == self.murder_weapon:
                yield TemporalClue(
                    description=f"{weapon_move.suspect} moved the murder weapon from the {weapon_move.from_room} to the {weapon_move.to_room} at {weapon_move.time}",
                    get_bool_var=lambda model, w=weapon_move, pt=prev_time: model.all(
                        model.suspect_time_room_vars[w.suspect][pt][w.from_room],
                        model.suspect_time_room_vars[w.suspect][w.time][w.to_room],
                        model.murder_weapon_time_room_var(pt, w.from_room),
                        model.murder_weapon_time_room_var(w.time, w.to_room),
                    ),
                    bias=0.3,
                )
            if len(self.motives) > 1:
                yield TemporalClue(
                    description=f"{'The' if self.unique_motives else 'A'} suspect motivated by {self.suspect_motives[weapon_move.suspect]} moved the {weapon_move.weapon} from the {weapon_move.from_room} to the {weapon_move.to_room} at {weapon_move.time}",
                    get_bool_var=lambda model, w=weapon_move, pt=prev_time: model.all(
                        model.motive_time_room_var(
                            self.suspect_motives[w.suspect], pt, w.from_room
                        ),
                        model.motive_time_room_var(
                            self.suspect_motives[w.suspect], w.time, w.to_room
                        ),
                        model.weapon_time_room_vars[w.weapon][pt][w.from_room],
                        model.weapon_time_room_vars[w.weapon][w.time][w.to_room],
                    ),
                    bias=0.15,
                )
            yield TemporalClue(
                description=f"{weapon_move.suspect} moved the {weapon_move.weapon} from the {weapon_move.from_room} to the {weapon_move.to_room} at {weapon_move.time}",
                get_bool_var=lambda model, w=weapon_move, pt=prev_time: model.all(
                    model.suspect_time_room_vars[w.suspect][pt][w.from_room],
                    model.suspect_time_room_vars[w.suspect][w.time][w.to_room],
                    model.weapon_time_room_vars[w.weapon][pt][w.from_room],
                    model.weapon_time_room_vars[w.weapon][w.time][w.to_room],
                ),
                bias=0.15,
            )

    def spatial_relation_clues(self) -> Iterable[TemporalClue]:
        """Generates clues based on spatial relationships between pieces at the same time."""
        for _ in range(10):
            time = random.choice(self.times)
            (room1, pieces1), (room2, pieces2) = random.sample(
                list(
                    (room, pieces)
                    for room, pieces in self.time_room_pieces[time].items()
                    if pieces
                ),
                k=2,
            )
            piece1, piece2 = random.choice(list(pieces1)), random.choice(list(pieces2))
            coords1, coords2 = self.room_coords[room1], self.room_coords[room2]
            dx = coords1[0] - coords2[0]
            dy = coords1[1] - coords2[1]
            if dx != 0 and dy != 0:
                continue
            if dx != 0:
                direction = "east" if dx > 0 else "west"
                distance = abs(dx)
            else:
                direction = "north" if dy > 0 else "south"
                distance = abs(dy)
            if distance > 1:
                continue
            # TODO: Bring back support for distance > 1
            # if distance > 1 or random.random() < 0.5:
            #     distance = 2
            pairs = [
                (r1, r2)
                for r1 in self.rooms
                for r2 in self.rooms
                if r1 != r2
                and (
                    self.room_coords[r1]
                    == (
                        self.room_coords[r2][0] + dx,
                        self.room_coords[r2][1] + dy,
                    )
                    if distance == 1
                    else (
                        (
                            dx > 0
                            and self.room_coords[r1][0] > self.room_coords[r2][0]
                            and self.room_coords[r1][1] == self.room_coords[r2][1]
                        )
                        or (
                            dx < 0
                            and self.room_coords[r1][0] < self.room_coords[r2][0]
                            and self.room_coords[r1][1] == self.room_coords[r2][1]
                        )
                        or (
                            dy > 0
                            and self.room_coords[r1][0] == self.room_coords[r2][0]
                            and self.room_coords[r1][1] > self.room_coords[r2][1]
                        )
                        or (
                            dy < 0
                            and self.room_coords[r1][0] == self.room_coords[r2][0]
                            and self.room_coords[r1][1] < self.room_coords[r2][1]
                        )
                    )
                )
            ]
            assert (room1, room2) in pairs
            yield TemporalClue(
                description=(
                    f"{self.articles.get(piece1, '').capitalize()}"
                    f"{piece1} was in {'a' if distance > 1 else 'the'} room "
                    f"{'just' if distance == 1 else 'due'} {direction} of "
                    f"{self.articles.get(piece2, '')}{piece2} at {time}"
                ),
                get_bool_var=lambda model, p1=piece1, p2=piece2, t=time, ps=pairs: model.sum(
                    *(
                        model.all(
                            model.piece_time_room_vars[p1][t][r1],
                            model.piece_time_room_vars[p2][t][r2],
                        )
                        for r1, r2 in ps
                    ),
                    value=1,
                ),
                bias=0.1,
            )

    def sufficient_clues(self, model: TemporalClueCpModel, debug: bool = False) -> bool:
        """
        Returns True if the clues in the model are sufficient to uniquely determine the solution.

        A set of clues is sufficient if there is only one valid solution. If there is no feasible
        solution or if the solution doesn't match the ground truth then an assertion error is raised.
        """
        callback = SolutionCallback()
        solver = cp_model.CpSolver()
        solver.parameters.enumerate_all_solutions = True
        status = solver.solve(model, callback)
        assert (
            status == cp_model.OPTIMAL or status == cp_model.FEASIBLE
        ), solver.status_name()
        if callback.num_solutions > 1 and not debug:
            return False
        self._validate_categorical_vars(
            callback,
            self.murderer,
            model.murderer_vars,
        )
        self._validate_categorical_vars(
            callback, self.murder_weapon, model.murder_weapon_vars
        )
        self._validate_categorical_vars(
            callback, self.murder_room, model.murder_room_vars
        )
        self._validate_categorical_vars(
            callback, self.murder_time, model.murder_time_vars
        )
        for piece, time_rooms in self.piece_time_rooms.items():
            for time, room in time_rooms.items():
                self._validate_categorical_vars(
                    callback, room, model.piece_time_room_vars[piece][time]
                )
        return True

    def _validate_categorical_vars(
        self,
        callback: cp_model.CpSolverSolutionCallback,
        value: H,
        vars: dict[H, cp_model.IntVar],
    ) -> None:
        for key, var in vars.items():
            assert callback.value(var) == (key == value), (
                f"Did not get expected value {value}"
                if key == value
                else f"Got unexpected value {key} instead of {value}"
            )

    def create_printable_boards(
        self, output_path: str, page_width: int = 2550, page_height: int = 3300
    ) -> None:
        """Creates printable images of the game boards for all time periods.

        Args:
            output_path: Path to save the output image(s). Will append page numbers if multiple pages are needed.
            page_width: Width of the page in pixels (default: 2550 for 8.5" at 300dpi)
            page_height: Height of the page in pixels (default: 3300 for 11" at 300dpi)
        """
        MIN_CELL_SIZE = 450  # Minimum size for a room cell in pixels
        MIN_MARGIN = 150  # Minimum margin between boards and edges
        PAGE_MARGIN = 50  # Minimum margin around the page edges

        # Adjust page dimensions to account for margins
        usable_width = page_width - (2 * PAGE_MARGIN)
        usable_height = page_height - (2 * PAGE_MARGIN)

        # Try both portrait and landscape orientations
        portrait_layout = self._calculate_board_layout(
            usable_width, usable_height, MIN_CELL_SIZE, MIN_MARGIN
        )
        landscape_layout = self._calculate_board_layout(
            usable_height, usable_width, MIN_CELL_SIZE, MIN_MARGIN
        )

        # Choose the better layout
        if landscape_layout["utilization"] > portrait_layout["utilization"]:
            usable_width, usable_height = usable_height, usable_width
            layout = landscape_layout
        else:
            layout = portrait_layout

        # Add page margin to layout margins
        layout["margin_x"] += PAGE_MARGIN
        layout["margin_y"] += PAGE_MARGIN

        # Calculate number of boards per page and total pages needed
        boards_per_page = layout["cols"] * layout["rows"]
        total_pages = math.ceil(len(self.times) / boards_per_page)

        # Setup font paths
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "C:/Windows/Fonts/arial.ttf",  # Windows
            "/usr/share/fonts/TTF/DejaVuSans.ttf",  # Some Linux
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Ubuntu
        ]

        # Store all pages
        pages = []

        # Create each page
        for page in range(total_pages):
            img = Image.new("RGB", (page_width, page_height), "white")
            draw = ImageDraw.Draw(img)

            # Load fonts for this page
            font = None
            time_font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(
                            font_path, size=layout["cell_size"] // 12
                        )
                        time_font = ImageFont.truetype(
                            font_path, size=layout["cell_size"] // 8
                        )
                        break
                    except OSError:
                        continue

            if font is None:
                font = ImageFont.load_default()
                font_size = max(layout["cell_size"] // 12, 24)
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)

            time_font = time_font or font

            # Draw boards for this page
            start_idx = page * boards_per_page
            end_idx = min((page + 1) * boards_per_page, len(self.times))

            for idx in range(start_idx, end_idx):
                relative_idx = idx - start_idx
                col = relative_idx % layout["cols"]
                row = relative_idx // layout["cols"]

                # Calculate board position with centered margins
                x_offset = layout["margin_x"] + col * (
                    layout["board_width"] + MIN_MARGIN
                )
                y_offset = layout["margin_y"] + row * (
                    layout["board_height"] + MIN_MARGIN
                )

                # Draw time label
                time = self.times[idx]
                draw.text(
                    (x_offset + layout["board_width"] // 2, y_offset - MIN_MARGIN // 2),
                    str(time),
                    fill="black",
                    font=time_font,
                    anchor="mm",
                )

                # Draw each cell in the board
                for y in range(self.board_rows):
                    for x in range(self.board_columns):
                        cell_x = x_offset + x * layout["cell_size"]
                        cell_y = y_offset + y * layout["cell_size"]

                        coords = (x, self.board_rows - y - 1)
                        room = self.coord_rooms.get(coords)

                        if room:
                            # Draw room
                            draw.rectangle(
                                [
                                    cell_x,
                                    cell_y,
                                    cell_x + layout["cell_size"],
                                    cell_y + layout["cell_size"],
                                ],
                                outline="black",
                                width=2,
                            )

                            # Draw room name
                            room_text = str(room)
                            draw.text(
                                (
                                    cell_x + layout["cell_size"] // 2,
                                    cell_y + layout["cell_size"] // 2,
                                ),
                                room_text,
                                fill=(0, 0, 0, 128),
                                font=font,
                                anchor="mm",
                            )

                            # Draw borders with adjacent rooms
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                adj_coords = (coords[0] + dx, coords[1] + dy)
                                if adj_coords in self.coord_rooms:
                                    # Draw thicker line between adjacent rooms
                                    if dx == 1:  # Right border
                                        draw.line(
                                            [
                                                cell_x + layout["cell_size"],
                                                cell_y,
                                                cell_x + layout["cell_size"],
                                                cell_y + layout["cell_size"],
                                            ],
                                            fill="black",
                                            width=4,
                                        )
                                    elif dx == -1:  # Left border
                                        draw.line(
                                            [
                                                cell_x,
                                                cell_y,
                                                cell_x,
                                                cell_y + layout["cell_size"],
                                            ],
                                            fill="black",
                                            width=4,
                                        )
                                    elif dy == 1:  # Bottom border
                                        draw.line(
                                            [
                                                cell_x,
                                                cell_y + layout["cell_size"],
                                                cell_x + layout["cell_size"],
                                                cell_y + layout["cell_size"],
                                            ],
                                            fill="black",
                                            width=4,
                                        )
                                    elif dy == -1:  # Top border
                                        draw.line(
                                            [
                                                cell_x,
                                                cell_y,
                                                cell_x + layout["cell_size"],
                                                cell_y,
                                            ],
                                            fill="black",
                                            width=4,
                                        )

            # Convert to RGB if needed (for PDF compatibility)
            if img.mode != "RGB":
                img = img.convert("RGB")
            pages.append(img)

        # Save all pages to a single PDF
        if pages:
            first_page = pages[0]
            if len(pages) > 1:
                first_page.save(
                    output_path,
                    "PDF",
                    resolution=300.0,
                    save_all=True,
                    append_images=pages[1:],
                )
            else:
                first_page.save(output_path, "PDF", resolution=300.0)

    def _calculate_board_layout(
        self, width: int, height: int, min_cell_size: int, padding: int
    ) -> dict:
        """Calculate the optimal board layout for given dimensions."""
        # Calculate maximum boards that can fit on a page
        max_board_width = min_cell_size * self.board_columns
        max_board_height = min_cell_size * self.board_rows

        # Calculate how many boards can fit horizontally and vertically
        cols = max(1, (width - 2 * padding) // (max_board_width + padding))
        rows = max(1, (height - 2 * padding) // (max_board_height + padding))

        # Calculate actual cell size based on available space
        cell_size = min(
            (width - (cols + 1) * padding) // (cols * self.board_columns),
            (height - (rows + 1) * padding) // (rows * self.board_rows),
        )

        board_width = cell_size * self.board_columns
        board_height = cell_size * self.board_rows

        # Calculate total width and height of all boards including internal padding
        total_boards_width = cols * board_width + (cols - 1) * padding
        total_boards_height = rows * board_height + (rows - 1) * padding

        # Calculate margins to center the content
        margin_x = (width - total_boards_width) // 2
        margin_y = (height - total_boards_height) // 2

        # Calculate space utilization
        used_space = cols * rows * board_width * board_height
        total_space = width * height
        utilization = used_space / total_space

        return {
            "cols": cols,
            "rows": rows,
            "cell_size": cell_size,
            "board_width": board_width,
            "board_height": board_height,
            "margin_x": margin_x,
            "margin_y": margin_y,
            "utilization": utilization,
        }
