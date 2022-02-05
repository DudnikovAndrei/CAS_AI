from collections import namedtuple

Experience = namedtuple("Experience",
                        field_names=["state", "action", "reward", "done", "new_state"])
