from gym_minigrid.minigrid import WorldObj, Key, Door


class ColorDoor(Door):
    """
    A Door instance where the key color can be specified and doesn't have to match the door
    """
    def __init__(self, color, is_open=False, is_locked=False, key_color=None):
        super().__init__(color, is_open, is_locked)
        self.is_open = is_open
        self.is_locked = is_locked
        if key_color:
            self.key_color = key_color
        else:
            self.key_color = color

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.key_color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True
