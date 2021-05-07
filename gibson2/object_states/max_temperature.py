from gibson2.object_states.temperature import Temperature
from gibson2.object_states.object_state_base import AbsoluteObjectState


class MaxTemperature(AbsoluteObjectState):
    """
    This state remembers the highest temperature reached by an object.
    """

    @staticmethod
    def get_dependencies():
        return AbsoluteObjectState.get_dependencies() + [Temperature]

    def __init__(self, obj):
        super(MaxTemperature, self).__init__(obj)

        self.value = float("-inf")

    def get_value(self):
        return self.value

    def set_value(self, new_value):
        self.value = new_value

    def _update(self, simulator):
        self.value = max(self.obj.states[Temperature].get_value(), self.value)

    # For our serialization, we just dump the value.
    def dump(self):
        return self.value

    def load(self, data):
        self.set_value(data)