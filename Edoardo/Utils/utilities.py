def create_id(class_type: type) -> int:
    pass

NEXT_INNOVATION_NUMBER = -1
def _get_next_innovation_number() -> int:
    global NEXT_INNOVATION_NUMBER
    NEXT_INNOVATION_NUMBER += 1
    return NEXT_INNOVATION_NUMBER