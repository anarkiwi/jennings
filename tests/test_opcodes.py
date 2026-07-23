"""The canonical opcode table: complete, well-formed, and shared with deity."""

from jennings.opcodes import OPCODES, ILLEGAL_OPCODES, MODE_LEN, MODE_TOKEN


def test_table_is_complete_and_wellformed():
    assert len(OPCODES) == 256
    assert len(ILLEGAL_OPCODES) == 105
    for op, (mn, mode) in OPCODES.items():
        assert 0 <= op <= 255
        assert mode in MODE_LEN and mode in MODE_TOKEN
        assert isinstance(mn, str) and mn


def test_deity_agrees_with_jennings_table():
    """jennings' canonical table must match whatever deity executes against.

    With deity-informant>=0.4.0 these are the same object (deity imports this
    module); against older deity they are still value-equal, which is the guard
    that matters -- disasm and execution must not diverge.
    """
    import deity_informant

    assert dict(deity_informant.OPS) == dict(OPCODES)
    assert set(deity_informant.ILLEGAL_OPCODES) == set(ILLEGAL_OPCODES)
