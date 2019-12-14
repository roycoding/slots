# Assuming pytest
from slots.slots import MAB

# Most basic test of defaults
def test_mab():
    mab = MAB()
    mab.run()
