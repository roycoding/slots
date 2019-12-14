# Assuming pytest
from slots.slots import MAB

# Most basic test of defaults
def test_mab():
    mab = MAB()
    mab.run()

def test_run():
    print('test')
