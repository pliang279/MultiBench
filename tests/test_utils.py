
def test_AUPRC():
    from utils.AUPRC import ptsort, AUPRC
    assert ptsort([0,1,2]) == 0
    assert AUPRC([(0,0),(1,1)]) == 1.0