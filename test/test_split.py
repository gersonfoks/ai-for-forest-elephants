import unittest

from src.data.event_logic import event_overlaps_with_window, fit_event_to_window
from src.data.split import extract_dir_and_nr, compare


class TestSplit(unittest.TestCase):

    def test_extract_dir_and_nr(self):
        
        cases = [
            ('../x/0/0.wav', (0,0)),
            ('../x/10/2.wav', (10,2)),
            
        ]
        
        for ref, expected in cases:
            (d, n) = extract_dir_and_nr(ref)
            
            self.assertEqual((d,n), expected)

    def test_compare_ref(self):
        cases = [
            ('../x/0/0.wav', '../x/0/0.wav', 0),
            ('../x/0/0.wav', '../x/0/1.wav', -1),
            ('../x/0/1.wav', '../x/0/0.wav', 1),
            ('../x/0/0.wav', '../x/1/0.wav', -1),
            ('../x/1/0.wav', '../x/0/0.wav', 1),
        ]

        for ref_1, ref_2, expected in cases:
            self.assertEqual(compare(ref_1, ref_2), expected)
