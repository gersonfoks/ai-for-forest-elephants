import unittest

from src.data.event_logic import event_overlaps_with_window, fit_event_to_window


class TestEventLogic(unittest.TestCase):

    def test_event_overlaps_with_window(self):

        # List if all the different cases
        cases = [
            # event, window_start, window_end, expected
            # event contains window
            ((0, 10), 5, 6, True),
            # event is contained in window
            ((5, 6), 0, 10, True),

            # event overlaps with window start
            ((0, 5), 4, 6, True),
            # event overlaps with window end
            ((4, 10), 0, 6, True),

            # event is before window
            ((0, 1), 2, 3, False),
            # event is after window
            ((2, 3), 0, 1, False),

        ]

        for case in cases:
            event, window_start, window_end, expected = case
            actual = event_overlaps_with_window(event, window_start, window_end)
            self.assertEqual(expected, actual)

    def test_fit_event_to_window(self):
        # List if all the different cases
        cases = [
            # event, window_start, window_end, expected
            # event contains window
            ((0, 10), 5, 6, (5, 6)),
            # event is contained in window
            ((5, 6), 0, 10, (5, 6)),
            # event overlaps with window start
            ((0, 5), 4, 6, (4, 5)),
            # event overlaps with window end
            ((4, 10), 0, 6, (4, 6)),
        ]

        for case in cases:
            event, window_start, window_end, expected = case
            actual = fit_event_to_window(event, window_start, window_end)
            self.assertEqual(expected, actual)
