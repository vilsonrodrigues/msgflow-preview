from datetime import datetime
from time import time
from typing import Dict


class EventsTiming:
    r""" The class is designed to capture and store the start and end times of events,
    along with their duration in seconds. Times are stored in both Unix timestamp
    (seconds since 1970) and ISO 8601 format.

    Attributes:
        events (Dict[str, Dict[str, str]]): A dictionary to store event timing information.
            Each event is stored as a key, with its value being a dictionary containing:
            - "start_unix": Start time in Unix timestamp.
            - "start_iso": Start time in ISO 8601 format.
            - "end_unix": End time in Unix timestamp.
            - "end_iso": End time in ISO 8601 format.
            - "duration_seconds": Duration of the event in seconds.

    Example:
        >>> events_timing = EventsTiming()
        >>> events_timing.start("generation")
        >>> # Simulate some processing time
        >>> from time import sleep
        >>> sleep(2)
        >>> events_timing.end("generation")
        >>> print(events_timing.get_event("generation"))
        {
            'start_unix': '1698765432',
            'start_iso': '2023-10-31T12:37:12.123456Z',
            'end_unix': '1698765434',
            'end_iso': '2023-10-31T12:37:14.123456Z',
            'duration_seconds': 2
        }
        >>> print(events_timing.get_events())
        {'generation': {
            'start_unix': '1698765432',
            'start_iso': '2023-10-31T12:37:12.123456Z',
            'end_unix': '1698765434',
            'end_iso': '2023-10-31T12:37:14.123456Z',
            'duration_seconds': 2
            }
        }
    """

    def __init__(self):
        self.events = {}

    @staticmethod
    def _get_iso_time():
        return datetime.now().isoformat() + "Z"
    
    def _get_unix_time():
        return str(int(time()))

    def start(self, event_name: str):
        """ Defines the start of an event """
        unix_time = EventsTiming._get_unix_time()
        iso_time = EventsTiming._get_iso_time()
        self.events[event_name] = {
            "start_unix": unix_time,
            "start_iso": iso_time,
            "end_unix": None,
            "end_iso": None,
            "duration_seconds": None
        }

    def end(self, event_name: str):
        """ Defines the end of an event and calculates the duration """
        if event_name not in self.events:
            raise ValueError(f"Event `{event_name}` not found")
        
        unix_time = EventsTiming._get_unix_time()
        iso_time = EventsTiming._get_iso_time()
        
        self.events[event_name]["end_unix"] = unix_time
        self.events[event_name]["end_iso"] = iso_time
        
        # Calculates event duration in seconds
        start_time = self.events[event_name]["start_unix"]
        self.events[event_name]["duration_seconds"] = str(int(unix_time) - int(start_time))

    def get_event(self, event_name) -> Dict[str, str]:
        """ Returns an event """
        if event_name not in self.events:
            raise ValueError(f"Event `{event_name}` not found")
        return self.events[event_name]

    def get_events(self) -> Dict[str, Dict[str, str]]:
        """ Returns captured events """
        return self.events
