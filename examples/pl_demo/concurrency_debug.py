# Minimal Predictive Race Detection Demo
from dataclasses import dataclass

@dataclass
class Event:
    thread_id: int
    op: str  # 'read', 'write'
    var: str

class RaceDetector:
    def __init__(self):
        self.events = []
    def add(self, e):
        self.events.append(e)
    def detect(self):
        races = []
        for i, e1 in enumerate(self.events):
            for e2 in self.events[i+1:]:
                if e1.var == e2.var and e1.thread_id != e2.thread_id and ('write' in (e1.op, e2.op)):
                    races.append((e1, e2))
        return races

def demo():
    d = RaceDetector()
    d.add(Event(1, 'write', 'x'))
    d.add(Event(2, 'read', 'x'))
    d.add(Event(1, 'read', 'y'))
    d.add(Event(2, 'write', 'y'))
    for i, (e1, e2) in enumerate(d.detect()):
        print(f"Race {i+1}: Thread {e1.thread_id} {e1.op} {e1.var} <-> Thread {e2.thread_id} {e2.op} {e2.var}")

if __name__ == "__main__":
    demo() 
    