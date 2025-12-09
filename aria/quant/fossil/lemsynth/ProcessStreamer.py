import time
import multiprocessing
import subprocess
import psutil
import warnings

from aria.quant.fossil.lemsynth.utils import StopProposal


class Timeout(Exception):
    pass


_process_timeout = 'ProcessTimeout'


class ProcessStreamer:
    """
    TODO: Write documentation
    """

    def __init__(self, cmdlist, timeout=None, logfile=None):
        self.cmdlist = cmdlist
        # Recommended to compute specialised log file name because it will act as a shared memory
        # File as shared memory is used to implement non-blocking reads
        if logfile is None:
            self._logfile = '{}.stream'.format(''.join([entry if isinstance(entry, str) else str(entry)
                                                        for entry in cmdlist]))
        else:
            self._logfile = logfile
        self._logfile_handle = None
        self.writer_proc = None
        self._writer_proc_pausable = None
        self.timer_proc = None
        # If timeout is None then no timeout is enforced
        self.timeout = timeout
        # Only one generator stream can be present at any point. This is a pragmatic and simplistic design decision.
        # In general, maintain count of active streams
        self._curr_stream = None

    def prologue(self):
        logfile_handle = open(self._logfile, 'w')
        self._logfile_handle = logfile_handle
        self.writer_proc = subprocess.Popen(self.cmdlist, stdout=logfile_handle, bufsize=0)
        self._writer_proc_pausable = psutil.Process(pid=self.writer_proc.pid)
        if self.timeout is not None:
            self.timer_proc = multiprocessing.Process(name='timer', target=self._timer, args=(self.timeout,))
            self.timer_proc.start()

    def stream(self, busywaiting=True, lazystream=False):
        # Only one generator stream can be present at any point in the present design.
        if self._curr_stream is not None:
            raise RuntimeError('Generator already initialised for object. '
                               'Close stale generator using .close_stream()')
        # infinite mode allowed only when there is no timeout
        if self.timeout is not None and lazystream:
            warnings.warn('Lazy streaming with timeout is not a usual option setting. '
                          'Consider setting the option that reflects your purpose best.')
        if lazystream:
            busywaiting = True
        # Declare and start processes.
        # In general, perform only when count of active streams is 0.
        self.prologue()
        # Set active stream. In general, increase count by 1.
        self._curr_stream = self._stream(busywaiting=busywaiting, lazystream=lazystream)
        # Return initialised generator
        return self._curr_stream

    def _stream(self, busywaiting, lazystream):
        currpos = 0
        while True:
            with open(self._logfile, 'r') as f:
                f.seek(currpos)
                text = f.readline()
                currpos = f.tell()
            try:
                if text == '':
                    # If busy waiting is enabled then try again after sometime.
                    if busywaiting:
                        time.sleep(2)
                        continue
                    # No content: yield None and optionally get input on waiting for the next solution.
                    yield None
                elif text == _process_timeout:
                    raise Timeout
                else:
                    # If infinite mode, pause writer proc before yielding and resume after
                    if lazystream:
                        self._writer_proc_pausable.suspend()
                    yield text
                    if lazystream:
                        self._writer_proc_pausable.resume()
            except (StopProposal, Timeout) as e:
                # No more active streams for the current object
                # In general, decrease count of active streams by one
                self._curr_stream = None
                # Epilogue computations
                # In general, perform when count of active streams hits zero
                self.epilogue()
                if isinstance(e, Timeout):
                    # If the stream has timed out then bubble the exception up the call stack
                    raise Timeout
                else:
                    return None

    def epilogue(self):
        if self.writer_proc.poll() is None:
            self.writer_proc.kill()
            # Make sure to wait otherwise zombie processes can clog up system
            self.writer_proc.wait()
        # multiprocessing 'processes' need to be killed differently: check if alive first
        # Last condition is added owing to a weird bug in multiprocessing that shows a finished process as alive
        if self.timeout is not None and self.timer_proc.is_alive() and self.timer_proc._popen is not None:
            self.timer_proc.terminate()

    # Auxiliary function to create a timer process
    def _timer(self, secs):
        time.sleep(secs)
        # Kill writer proc
        if self.writer_proc.poll() is None:
            self.writer_proc.kill()
            self.writer_proc.wait()
        with open(self._logfile, 'a+') as f:
            # Write newline if not already present.
            f.seek(f.tell() - 1)
            lastchar = f.read()
            if lastchar != '\n':
                f.write('\n')
            # Write process timeout signature string.
            # Doing this makes the logfile a shared memory.
            f.write(_process_timeout)
        return self.epilogue()
