import sys

if sys.platform == "win32":
    import win32event
    import win32con
    import pywintypes
    import win32api

    class CrossPlatformNamedSemaphore:
        def __init__(self, name, flags=None, initial_value=1):
            self._name = name.lstrip("/") if name else None
            self._closed = False
            self._handle = None
            try:
                self._handle = win32event.CreateSemaphore(
                    None, initial_value, 0x7FFFFFFF, self._name
                )
                self._created = win32api.GetLastError() != win32con.ERROR_ALREADY_EXISTS
            except pywintypes.error as e:
                if e.winerror == win32con.ERROR_ALREADY_EXISTS:
                    self._handle = win32event.OpenSemaphore(
                        win32con.SEMAPHORE_ALL_ACCESS, False, self._name
                    )
                    self._created = False
                else:
                    raise

        def release(self):
            win32event.ReleaseSemaphore(self._handle, 1)

        def acquire(self, timeout=None):
            if timeout is None:
                timeout_ms = win32con.INFINITE
            else:
                timeout_ms = int(timeout * 1000)
            rc = win32event.WaitForSingleObject(self._handle, timeout_ms)
            if rc == win32con.WAIT_OBJECT_0:
                return True
            elif rc == win32con.WAIT_TIMEOUT:
                raise BusyError("Semaphore acquire timed out")
            else:
                raise RuntimeError("Semaphore wait failed")

        def close(self):
            if not self._closed and self._handle:
                win32api.CloseHandle(self._handle)
                self._closed = True

        def unlink(self):
            pass

    Semaphore = CrossPlatformNamedSemaphore

    class BusyError(Exception):
        pass

else:
    import posix_ipc

    Semaphore = posix_ipc.Semaphore
    BusyError = posix_ipc.BusyError
