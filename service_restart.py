"""A hidden, bounded local restart helper. Never kills unrelated processes."""
import pathlib, subprocess, sys, time, urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
def main():
    # Wait for the previous listener to close gracefully, then launch once.
    for _ in range(150):
        try:
            urllib.request.urlopen('http://127.0.0.1:8769/api/status',timeout=.4).close()
        except (OSError,TimeoutError):break
        time.sleep(.2)
    else:return
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'DETACHED_PROCESS',0)
    with (ROOT/'server.out.log').open('ab') as out,(ROOT/'server.err.log').open('ab') as err:
        process=subprocess.Popen([sys.executable,'-X','utf8',str(ROOT/'server.py')],cwd=ROOT,
            creationflags=flags,stdin=subprocess.DEVNULL,stdout=out,stderr=err,close_fds=True)
    (ROOT/'server.pid').write_text(str(process.pid),encoding='ascii')
if __name__=='__main__':main()
