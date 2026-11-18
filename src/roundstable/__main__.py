import argparse,importlib,json,math,sys,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from . import api,__version__

def examples():return api.examples() if hasattr(api,'examples') else api.EXAMPLES

def safe(value):
    if isinstance(value,float) and not math.isfinite(value):return 'NaN' if math.isnan(value) else 'Infinity' if value>0 else '-Infinity'
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    return value

def demo():return {name:safe(api.OPERATIONS[name](data)) for name,data in examples().items()}

class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address):self.slots=threading.BoundedSemaphore(4);super().__init__(address,Handler)
class Handler(BaseHTTPRequestHandler):
    server_version='DataComponent/0.1'
    def log_message(self,*args):pass
    def reply(self,status,payload):
        data=json.dumps(safe(payload),ensure_ascii=False,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_GET(self):
        if self.path=='/health':return self.reply(200,{'status':'ok','version':__version__})
        if self.path=='/capabilities':return self.reply(200,{'operations':sorted(api.OPERATIONS)})
        self.reply(404,{'error':'route not found'})
    def do_POST(self):
        if not self.path.startswith('/v1/') or self.path[4:] not in api.OPERATIONS:return self.reply(404,{'error':'unknown operation'})
        self.connection.settimeout(15)
        if not self.server.slots.acquire(blocking=False):return self.reply(429,{'error':'request capacity reached'})
        try:
            if self.headers.get('Transfer-Encoding'):raise ValueError('chunked bodies are unsupported')
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=1_000_000:return self.reply(413,{'error':'body size limit'})
            raw=self.rfile.read(n)
            if len(raw)!=n:raise ValueError('incomplete body')
            data=json.loads(raw)
            if not isinstance(data,dict):raise ValueError('JSON object required')
            result=api.OPERATIONS[self.path[4:]](data);self.reply(200,{'result':result})
        except Exception as e:self.reply(400,{'error':str(e)[:400],'kind':type(e).__name__})
        finally:self.server.slots.release()

def main():
    parser=argparse.ArgumentParser(description='Data component library service')
    parser.add_argument('action',choices=['demo','serve']);parser.add_argument('--port',type=int,default=8080);args=parser.parse_args()
    if args.action=='demo':print(json.dumps(demo(),ensure_ascii=False,allow_nan=False));return
    if not 0<=args.port<=65535:parser.error('port outside valid range')
    server=Server(('127.0.0.1',args.port))
    print(json.dumps({'listening':'127.0.0.1','port':server.server_port}),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
if __name__=='__main__':main()
