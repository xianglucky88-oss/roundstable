import json
from . import api

def main():
    values=api.examples() if hasattr(api,'examples') else api.EXAMPLES
    print(json.dumps(values,ensure_ascii=False))

if __name__=='__main__':main()
