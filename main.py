import argparse
import fileinput
import functools
import sys
import openai
import os
parser = argparse.ArgumentParser()
parser.add_argument('--engine', default='ada')
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--max_tokens', type=int, default=64)
parser.add_argument('--top_p', type=float, default=1.0)
parser.add_argument('--frequency_penalty', type=float, default=0.0)
parser.add_argument('--presence_penalty', type=float, default=0.0)
parser.add_argument('--stop', type=str)
args = parser.parse_args()

stop=None
if args.stop != None:
    stop = [args.stop]

openai.api_key = os.getenv("OPENAI_API_KEY")

engines = []
fd = os.open('.engine_cache', os.O_CREAT | os.O_RDWR)
if fd != -1:
    with os.fdopen(fd, 'r') as f:
        engines = f.read().strip().split('\n')
fd = os.open('.engine_cache', os.O_CREAT | os.O_RDWR)
if 'davinci' not in engines:
    engines = openai.Engine.list()
    with os.fdopen(fd, 'w') as f:
        engines = [engine.id for engine in engines.data]
        f.write(('\n').join(engines))

if args.engine not in engines:
    print(('\n').join(engines))
    sys.exit(1)


prompt = ""
for line in sys.stdin:
    prompt += line
prompt = prompt.strip()
response = openai.Completion.create(
  engine=args.engine,
  prompt=prompt,
  temperature=args.temperature,
  max_tokens=args.max_tokens,
  top_p=args.top_p,
  frequency_penalty=args.frequency_penalty,
  presence_penalty=args.presence_penalty,
  stop=stop
)

print(response.choices[0].text)

