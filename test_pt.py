from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "llama3", "llama2", "mixtral-8x7b", "claude-3-opus"]
completer = WordCompleter(models, ignore_case=True, match_middle=True)

try:
    val = prompt("Select model: ", completer=completer)
    print("Selected:", val)
except Exception as e:
    print(e)
